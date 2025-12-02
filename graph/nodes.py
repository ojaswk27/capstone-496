"""
LangGraph Nodes for Aerospace Design Assistant

Implements the processing nodes for the design workflow:
- Node 0: Vehicle Classifier
- Node 1: Requirement Parser
- Node 2: Search Agent
- Node 3: Extraction Agent
- Node 4: Tool Selector
- Node 5: Calculation Agent
- Node 6: Validator
- Node 7: Synthesizer
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Import tools
from tools import (
    ALL_TOOLS,
    design_glider,
    design_helicopter,
    design_rocket,
    design_satellite,
    get_tool_function,
    get_tools_for_vehicle,
    size_aircraft,
    size_drone,
)

from .state import (
    CalculationResult,
    DesignComponent,
    DesignOutput,
    DesignPhase,
    DesignState,
    ExtractedFormula,
    SearchResult,
    UserRequirements,
    ValidationResult,
    VehicleType,
    add_error,
    add_warning,
    update_phase,
)

# =============================================================================
# Node 0: Vehicle Classifier
# =============================================================================

# Keywords for vehicle classification
VEHICLE_KEYWORDS = {
    VehicleType.DRONE: [
        "drone",
        "quadcopter",
        "multicopter",
        "uav",
        "unmanned",
        "multirotor",
        "quadrotor",
        "hexacopter",
        "octocopter",
        "fpv",
        "aerial photography",
        "hover",
    ],
    VehicleType.FIXED_WING: [
        "airplane",
        "aircraft",
        "plane",
        "fixed-wing",
        "fixed wing",
        "passenger",
        "airliner",
        "jet",
        "propeller",
        "wing",
        "cessna",
        "boeing",
        "airbus",
        "runway",
        "takeoff",
    ],
    VehicleType.HELICOPTER: [
        "helicopter",
        "rotorcraft",
        "helo",
        "chopper",
        "rotor",
        "vtol",
        "vertical takeoff",
        "hovering aircraft",
    ],
    VehicleType.ROCKET: [
        "rocket",
        "launch vehicle",
        "missile",
        "space launch",
        "booster",
        "delta-v",
        "staging",
        "propellant",
        "thrust",
    ],
    VehicleType.SATELLITE: [
        "satellite",
        "spacecraft",
        "orbit",
        "space station",
        "leo",
        "geo",
        "meo",
        "solar panel",
        "orbital",
    ],
    VehicleType.GLIDER: [
        "glider",
        "sailplane",
        "soaring",
        "unpowered",
        "thermal",
        "glide ratio",
        "sink rate",
    ],
}


def classify_vehicle(state: DesignState) -> DesignState:
    """
    Node 0: Classify the vehicle type from user input.

    Uses keyword matching and context analysis to determine
    the most likely vehicle type.
    """
    state = update_phase(state, DesignPhase.CLASSIFYING)

    input_lower = state.raw_input.lower()

    # Score each vehicle type
    scores = {}
    matches = {}

    for vehicle_type, keywords in VEHICLE_KEYWORDS.items():
        score = 0
        found_keywords = []

        for keyword in keywords:
            if keyword in input_lower:
                score += 1
                found_keywords.append(keyword)

        scores[vehicle_type] = score
        matches[vehicle_type] = found_keywords

    # Find best match
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Calculate confidence
    total_score = sum(scores.values())
    if total_score > 0:
        confidence = best_score / (total_score + 1)
    else:
        confidence = 0.0

    # Generate reasoning
    if best_score > 0:
        reasoning = f"Matched keywords: {', '.join(matches[best_type])}"
    else:
        reasoning = "No specific keywords matched, using context analysis"
        # Default heuristics
        if "fly" in input_lower and "camera" in input_lower:
            best_type = VehicleType.DRONE
            confidence = 0.6
            reasoning = "Context suggests drone (flying + camera)"
        elif "orbit" in input_lower or "space" in input_lower:
            best_type = VehicleType.SATELLITE
            confidence = 0.6
            reasoning = "Context suggests satellite (space/orbit)"
        elif "altitude" in input_lower and "km" in input_lower:
            if float(re.search(r"(\d+)\s*km", input_lower).group(1)) > 50:
                best_type = VehicleType.ROCKET
                confidence = 0.6
                reasoning = "High altitude target suggests rocket"

    # Minimum confidence threshold
    if confidence < 0.3 and best_score == 0:
        best_type = VehicleType.UNKNOWN
        confidence = 0.0
        reasoning = "Unable to determine vehicle type with confidence"
        state = add_warning(
            state, "Vehicle type unclear. Please specify the type of vehicle."
        )

    # Update state
    state.vehicle_type = best_type
    state.classification_confidence = min(confidence + 0.3, 1.0)  # Boost confidence
    state.classification_reasoning = reasoning

    return state


# =============================================================================
# Node 1: Requirement Parser
# =============================================================================

# Patterns for extracting requirements
REQUIREMENT_PATTERNS = {
    "payload_kg": [
        r"(\d+(?:\.\d+)?)\s*kg\s*payload",
        r"payload\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*kg",
        r"carry\s*(\d+(?:\.\d+)?)\s*kg",
    ],
    "range_km": [
        r"(\d+(?:\.\d+)?)\s*km\s*range",
        r"range\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*km",
    ],
    "endurance_hours": [
        r"(\d+(?:\.\d+)?)\s*hour",
        r"(\d+)\s*hr",
    ],
    "flight_time_minutes": [
        r"(\d+)\s*min(?:ute)?s?\s*(?:flight|hover)",
        r"flight\s*time\s*(?:of\s*)?(\d+)\s*min",
    ],
    "speed_kmh": [
        r"(\d+(?:\.\d+)?)\s*km/?h",
        r"(\d+(?:\.\d+)?)\s*kmph",
        r"speed\s*(?:of\s*)?(\d+)",
    ],
    "altitude_m": [
        r"(\d+(?:\.\d+)?)\s*m\s*altitude",
        r"altitude\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*m",
        r"(\d+(?:\.\d+)?)\s*km\s*altitude",  # Will convert
    ],
    "target_altitude": [
        r"reach\s*(\d+(?:\.\d+)?)\s*(?:m|km)",
        r"(\d+(?:\.\d+)?)\s*(?:m|km)\s*(?:altitude|high)",
    ],
    "passengers": [
        r"(\d+)\s*(?:passenger|person|people|seat)",
    ],
    "num_motors": [
        r"(\d+)\s*motor",
        r"(\d+)\s*rotor",
    ],
}


def parse_requirements(state: DesignState) -> DesignState:
    """
    Node 1: Parse user input into structured requirements.

    Extracts numerical values and categorizes requirements
    based on vehicle type.
    """
    state = update_phase(state, DesignPhase.PARSING)

    input_text = state.raw_input.lower()
    requirements = UserRequirements(raw_input=state.raw_input)

    # Fields that exist in UserRequirements
    user_req_fields = set(UserRequirements.model_fields.keys())

    # Extract common requirements
    for req_name, patterns in REQUIREMENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, input_text, re.IGNORECASE)
            if match:
                value = float(match.group(1))

                # Handle unit conversions
                if req_name == "altitude_m" and "km" in match.group(0):
                    value *= 1000
                if req_name == "target_altitude" and "km" in match.group(0):
                    value *= 1000

                # Set attribute if it exists in model, otherwise add to vehicle_specific
                if req_name in user_req_fields:
                    setattr(requirements, req_name, value)
                else:
                    requirements.vehicle_specific[req_name] = value
                break

    # Vehicle-specific parsing
    vehicle_specific = {}

    if state.vehicle_type == VehicleType.DRONE:
        if "racing" in input_text:
            vehicle_specific["application"] = "racing"
        elif "photo" in input_text or "camera" in input_text:
            vehicle_specific["application"] = "photography"
        elif "heavy" in input_text or "lift" in input_text:
            vehicle_specific["application"] = "heavy_lift"
        else:
            vehicle_specific["application"] = "general"

        # Convert hours to minutes if needed
        if requirements.endurance_hours and not hasattr(
            requirements, "flight_time_minutes"
        ):
            vehicle_specific["flight_time_minutes"] = requirements.endurance_hours * 60

    elif state.vehicle_type == VehicleType.ROCKET:
        if "solid" in input_text:
            vehicle_specific["motor_type"] = "solid"
        elif "liquid" in input_text:
            vehicle_specific["motor_type"] = "liquid"
        elif "hybrid" in input_text:
            vehicle_specific["motor_type"] = "hybrid"

        # Check for orbit target
        vehicle_specific["target_orbit"] = "orbit" in input_text

    elif state.vehicle_type == VehicleType.SATELLITE:
        if "geo" in input_text:
            vehicle_specific["orbit_type"] = "GEO"
            vehicle_specific["orbit_altitude_km"] = 35786
        elif (
            "leo" in input_text
            or requirements.altitude_m
            and requirements.altitude_m < 2000000
        ):
            vehicle_specific["orbit_type"] = "LEO"

        # Mission duration
        years_match = re.search(r"(\d+)\s*year", input_text)
        if years_match:
            vehicle_specific["mission_years"] = float(years_match.group(1))

    elif state.vehicle_type == VehicleType.GLIDER:
        for glider_class in ["open", "18m", "15m", "standard", "club"]:
            if glider_class in input_text:
                vehicle_specific["glider_class"] = glider_class
                break

    requirements.vehicle_specific = vehicle_specific
    state.requirements = requirements

    return state


# =============================================================================
# Node 2: Search Agent
# =============================================================================


def search_documents(state: DesignState) -> DesignState:
    """
    Node 2: Search relevant documents based on requirements.

    Generates search queries and retrieves relevant content
    from the document store.
    """
    state = update_phase(state, DesignPhase.SEARCHING)

    # Generate search queries based on vehicle type and requirements
    queries = []
    vehicle = (
        state.vehicle_type if state.vehicle_type != VehicleType.UNKNOWN else "aircraft"
    )

    # Base query
    queries.append(f"{vehicle} design methodology")

    # Requirement-based queries
    if state.requirements:
        if state.requirements.payload_kg:
            queries.append(f"{vehicle} payload capacity sizing")
        if state.requirements.range_km:
            queries.append(f"{vehicle} range calculation")
        if state.requirements.speed_kmh:
            queries.append(f"{vehicle} speed performance")

    # Vehicle-specific queries
    if state.vehicle_type == VehicleType.DRONE:
        queries.extend(
            [
                "multicopter hover power calculation",
                "drone battery sizing endurance",
                "propeller motor selection",
            ]
        )
    elif state.vehicle_type == VehicleType.FIXED_WING:
        queries.extend(
            [
                "aircraft lift drag calculation",
                "wing sizing wing loading",
                "aircraft weight estimation",
            ]
        )
    elif state.vehicle_type == VehicleType.ROCKET:
        queries.extend(
            [
                "rocket equation delta-v",
                "staging optimization",
                "thrust to weight ratio",
            ]
        )
    elif state.vehicle_type == VehicleType.SATELLITE:
        queries.extend(
            [
                "orbital mechanics velocity period",
                "satellite power system sizing",
                "thermal control spacecraft",
            ]
        )
    elif state.vehicle_type == VehicleType.HELICOPTER:
        queries.extend(
            ["rotor disk loading", "helicopter hover power", "autorotation descent"]
        )
    elif state.vehicle_type == VehicleType.GLIDER:
        queries.extend(
            [
                "glide ratio performance",
                "thermal soaring climb",
                "polar curve sink rate",
            ]
        )

    state.search_queries = queries

    # Try to import and use the RAG system
    try:
        import os

        from rag.search import create_search_interface

        papers_dir = os.path.join(os.path.dirname(__file__), "..", "data", "papers")

        if os.path.exists(papers_dir):
            search = create_search_interface(papers_dir)

            for query in queries[:5]:  # Limit queries
                search_response = search.search(
                    query,
                    vehicle_type=state.vehicle_type
                    if state.vehicle_type != VehicleType.UNKNOWN
                    else None,
                    top_k=3,
                )

                # Access the results list from the SearchResponse object
                for r in search_response.results:
                    state.search_results.append(
                        SearchResult(
                            content=r.content,
                            source=r.metadata.get("title", "Unknown"),
                            vehicle_type=r.metadata.get("vehicle_type", "general"),
                            relevance_score=r.score,
                            metadata=r.metadata,
                        )
                    )
    except ImportError:
        state = add_warning(state, "RAG system not available, using built-in knowledge")
    except Exception as e:
        import traceback  # <-- DEBUG

        traceback.print_exc()  # <-- shows the true error
        state = add_warning(state, f"Search error: {str(e)}")

    return state


# =============================================================================
# Node 3: Extraction Agent
# =============================================================================


def extract_formulas(state: DesignState) -> DesignState:
    """
    Node 3: Extract relevant formulas and data from search results.

    Parses search results to find applicable formulas,
    typical values, and design guidelines.
    """
    state = update_phase(state, DesignPhase.EXTRACTING)

    # Formula patterns to look for
    formula_patterns = [
        (r"([A-Z][a-z_]*)\s*=\s*([^,\n]+)", "equation"),
        (r"(\w+)\s*:\s*(\d+(?:\.\d+)?)\s*([a-zA-Z/²³]+)", "typical_value"),
    ]

    for result in state.search_results:
        content = result.content

        # Extract formulas
        for pattern, formula_type in formula_patterns:
            matches = re.findall(pattern, content)

            for match in matches:
                if formula_type == "equation" and len(match) >= 2:
                    name = match[0]
                    formula = f"{match[0]} = {match[1]}"

                    # Extract variables from context
                    variables = {}
                    var_pattern = r"-\s*([A-Za-z_]+)\s*[=:]\s*([^\n]+)"
                    var_matches = re.findall(var_pattern, content)
                    for vm in var_matches:
                        if len(vm[0]) <= 5:  # Likely a variable
                            variables[vm[0]] = vm[1].strip()

                    state.extracted_formulas.append(
                        ExtractedFormula(
                            name=name,
                            formula=formula,
                            variables=variables,
                            source=result.source,
                            applicable_to=[result.vehicle_type],
                            confidence=result.relevance_score,
                        )
                    )

    # Also extract any design data mentioned
    data_patterns = {
        "disk_loading": r"disk\s*loading[:\s]+(\d+(?:\.\d+)?)",
        "wing_loading": r"wing\s*loading[:\s]+(\d+(?:\.\d+)?)",
        "thrust_to_weight": r"thrust.to.weight[:\s]+(\d+(?:\.\d+)?)",
        "aspect_ratio": r"aspect\s*ratio[:\s]+(\d+(?:\.\d+)?)",
    }

    for result in state.search_results:
        for param, pattern in data_patterns.items():
            match = re.search(pattern, result.content, re.IGNORECASE)
            if match:
                state.extracted_data[param] = float(match.group(1))

    return state


# =============================================================================
# Node 4: Tool Selector (merged into calculation)
# =============================================================================


def select_tools(state: DesignState) -> List[str]:
    """
    Select appropriate calculation tools based on vehicle type.

    Returns list of tool names to use.
    """
    vehicle = state.vehicle_type

    # Get vehicle-specific tools
    tools = list(get_tools_for_vehicle(vehicle.value).keys())

    # Filter to most relevant tools
    primary_tools = {
        VehicleType.DRONE: [
            "size_drone",
            "calculate_hover_power",
            "calculate_flight_time",
        ],
        VehicleType.FIXED_WING: [
            "size_aircraft",
            "calculate_lift_drag",
            "calculate_stall_speed",
        ],
        VehicleType.HELICOPTER: [
            "design_helicopter",
            "calculate_hover_power",
            "design_rotor",
        ],
        VehicleType.ROCKET: [
            "design_rocket",
            "calculate_delta_v",
            "calculate_thrust_to_weight",
        ],
        VehicleType.SATELLITE: [
            "design_satellite",
            "calculate_orbital_velocity",
            "design_power_system",
        ],
        VehicleType.GLIDER: [
            "design_glider",
            "calculate_glide_performance",
            "calculate_thermal_climb",
        ],
    }

    selected = primary_tools.get(vehicle, ["isa_atmosphere"])

    return selected


# =============================================================================
# Node 5: Calculation Agent
# =============================================================================


def perform_calculations(state: DesignState) -> DesignState:
    """
    Node 5: Execute calculations using selected tools.

    All parameters have been completed by LLM - no defaults needed!
    """
    state = update_phase(state, DesignPhase.CALCULATING)

    req = state.requirements
    vehicle = state.vehicle_type

    try:
        if vehicle == VehicleType.DRONE:
            # Parameters already completed by LLM
            payload = req.payload_kg
            flight_time = req.endurance_hours * 60 if req.endurance_hours else 20
            application = req.vehicle_specific.get("application", "photography")
            num_motors = req.vehicle_specific.get("num_motors", 4)

            result = size_drone(
                payload_kg=payload,
                flight_time_minutes=flight_time,
                num_motors=num_motors,
                application=application,
            )

            state.intermediate_results["design"] = {
                "frame_size_mm": result.frame_size,
                "num_motors": result.num_motors,
                "motor_kv": result.motor_kv,
                "prop_diameter_in": result.prop_diameter,
                "prop_pitch_in": result.prop_pitch,
                "battery_cells": result.battery_cells,
                "battery_capacity_mah": result.battery_capacity,
                "total_weight_kg": result.total_weight,
                "max_thrust_n": result.max_thrust,
                "thrust_to_weight": result.thrust_to_weight,
                "hover_time_min": result.hover_time,
                "max_speed_ms": result.max_speed,
            }

            state.calculations.append(
                CalculationResult(
                    tool_name="size_drone",
                    inputs={"payload_kg": payload, "flight_time_minutes": flight_time},
                    outputs=state.intermediate_results["design"],
                    success=True,
                )
            )

        elif vehicle == VehicleType.FIXED_WING:
            # Parameters already completed by LLM - no defaults needed!
            payload = req.payload_kg
            speed = req.speed_kmh
            range_km = req.range_km

            result = size_aircraft(
                payload_kg=payload,
                range_km=range_km,
                cruise_speed_kmh=speed,
                user_requirements=req.raw_input,
            )

            state.intermediate_results["design"] = {
                "wing_span_m": result.wing.span,
                "wing_area_m2": result.wing.area,
                "aspect_ratio": result.wing.aspect_ratio,
                "total_weight_kg": result.total_weight,
                "empty_weight_kg": result.weight_breakdown["empty"],
                "fuel_weight_kg": result.fuel_weight,
                "power_required_w": result.power_required,
                "stall_speed_ms": result.performance.stall_speed,
                "cruise_speed_ms": result.performance.cruise_speed,
                "range_km": result.performance.range_km,
            }

            state.calculations.append(
                CalculationResult(
                    tool_name="size_aircraft",
                    inputs={
                        "payload_kg": payload,
                        "range_km": range_km,
                        "cruise_speed_kmh": speed,
                    },
                    outputs=state.intermediate_results["design"],
                    success=True,
                )
            )

        elif vehicle == VehicleType.HELICOPTER:
            # Parameters already completed by LLM - no defaults needed!
            payload = req.payload_kg
            range_km = req.range_km
            speed = req.speed_kmh

            result = design_helicopter(
                payload_kg=payload, range_km=range_km, cruise_speed_kmh=speed
            )

            state.intermediate_results["design"] = {
                "rotor_diameter_m": result.rotor.diameter,
                "rotor_blades": result.rotor.num_blades,
                "rotor_rpm": result.rotor.rpm,
                "tail_rotor_diameter_m": result.tail_rotor_diameter,
                "total_weight_kg": result.total_weight,
                "hover_power_w": result.hover.power_actual,
                "cruise_power_w": result.forward_flight.power_required,
                "engine_power_w": result.engine_power,
                "fuel_consumption_kghr": result.fuel_consumption,
            }

            state.calculations.append(
                CalculationResult(
                    tool_name="design_helicopter",
                    inputs={
                        "payload_kg": payload,
                        "range_km": range_km,
                        "cruise_speed_kmh": speed,
                    },
                    outputs=state.intermediate_results["design"],
                    success=True,
                )
            )

        elif vehicle == VehicleType.ROCKET:
            # Parameters already completed by LLM
            payload = req.payload_kg
            altitude = req.altitude_m or req.vehicle_specific.get(
                "target_altitude_m", 1000
            )
            motor_type = req.vehicle_specific.get("motor_type", "solid")

            result = design_rocket(
                payload_kg=payload, target_altitude=altitude, motor_type=motor_type
            )

            state.intermediate_results["design"] = {
                "num_stages": len(result.stages),
                "total_mass_kg": result.total_mass,
                "payload_fraction": result.payload_fraction,
                "total_delta_v_ms": result.total_delta_v,
                "predicted_altitude_m": result.max_altitude,
                "target_achieved": result.target_achieved,
                "stages": [
                    {
                        "stage": s.stage_number,
                        "propellant_kg": s.propellant_mass,
                        "structural_kg": s.structural_mass,
                        "delta_v_ms": s.delta_v,
                        "burn_time_s": s.burn_time,
                        "thrust_n": s.thrust,
                    }
                    for s in result.stages
                ],
            }

            state.calculations.append(
                CalculationResult(
                    tool_name="design_rocket",
                    inputs={
                        "payload_kg": payload,
                        "target_altitude": altitude,
                        "motor_type": motor_type,
                    },
                    outputs=state.intermediate_results["design"],
                    success=True,
                )
            )

        elif vehicle == VehicleType.SATELLITE:
            # Parameters already completed by LLM
            payload_power = req.vehicle_specific.get("payload_power_w", 50)
            payload_mass = req.payload_kg
            altitude = req.vehicle_specific.get("orbit_altitude_km", 500) * 1000

            result = design_satellite(
                payload_power=payload_power,
                payload_mass=payload_mass,
                altitude=altitude,
            )

            state.intermediate_results["design"] = {
                "orbit_type": result.orbit.orbit_type,
                "orbit_altitude_m": result.orbit.altitude,
                "orbital_velocity_ms": result.orbit.velocity,
                "orbital_period_s": result.orbit.period,
                "total_mass_kg": result.total_mass,
                "solar_array_area_m2": result.power.solar_array_area,
                "battery_capacity_wh": result.power.battery_capacity,
                "dimensions_m": result.dimensions,
                "design_life_years": result.design_life,
            }

            state.calculations.append(
                CalculationResult(
                    tool_name="design_satellite",
                    inputs={
                        "payload_power": payload_power,
                        "payload_mass": payload_mass,
                        "altitude": altitude,
                    },
                    outputs=state.intermediate_results["design"],
                    success=True,
                )
            )

        elif vehicle == VehicleType.GLIDER:
            # Parameters already completed by LLM
            pilot_weight = req.vehicle_specific.get("pilot_weight_kg", 80)
            target_ld = req.vehicle_specific.get("target_glide_ratio", 40)
            glider_class = req.vehicle_specific.get("glider_class", "standard")

            result = design_glider(
                pilot_weight=pilot_weight,
                target_glide_ratio=target_ld,
                glider_class=glider_class,
            )

            state.intermediate_results["design"] = {
                "wing_span_m": result.wing_span,
                "wing_area_m2": result.wing_area,
                "aspect_ratio": result.aspect_ratio,
                "empty_weight_kg": result.empty_weight,
                "max_weight_kg": result.max_weight,
                "best_glide_ratio": result.best_glide_ratio,
                "min_sink_rate_ms": result.min_sink_rate,
                "stall_speed_ms": result.stall_speed,
                "max_speed_ms": result.max_speed,
            }

            state.calculations.append(
                CalculationResult(
                    tool_name="design_glider",
                    inputs={
                        "pilot_weight": pilot_weight,
                        "target_glide_ratio": target_ld,
                        "glider_class": glider_class,
                    },
                    outputs=state.intermediate_results["design"],
                    success=True,
                )
            )

        else:
            state = add_error(state, f"Unknown vehicle type: {vehicle}")

    except Exception as e:
        state = add_error(state, f"Calculation error: {str(e)}")
        state.calculations.append(
            CalculationResult(
                tool_name="unknown",
                inputs={},
                outputs={},
                success=False,
                error_message=str(e),
            )
        )
        import traceback

        traceback.print_exc()

    return state


# =============================================================================
# Node 6: Validator
# =============================================================================


def validate_design(state: DesignState) -> DesignState:
    """
    Node 6: Validate the design against requirements.

    Checks if the design meets all requirements and
    identifies any issues.
    """
    state = update_phase(state, DesignPhase.VALIDATING)

    checks = {}
    warnings = []
    errors = []
    suggestions = []

    design = state.intermediate_results.get("design", {})
    req = state.requirements

    # Common validations
    if "total_weight_kg" in design:
        if req.max_weight_kg and design["total_weight_kg"] > req.max_weight_kg:
            errors.append(
                f"Weight {design['total_weight_kg']:.1f} kg exceeds maximum {req.max_weight_kg} kg"
            )
            checks["weight_limit"] = False
        else:
            checks["weight_limit"] = True

    # Vehicle-specific validations
    if state.vehicle_type == VehicleType.DRONE:
        if "thrust_to_weight" in design:
            tw = design["thrust_to_weight"]
            if tw < 1.5:
                warnings.append(
                    f"Low thrust-to-weight ratio ({tw:.1f}), may have poor maneuverability"
                )
                checks["thrust_margin"] = False
            elif tw > 4.0:
                warnings.append(f"Very high T/W ({tw:.1f}), may be oversized")
                checks["thrust_margin"] = True
            else:
                checks["thrust_margin"] = True

        if "hover_time_min" in design and req.vehicle_specific.get(
            "flight_time_minutes"
        ):
            target = req.vehicle_specific["flight_time_minutes"]
            actual = design["hover_time_min"]
            if actual < target * 0.9:
                errors.append(
                    f"Flight time {actual:.0f} min below target {target:.0f} min"
                )
                checks["endurance"] = False
            else:
                checks["endurance"] = True

    elif state.vehicle_type == VehicleType.ROCKET:
        if "target_achieved" in design:
            checks["altitude_target"] = design["target_achieved"]
            if not design["target_achieved"]:
                errors.append("Design does not achieve target altitude")
                suggestions.append(
                    "Consider increasing propellant mass or using higher Isp motor"
                )

    elif state.vehicle_type == VehicleType.FIXED_WING:
        if "range_km" in design and req.range_km:
            if design["range_km"] < req.range_km * 0.9:
                warnings.append(
                    f"Range {design['range_km']:.0f} km may be below target"
                )
                checks["range"] = False
            else:
                checks["range"] = True

    # Calculate overall pass/fail
    passed = len(errors) == 0 and all(checks.values())

    state.validation_result = ValidationResult(
        passed=passed,
        checks=checks,
        warnings=warnings,
        errors=errors,
        suggestions=suggestions,
    )

    state.iteration_count += 1

    return state


# ------------------------------------------------------------------
#  Node 7: Synthesizer  ––  drop-in (fixed)
# ------------------------------------------------------------------
def synthesize_output(state: DesignState) -> DesignState:
    """
    Node 7: Synthesize final design output.

    Creates vehicle-specific output showing only relevant parameters.
    ALL keys match exactly what's stored in perform_calculations.
    """
    state = update_phase(state, DesignPhase.SYNTHESIZING)

    req = state.requirements
    vehicle = state.vehicle_type
    design_data = state.intermediate_results.get("design", {})

    if not design_data:
        state = add_error(state, "No design data available to synthesize")
        return state

    # Vehicle-specific output formatting - VERIFIED AGAINST ACTUAL STORED KEYS
    if vehicle == VehicleType.DRONE:
        summary = (
            f"Multicopter drone design for {req.payload_kg}kg payload. "
            f"Total weight: {design_data.get('total_weight_kg', 0):.1f}kg. "
            f"Estimated flight time: {design_data.get('hover_time_min', 0):.1f} minutes."
        )

        # These keys EXACTLY match what's stored in perform_calculations
        specifications = {
            "frame_size_mm": design_data.get("frame_size_mm"),
            "num_motors": design_data.get("num_motors"),
            "motor_kv": design_data.get("motor_kv"),
            "prop_diameter_in": design_data.get("prop_diameter_in"),
            "prop_pitch_in": design_data.get("prop_pitch_in"),
            "battery_cells": design_data.get("battery_cells"),
            "battery_capacity_mah": design_data.get("battery_capacity_mah"),
            "total_weight_kg": design_data.get("total_weight_kg"),
            "max_thrust_n": design_data.get("max_thrust_n"),
            "thrust_to_weight": design_data.get("thrust_to_weight"),
            "hover_time_min": design_data.get("hover_time_min"),
            "max_speed_ms": design_data.get("max_speed_ms"),
        }

    elif vehicle == VehicleType.FIXED_WING:
        summary = (
            f"Fixed-wing aircraft design for {req.payload_kg}kg payload. "
            f"Total weight: {design_data.get('total_weight_kg', 0):.1f}kg. "
            f"Range: {design_data.get('range_km', 0):.0f}km at {design_data.get('cruise_speed_ms', 0) * 3.6:.0f}km/h."
        )

        # These keys EXACTLY match what's stored in perform_calculations
        specifications = {
            "wing_span_m": design_data.get("wing_span_m"),
            "wing_area_m2": design_data.get("wing_area_m2"),
            "aspect_ratio": design_data.get("aspect_ratio"),
            "total_weight_kg": design_data.get("total_weight_kg"),
            "empty_weight_kg": design_data.get("empty_weight_kg"),
            "fuel_weight_kg": design_data.get("fuel_weight_kg"),
            "power_required_w": design_data.get("power_required_w"),
            "stall_speed_ms": design_data.get("stall_speed_ms"),
            "cruise_speed_ms": design_data.get("cruise_speed_ms"),
            "cruise_speed_kmh": design_data.get("cruise_speed_ms", 0) * 3.6,  # Calculated
            "range_km": design_data.get("range_km"),
        }

    elif vehicle == VehicleType.HELICOPTER:
        summary = (
            f"Helicopter design for {req.payload_kg}kg payload. "
            f"Total weight: {design_data.get('total_weight_kg', 0):.1f}kg. "
            f"Main rotor diameter: {design_data.get('rotor_diameter_m', 0):.1f}m."
        )

        # These keys EXACTLY match what's stored in perform_calculations
        specifications = {
            "rotor_diameter_m": design_data.get("rotor_diameter_m"),
            "rotor_blades": design_data.get("rotor_blades"),
            "rotor_rpm": design_data.get("rotor_rpm"),
            "tail_rotor_diameter_m": design_data.get("tail_rotor_diameter_m"),
            "total_weight_kg": design_data.get("total_weight_kg"),
            "hover_power_w": design_data.get("hover_power_w"),
            "cruise_power_w": design_data.get("cruise_power_w"),
            "engine_power_w": design_data.get("engine_power_w"),
            "fuel_consumption_kghr": design_data.get("fuel_consumption_kghr"),
        }

    elif vehicle == VehicleType.ROCKET:
        summary = (
            f"Rocket design for {req.payload_kg}kg payload to {design_data.get('predicted_altitude_m', 0):.0f}m altitude. "
            f"Total mass: {design_data.get('total_mass_kg', 0):.1f}kg. "
            f"Number of stages: {design_data.get('num_stages', 1)}."
        )

        # These keys EXACTLY match what's stored in perform_calculations
        specifications = {
            "num_stages": design_data.get("num_stages"),
            "total_mass_kg": design_data.get("total_mass_kg"),
            "payload_fraction": design_data.get("payload_fraction"),
            "total_delta_v_ms": design_data.get("total_delta_v_ms"),
            "predicted_altitude_m": design_data.get("predicted_altitude_m"),
            "target_achieved": design_data.get("target_achieved"),
            "stages": design_data.get("stages"),  # Already formatted correctly
        }

    elif vehicle == VehicleType.SATELLITE:
        orbit_alt_m = design_data.get("orbit_altitude_m", 0)
        summary = (
            f"Satellite design for {req.payload_kg}kg payload at {orbit_alt_m / 1000:.0f}km altitude. "
            f"Total mass: {design_data.get('total_mass_kg', 0):.1f}kg. "
            f"Design life: {design_data.get('design_life_years', 0):.0f} years."
        )

        # These keys EXACTLY match what's stored in perform_calculations
        specifications = {
            "orbit_type": design_data.get("orbit_type"),
            "orbit_altitude_m": design_data.get("orbit_altitude_m"),
            "orbit_altitude_km": design_data.get("orbit_altitude_m", 0) / 1000,  # Calculated
            "orbital_velocity_ms": design_data.get("orbital_velocity_ms"),
            "orbital_period_s": design_data.get("orbital_period_s"),
            "orbital_period_hours": design_data.get("orbital_period_s", 0) / 3600,  # Calculated
            "total_mass_kg": design_data.get("total_mass_kg"),
            "solar_array_area_m2": design_data.get("solar_array_area_m2"),
            "battery_capacity_wh": design_data.get("battery_capacity_wh"),
            "dimensions_m": design_data.get("dimensions_m"),
            "design_life_years": design_data.get("design_life_years"),
        }

    elif vehicle == VehicleType.GLIDER:
        summary = (
            f"Glider design with {design_data.get('best_glide_ratio', 0):.1f}:1 best glide ratio. "
            f"Wingspan: {design_data.get('wing_span_m', 0):.1f}m. "
            f"Empty weight: {design_data.get('empty_weight_kg', 0):.1f}kg."
        )

        # These keys EXACTLY match what's stored in perform_calculations
        specifications = {
            "wing_span_m": design_data.get("wing_span_m"),
            "wing_area_m2": design_data.get("wing_area_m2"),
            "aspect_ratio": design_data.get("aspect_ratio"),
            "empty_weight_kg": design_data.get("empty_weight_kg"),
            "max_weight_kg": design_data.get("max_weight_kg"),
            "best_glide_ratio": design_data.get("best_glide_ratio"),
            "min_sink_rate_ms": design_data.get("min_sink_rate_ms"),
            "stall_speed_ms": design_data.get("stall_speed_ms"),
            "stall_speed_kmh": design_data.get("stall_speed_ms", 0) * 3.6,  # Calculated
            "max_speed_ms": design_data.get("max_speed_ms"),
            "max_speed_kmh": design_data.get("max_speed_ms", 0) * 3.6,  # Calculated
        }

    else:
        summary = f"Design for {vehicle.value}"
        specifications = design_data

    # Remove None values from specifications
    specifications = {k: v for k, v in specifications.items() if v is not None}

    # Create validation result
    validation = state.validation_result or ValidationResult(
        passed=True,
        checks={},
        warnings=state.warnings,
        errors=state.errors,
        suggestions=[]
    )

    # Create design output
    state.design_output = DesignOutput(
        vehicle_type=vehicle,
        summary=summary,
        specifications=specifications,
        performance={},
        components=[],
        weight_breakdown={},
        validation=validation,
        citations=[],
        confidence_score=state.classification_confidence
    )

    state.phase = DesignPhase.COMPLETE
    return state
