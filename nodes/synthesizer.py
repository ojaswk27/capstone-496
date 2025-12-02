"""
Synthesizer Node

Synthesizes final design output from all collected information.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# FIXED: Import from graph.state
sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.state import (
    CalculationResult,
    DesignOutput,  # CHANGED: Was DesignSpecification
    DesignPhase,
    DesignState,
    SearchResult,
    UserRequirements,
    ValidationResult,
    VehicleType,
)

# ... rest of the file remains the same


# =============================================================================
# Output Formatting
# =============================================================================


def extract_design_components(
    vehicle_type: VehicleType, calculations: List[CalculationResult]
) -> Dict[str, Any]:
    """
    Extract component specifications from calculation results.

    Args:
        vehicle_type: Vehicle type
        calculations: Calculation results

    Returns:
        Dict of component specifications
    """
    components = {}

    for calc in calculations:
        if not calc.success:
            continue

        outputs = calc.outputs

        if vehicle_type == VehicleType.DRONE:
            if "frame_size" in outputs:
                components["frame"] = {
                    "size_mm": outputs.get("frame_size"),
                    "type": f"{outputs.get('num_motors', 4)}-motor configuration",
                }
            if "motor_kv" in outputs:
                components["motors"] = {
                    "kv_rating": outputs.get("motor_kv"),
                    "quantity": outputs.get("num_motors", 4),
                }
            if "prop_diameter" in outputs:
                components["propellers"] = {
                    "diameter_inches": outputs.get("prop_diameter"),
                    "pitch_inches": outputs.get("prop_pitch"),
                }
            if "battery_cells" in outputs:
                components["battery"] = {
                    "cells": outputs.get("battery_cells"),
                    "capacity_mah": outputs.get("battery_capacity"),
                    "type": "LiPo",
                }

        elif vehicle_type == VehicleType.FIXED_WING:
            if "wing" in outputs:
                wing = outputs["wing"]
                components["wing"] = {
                    "span_m": wing.get("span"),
                    "area_m2": wing.get("area"),
                    "aspect_ratio": wing.get("aspect_ratio"),
                    "loading_N_m2": wing.get("wing_loading"),
                }
            if "power_required" in outputs:
                components["powerplant"] = {
                    "power_kw": outputs.get("power_required", 0) / 1000,
                    "type": "Piston engine",
                }
            if "fuel_weight" in outputs:
                components["fuel_system"] = {
                    "fuel_kg": outputs.get("fuel_weight"),
                    "type": "Avgas",
                }

        elif vehicle_type == VehicleType.HELICOPTER:
            if "rotor" in outputs:
                rotor = outputs["rotor"]
                components["main_rotor"] = {
                    "diameter_m": rotor.get("diameter"),
                    "blades": rotor.get("num_blades"),
                    "rpm": rotor.get("rpm"),
                }
            if "tail_rotor_diameter" in outputs:
                components["tail_rotor"] = {
                    "diameter_m": outputs.get("tail_rotor_diameter"),
                }
            if "engine_power" in outputs:
                components["engine"] = {
                    "power_kw": outputs.get("engine_power", 0) / 1000,
                    "type": "Turboshaft",
                }

        elif vehicle_type == VehicleType.ROCKET:
            if "stages" in outputs:
                stages = outputs["stages"]
                for i, stage in enumerate(stages, 1):
                    if isinstance(stage, dict):
                        components[f"stage_{i}"] = {
                            "propellant_kg": stage.get("propellant_mass"),
                            "structural_kg": stage.get("structural_mass"),
                            "delta_v_ms": stage.get("delta_v"),
                        }
            if "total_delta_v" in outputs:
                components["propulsion_summary"] = {
                    "total_delta_v_ms": outputs.get("total_delta_v"),
                    "payload_fraction": outputs.get("payload_fraction"),
                }

        elif vehicle_type == VehicleType.SATELLITE:
            if "orbit" in outputs:
                orbit = outputs["orbit"]
                components["orbit"] = {
                    "altitude_km": orbit.get("altitude", 0) / 1000,
                    "velocity_ms": orbit.get("velocity"),
                    "period_min": orbit.get("period", 0) / 60,
                    "type": orbit.get("orbit_type"),
                }
            if "power" in outputs:
                power = outputs["power"]
                components["power_system"] = {
                    "solar_array_m2": power.get("solar_array_area"),
                    "battery_wh": power.get("battery_capacity"),
                    "average_power_w": power.get("average_power"),
                }

        elif vehicle_type == VehicleType.GLIDER:
            if "wing_span" in outputs:
                components["wing"] = {
                    "span_m": outputs.get("wing_span"),
                    "area_m2": outputs.get("wing_area"),
                    "aspect_ratio": outputs.get("aspect_ratio"),
                }
            if "glide" in outputs:
                glide = outputs["glide"]
                components["performance"] = {
                    "best_ld": glide.get("best_ld"),
                    "min_sink_ms": glide.get("min_sink_rate"),
                }

    return components


def extract_performance(
    vehicle_type: VehicleType, calculations: List[CalculationResult]
) -> Dict[str, Any]:
    """Extract performance predictions."""
    performance = {}

    for calc in calculations:
        if not calc.success:
            continue

        outputs = calc.outputs

        if vehicle_type == VehicleType.DRONE:
            if "thrust_to_weight" in outputs:
                performance["thrust_to_weight"] = outputs["thrust_to_weight"]
            if "hover_time" in outputs:
                performance["flight_time_min"] = outputs["hover_time"]
            if "max_speed" in outputs:
                performance["max_speed_ms"] = outputs["max_speed"]

        elif vehicle_type == VehicleType.FIXED_WING:
            if "performance" in outputs:
                perf = outputs["performance"]
                performance["stall_speed_ms"] = perf.get("stall_speed")
                performance["cruise_speed_ms"] = perf.get("cruise_speed")
                performance["range_km"] = perf.get("range_km")

        elif vehicle_type == VehicleType.HELICOPTER:
            if "hover" in outputs:
                hover = outputs["hover"]
                performance["hover_power_kw"] = hover.get("power_actual", 0) / 1000
            if "forward_flight" in outputs:
                ff = outputs["forward_flight"]
                performance["cruise_power_kw"] = ff.get("power_required", 0) / 1000

        elif vehicle_type == VehicleType.ROCKET:
            if "max_altitude" in outputs:
                performance["max_altitude_m"] = outputs["max_altitude"]
            if "total_delta_v" in outputs:
                performance["total_delta_v_ms"] = outputs["total_delta_v"]

        elif vehicle_type == VehicleType.SATELLITE:
            if "thermal" in outputs:
                thermal = outputs["thermal"]
                performance["hot_case_C"] = thermal.get("hot_case_temp", 273) - 273
                performance["cold_case_C"] = thermal.get("cold_case_temp", 273) - 273

        elif vehicle_type == VehicleType.GLIDER:
            if "glide" in outputs:
                glide = outputs["glide"]
                performance["best_ld"] = glide.get("best_ld")
                performance["best_ld_speed_ms"] = glide.get("best_ld_speed")
                performance["min_sink_ms"] = glide.get("min_sink_rate")

    return performance


def extract_weights(
    calculations: List[CalculationResult],
) -> tuple[Dict[str, float], float]:
    """Extract weight breakdown."""
    weights = {}
    total = 0

    for calc in calculations:
        if not calc.success:
            continue

        outputs = calc.outputs

        if "total_weight" in outputs:
            total = outputs["total_weight"]
        if "weight_breakdown" in outputs:
            weights = outputs["weight_breakdown"]
        if "total_mass" in outputs:
            total = outputs["total_mass"]
        if "empty_weight" in outputs:
            weights["empty"] = outputs["empty_weight"]
        if "fuel_weight" in outputs:
            weights["fuel"] = outputs["fuel_weight"]
        if "payload_mass" in outputs:
            weights["payload"] = outputs["payload_mass"]

    return weights, total


def generate_rationale(
    vehicle_type: VehicleType,
    requirements: Optional[UserRequirements],
    calculations: List[CalculationResult],
    search_results: List[SearchResult],
) -> List[str]:
    """Generate design rationale."""
    rationale = []

    # Vehicle type rationale
    rationale.append(
        f"Designed as a {vehicle_type.value.replace('_', ' ')} based on mission requirements."
    )

    # Requirements-based rationale
    if requirements:
        if requirements.payload_kg:
            rationale.append(f"Sized to carry {requirements.payload_kg} kg payload.")
        if requirements.range_km:
            rationale.append(
                f"Designed for {requirements.range_km} km range capability."
            )
        if requirements.endurance_hours:
            rationale.append(
                f"Optimized for {requirements.endurance_hours * 60:.0f} minute flight time."
            )
        if requirements.mission_type:
            rationale.append(
                f"Configuration optimized for {requirements.mission_type} mission profile."
            )

    # Calculation-based rationale
    for calc in calculations:
        if calc.success:
            rationale.append(f"Used {calc.tool_name} for preliminary sizing.")

    return rationale


def generate_citations(search_results: List[SearchResult]) -> List[str]:
    """Generate citations from search results."""
    citations = []
    seen_sources = set()

    for result in search_results:
        if result.source not in seen_sources:
            seen_sources.add(result.source)
            citations.append(result.source)

    if not citations:
        citations.append("Design based on standard aerospace engineering principles")

    return citations


def calculate_confidence(
    validation: Optional[ValidationResult], calculations: List[CalculationResult]
) -> float:
    """Calculate overall design confidence."""
    if not calculations:
        return 0.0

    # Base confidence from successful calculations
    successful = sum(1 for c in calculations if c.success)
    calc_confidence = successful / len(calculations)

    # Adjust for validation
    if validation:
        if validation.is_valid:
            val_factor = 1.0
        else:
            # Reduce confidence based on failures
            failure_penalty = len(validation.checks_failed) * 0.1
            val_factor = max(0.5, 1.0 - failure_penalty)
    else:
        val_factor = 0.8

    return min(1.0, calc_confidence * val_factor)


# =============================================================================
# Summary Generation
# =============================================================================


def generate_summary(
    vehicle_type: VehicleType,
    components: Dict[str, Any],
    performance: Dict[str, Any],
    total_weight: float,
) -> str:
    """Generate a human-readable summary."""
    parts = [f"Preliminary design for a {vehicle_type.value.replace('_', ' ')}."]

    if total_weight > 0:
        parts.append(f"Total estimated weight: {total_weight:.2f} kg.")

    # Add key performance metrics
    if "flight_time_min" in performance:
        parts.append(
            f"Estimated flight time: {performance['flight_time_min']:.1f} minutes."
        )
    if "range_km" in performance:
        parts.append(f"Range: {performance['range_km']:.0f} km.")
    if "max_altitude_m" in performance:
        parts.append(f"Maximum altitude: {performance['max_altitude_m']:.0f} m.")
    if "best_ld" in performance:
        parts.append(f"Best L/D ratio: {performance['best_ld']:.1f}.")

    return " ".join(parts)


# =============================================================================
# Node Function
# =============================================================================


def synthesizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for synthesizing final design.

    Args:
        state: Current graph state

    Returns:
        State updates
    """
    vehicle_type = state["vehicle_type"]
    calculations = state.get("calculation_results", [])
    search_results = state.get("search_results", [])
    requirements = state.get("requirements")
    validation = state.get("validation_result")

    # Extract all design elements
    components = extract_design_components(vehicle_type, calculations)
    performance = extract_performance(vehicle_type, calculations)
    weights, total_weight = extract_weights(calculations)
    rationale = generate_rationale(
        vehicle_type, requirements, calculations, search_results
    )
    citations = generate_citations(search_results)
    confidence = calculate_confidence(validation, calculations)
    summary = generate_summary(vehicle_type, components, performance, total_weight)

    # Create design specification
    design_spec = DesignSpecification(
        vehicle_type=vehicle_type.value,
        summary=summary,
        components=components,
        performance=performance,
        weights=weights,
        total_weight=total_weight,
        rationale=rationale,
        citations=citations,
        confidence=confidence,
    )

    return {
        "design_spec": design_spec,
        "current_phase": DesignPhase.COMPLETE,
        "messages": [
            {
                "role": "system",
                "content": f"Design synthesis complete with {confidence:.0%} confidence",
            }
        ],
    }


def format_design_output(spec: DesignSpecification) -> str:
    """
    Format the design specification for display.

    Args:
        spec: Design specification

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"AEROSPACE DESIGN SPECIFICATION")
    lines.append(f"Vehicle Type: {spec.vehicle_type.upper()}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    lines.append("\n## SUMMARY")
    lines.append(spec.summary)

    lines.append("\n## COMPONENTS")
    for comp_name, comp_data in spec.components.items():
        lines.append(f"\n### {comp_name.replace('_', ' ').title()}")
        if isinstance(comp_data, dict):
            for key, value in comp_data.items():
                if value is not None:
                    lines.append(f"  - {key.replace('_', ' ')}: {value}")

    lines.append("\n## PERFORMANCE")
    for key, value in spec.performance.items():
        if value is not None:
            lines.append(f"  - {key.replace('_', ' ')}: {value}")

    if spec.weights:
        lines.append("\n## WEIGHT BREAKDOWN")
        for key, value in spec.weights.items():
            lines.append(f"  - {key}: {value:.2f} kg")
        lines.append(f"  - TOTAL: {spec.total_weight:.2f} kg")

    lines.append("\n## DESIGN RATIONALE")
    for item in spec.rationale:
        lines.append(f"  • {item}")

    lines.append("\n## REFERENCES")
    for citation in spec.citations:
        lines.append(f"  [{citation}]")

    lines.append(f"\n## CONFIDENCE: {spec.confidence:.0%}")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test synthesizer
    print("=== Synthesizer Node Test ===\n")

    # Create mock calculation results for a drone
    calcs = [
        CalculationResult(
            tool_name="size_drone",
            inputs={"payload_kg": 0.5, "flight_time_minutes": 25},
            outputs={
                "frame_size": 450,
                "num_motors": 4,
                "motor_kv": 920,
                "prop_diameter": 10,
                "prop_pitch": 4.5,
                "battery_cells": 4,
                "battery_capacity": 5000,
                "total_weight": 1.8,
                "thrust_to_weight": 2.3,
                "hover_time": 23,
                "max_speed": 15,
            },
            success=True,
        )
    ]

    validation = ValidationResult(
        is_valid=True,
        checks_passed=["T/W ratio good", "Flight time acceptable"],
        checks_failed=[],
        warnings=[],
        suggestions=[],
    )

    # Extract components
    components = extract_design_components(VehicleType.DRONE, calcs)
    performance = extract_performance(VehicleType.DRONE, calcs)
    weights, total = extract_weights(calcs)

    spec = DesignSpecification(
        vehicle_type="drone",
        summary="Test drone design",
        components=components,
        performance=performance,
        weights=weights,
        total_weight=total,
        rationale=["Test design"],
        citations=["Test source"],
        confidence=0.9,
    )

    print(format_design_output(spec))
