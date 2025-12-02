"""
Calculator Node

Executes aerospace calculations using the appropriate tools.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# FIXED: Import from graph.state
sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.state import (
    CalculationResult,
    DesignPhase,
    DesignState,
    UserRequirements,
    VehicleType,
)

# ... rest of the file remains the same


# Import tools - handle both relative and absolute imports
try:
    from tools import (
        DRONE_TOOLS,
        FIXED_WING_TOOLS,
        GLIDER_TOOLS,
        HELICOPTER_TOOLS,
        ROCKET_TOOLS,
        SATELLITE_TOOLS,
        get_tool_function,
        get_tools_for_vehicle,
    )
except ImportError:
    from ..tools import (
        DRONE_TOOLS,
        FIXED_WING_TOOLS,
        GLIDER_TOOLS,
        HELICOPTER_TOOLS,
        ROCKET_TOOLS,
        SATELLITE_TOOLS,
        get_tool_function,
        get_tools_for_vehicle,
    )


# =============================================================================
# Calculation Planning
# =============================================================================


def get_calculation_plan(
    vehicle_type: VehicleType, requirements: Optional[UserRequirements]
) -> List[Dict[str, Any]]:
    """
    Generate a calculation plan based on vehicle type and requirements.

    Args:
        vehicle_type: Type of vehicle
        requirements: Parsed requirements

    Returns:
        List of calculation specifications
    """
    plan = []

    if vehicle_type == VehicleType.DRONE:
        plan = plan_drone_calculations(requirements)
    elif vehicle_type == VehicleType.FIXED_WING:
        plan = plan_fixed_wing_calculations(requirements)
    elif vehicle_type == VehicleType.HELICOPTER:
        plan = plan_helicopter_calculations(requirements)
    elif vehicle_type == VehicleType.ROCKET:
        plan = plan_rocket_calculations(requirements)
    elif vehicle_type == VehicleType.SATELLITE:
        plan = plan_satellite_calculations(requirements)
    elif vehicle_type == VehicleType.GLIDER:
        plan = plan_glider_calculations(requirements)

    return plan


def plan_drone_calculations(req: Optional[UserRequirements]) -> List[Dict]:
    """Plan calculations for drone design."""
    # Default values
    payload = req.payload_kg if req and req.payload_kg else 0.5
    flight_time = (req.endurance_hours * 60) if req and req.endurance_hours else 20

    return [
        {
            "tool": "size_drone",
            "inputs": {
                "payload_kg": payload,
                "flight_time_minutes": flight_time,
                "num_motors": 4,
                "application": "photography",
            },
            "description": "Complete drone sizing",
        }
    ]


def plan_fixed_wing_calculations(req: Optional[UserRequirements]) -> List[Dict]:
    """Plan calculations for fixed-wing aircraft."""
    payload = req.payload_kg if req and req.payload_kg else 200
    range_km = req.range_km if req and req.range_km else 500
    speed = req.speed_kmh if req and req.speed_kmh else 200

    return [
        {
            "tool": "size_aircraft",
            "inputs": {
                "payload_kg": payload,
                "range_km": range_km,
                "cruise_speed_kmh": speed,
                "aircraft_type": "single_engine_ga",
            },
            "description": "Complete aircraft sizing",
        }
    ]


def plan_helicopter_calculations(req: Optional[UserRequirements]) -> List[Dict]:
    """Plan calculations for helicopter design."""
    payload = req.payload_kg if req and req.payload_kg else 400
    range_km = req.range_km if req and req.range_km else 300
    speed = req.speed_kmh if req and req.speed_kmh else 200

    return [
        {
            "tool": "design_helicopter",
            "inputs": {
                "payload_kg": payload,
                "range_km": range_km,
                "cruise_speed_kmh": speed,
            },
            "description": "Complete helicopter sizing",
        }
    ]


def plan_rocket_calculations(req: Optional[UserRequirements]) -> List[Dict]:
    """Plan calculations for rocket design."""
    payload = req.payload_kg if req and req.payload_kg else 0.5
    altitude = req.target_altitude_m if req and req.target_altitude_m else 1000

    return [
        {
            "tool": "design_rocket",
            "inputs": {
                "payload_kg": payload,
                "target_altitude": altitude,
                "motor_type": "solid",
            },
            "description": "Complete rocket sizing",
        }
    ]


def plan_satellite_calculations(req: Optional[UserRequirements]) -> List[Dict]:
    """Plan calculations for satellite design."""
    payload_power = 50  # W default
    payload_mass = req.payload_kg if req and req.payload_kg else 20
    altitude = (
        (req.orbit_altitude_km * 1000) if req and req.orbit_altitude_km else 400000
    )
    mission_years = req.mission_years if req and req.mission_years else 5

    return [
        {
            "tool": "design_satellite",
            "inputs": {
                "payload_power": payload_power,
                "payload_mass": payload_mass,
                "altitude": altitude,
                "mission_years": mission_years,
            },
            "description": "Complete satellite sizing",
        }
    ]


def plan_glider_calculations(req: Optional[UserRequirements]) -> List[Dict]:
    """Plan calculations for glider design."""
    pilot_weight = 80
    target_ld = req.constraints.get("target_ld", 40) if req else 40

    return [
        {
            "tool": "design_glider",
            "inputs": {
                "pilot_weight": pilot_weight,
                "target_ld": target_ld,
                "class_type": "standard",
            },
            "description": "Complete glider sizing",
        }
    ]


# =============================================================================
# Calculation Execution
# =============================================================================


def execute_calculation(tool_name: str, inputs: Dict[str, Any]) -> CalculationResult:
    """
    Execute a single calculation.

    Args:
        tool_name: Name of the tool to use
        inputs: Input parameters

    Returns:
        CalculationResult
    """
    try:
        # Get the tool function
        tool_func = get_tool_function(tool_name)

        # Execute
        result = tool_func(**inputs)

        # Convert result to dict
        if hasattr(result, "__dict__"):
            outputs = {
                k: v for k, v in result.__dict__.items() if not k.startswith("_")
            }
        elif hasattr(result, "_asdict"):
            outputs = result._asdict()
        elif isinstance(result, dict):
            outputs = result
        else:
            outputs = {"result": result}

        # Handle nested objects
        clean_outputs = {}
        for k, v in outputs.items():
            if hasattr(v, "__dict__"):
                clean_outputs[k] = {
                    kk: vv for kk, vv in v.__dict__.items() if not kk.startswith("_")
                }
            elif isinstance(v, list):
                clean_outputs[k] = [
                    item.__dict__ if hasattr(item, "__dict__") else item
                    for item in v[:5]  # Limit list length
                ]
            else:
                clean_outputs[k] = v

        return CalculationResult(
            tool_name=tool_name, inputs=inputs, outputs=clean_outputs, success=True
        )

    except Exception as e:
        return CalculationResult(
            tool_name=tool_name,
            inputs=inputs,
            outputs={},
            success=False,
            error_message=str(e),
        )


def execute_calculation_plan(plan: List[Dict[str, Any]]) -> List[CalculationResult]:
    """
    Execute a full calculation plan.

    Args:
        plan: List of calculation specifications

    Returns:
        List of CalculationResult
    """
    results = []

    for step in plan:
        result = execute_calculation(step["tool"], step["inputs"])
        results.append(result)

        # If a critical calculation fails, we might want to stop
        if not result.success and "size" in step["tool"]:
            break

    return results


# =============================================================================
# Node Function
# =============================================================================


def calculator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for performing calculations.

    Args:
        state: Current graph state

    Returns:
        State updates
    """
    vehicle_type = state["vehicle_type"]
    requirements = state.get("requirements")

    # Generate calculation plan
    plan = get_calculation_plan(vehicle_type, requirements)

    # Execute calculations
    results = execute_calculation_plan(plan)

    # Determine success
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if successful:
        message = f"Completed {len(successful)} calculations successfully"
        if failed:
            message += f", {len(failed)} failed"
        next_phase = DesignPhase.VALIDATING
    else:
        message = f"All {len(failed)} calculations failed"
        next_phase = DesignPhase.ERROR

    return {
        "calculation_results": results,
        "current_phase": next_phase,
        "messages": [{"role": "system", "content": message}],
    }


if __name__ == "__main__":
    # Test calculations
    print("=== Calculator Node Tests ===\n")

    test_cases = [
        (
            VehicleType.DRONE,
            UserRequirements(raw_input="test", payload_kg=0.5, endurance_hours=0.5),
        ),
        (
            VehicleType.FIXED_WING,
            UserRequirements(
                raw_input="test", payload_kg=200, range_km=500, speed_kmh=250
            ),
        ),
        (
            VehicleType.ROCKET,
            UserRequirements(raw_input="test", payload_kg=0.5, target_altitude_m=1000),
        ),
    ]

    for vtype, req in test_cases:
        print(f"Vehicle Type: {vtype.value}")

        plan = get_calculation_plan(vtype, req)
        print(f"  Plan: {[p['tool'] for p in plan]}")

        results = execute_calculation_plan(plan)

        for result in results:
            print(f"  {result.tool_name}: {'SUCCESS' if result.success else 'FAILED'}")
            if result.success:
                # Print key outputs
                for key, value in list(result.outputs.items())[:3]:
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value}")
                    elif isinstance(value, dict):
                        print(f"    {key}: <nested object>")
            else:
                print(f"    Error: {result.error_message}")
        print()
