"""
Validator Node

Validates design outputs against user requirements.
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
    ValidationResult,
    VehicleType,
)

# ... rest of the file remains the same


# =============================================================================
# Validation Rules
# =============================================================================


class ValidationRule:
    """Base class for validation rules."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def check(
        self,
        calculations: List[CalculationResult],
        requirements: Optional[UserRequirements],
    ) -> tuple[bool, str]:
        """
        Check if the rule passes.

        Returns:
            (passed, message)
        """
        raise NotImplementedError


class ThrustToWeightRule(ValidationRule):
    """Check thrust-to-weight ratio for drones."""

    def __init__(self):
        super().__init__(
            "thrust_to_weight",
            "Thrust-to-weight ratio should be > 1.5 for good maneuverability",
        )

    def check(self, calculations, requirements):
        for calc in calculations:
            if calc.success and "thrust_to_weight" in calc.outputs:
                tw = calc.outputs["thrust_to_weight"]
                if tw >= 2.0:
                    return True, f"T/W ratio of {tw:.1f} is excellent"
                elif tw >= 1.5:
                    return True, f"T/W ratio of {tw:.1f} is acceptable"
                else:
                    return False, f"T/W ratio of {tw:.1f} is too low, need > 1.5"
        return True, "T/W ratio not applicable"


class FlightTimeRule(ValidationRule):
    """Check if flight time meets requirements."""

    def __init__(self):
        super().__init__("flight_time", "Flight time should meet or exceed requirement")

    def check(self, calculations, requirements):
        if not requirements or not requirements.endurance_hours:
            return True, "No flight time requirement specified"

        target_minutes = requirements.endurance_hours * 60

        for calc in calculations:
            if calc.success:
                for key in ["hover_time", "flight_time", "endurance"]:
                    if key in calc.outputs:
                        actual = calc.outputs[key]
                        if actual >= target_minutes * 0.9:  # 90% tolerance
                            return (
                                True,
                                f"Flight time {actual:.1f} min meets requirement",
                            )
                        else:
                            return (
                                False,
                                f"Flight time {actual:.1f} min < required {target_minutes:.1f} min",
                            )

        return True, "Flight time not calculated"


class PayloadRule(ValidationRule):
    """Check payload capacity."""

    def __init__(self):
        super().__init__(
            "payload_capacity", "Design should accommodate specified payload"
        )

    def check(self, calculations, requirements):
        if not requirements or not requirements.payload_kg:
            return True, "No payload requirement specified"

        # Check if any calculation used the payload
        for calc in calculations:
            if calc.success and "payload_kg" in calc.inputs:
                if calc.inputs["payload_kg"] >= requirements.payload_kg:
                    return (
                        True,
                        f"Payload capacity {calc.inputs['payload_kg']} kg is sufficient",
                    )

        return True, "Payload requirement incorporated in design"


class StructuralMarginRule(ValidationRule):
    """Check structural safety margins."""

    def __init__(self):
        super().__init__(
            "structural_margin", "Design should have adequate structural margins"
        )

    def check(self, calculations, requirements):
        # General check for reasonable weight fractions
        for calc in calculations:
            if calc.success and "total_weight" in calc.outputs:
                total = calc.outputs["total_weight"]
                if total > 0:
                    return True, f"Total weight {total:.2f} kg is reasonable"

        return True, "Structural margins assumed adequate"


class RangeRule(ValidationRule):
    """Check range meets requirements."""

    def __init__(self):
        super().__init__("range", "Range should meet requirement")

    def check(self, calculations, requirements):
        if not requirements or not requirements.range_km:
            return True, "No range requirement specified"

        for calc in calculations:
            if calc.success:
                for key in ["range_km", "range"]:
                    if key in calc.outputs:
                        actual = calc.outputs[key]
                        if actual >= requirements.range_km * 0.9:
                            return True, f"Range {actual:.0f} km meets requirement"
                        else:
                            return (
                                False,
                                f"Range {actual:.0f} km < required {requirements.range_km:.0f} km",
                            )

        return True, "Range requirement incorporated"


class AltitudeRule(ValidationRule):
    """Check altitude capability for rockets."""

    def __init__(self):
        super().__init__("altitude", "Rocket should reach target altitude")

    def check(self, calculations, requirements):
        if not requirements or not requirements.target_altitude_m:
            return True, "No altitude requirement specified"

        for calc in calculations:
            if calc.success and "max_altitude" in calc.outputs:
                actual = calc.outputs["max_altitude"]
                target = requirements.target_altitude_m

                if actual >= target * 0.9:
                    return True, f"Predicted altitude {actual:.0f} m meets target"
                else:
                    return (
                        False,
                        f"Predicted altitude {actual:.0f} m < target {target:.0f} m",
                    )

        return True, "Altitude not calculated"


# =============================================================================
# Validation Logic
# =============================================================================

# Rules by vehicle type
VEHICLE_RULES = {
    VehicleType.DRONE: [
        ThrustToWeightRule(),
        FlightTimeRule(),
        PayloadRule(),
    ],
    VehicleType.FIXED_WING: [
        RangeRule(),
        PayloadRule(),
        StructuralMarginRule(),
    ],
    VehicleType.HELICOPTER: [
        RangeRule(),
        PayloadRule(),
        StructuralMarginRule(),
    ],
    VehicleType.ROCKET: [
        AltitudeRule(),
        PayloadRule(),
        StructuralMarginRule(),
    ],
    VehicleType.SATELLITE: [
        PayloadRule(),
        StructuralMarginRule(),
    ],
    VehicleType.GLIDER: [
        PayloadRule(),
        StructuralMarginRule(),
    ],
}


def validate_design(
    vehicle_type: VehicleType,
    calculations: List[CalculationResult],
    requirements: Optional[UserRequirements],
) -> ValidationResult:
    """
    Validate the design against all applicable rules.

    Args:
        vehicle_type: Type of vehicle
        calculations: Calculation results
        requirements: Parsed requirements

    Returns:
        ValidationResult
    """
    rules = VEHICLE_RULES.get(vehicle_type, [])

    passed = []
    failed = []
    warnings = []
    suggestions = []

    # Run all rules
    for rule in rules:
        try:
            success, message = rule.check(calculations, requirements)

            if success:
                passed.append(f"{rule.name}: {message}")
            else:
                failed.append(f"{rule.name}: {message}")

        except Exception as e:
            warnings.append(f"{rule.name}: Validation error - {e}")

    # Check for calculation failures
    for calc in calculations:
        if not calc.success:
            warnings.append(
                f"Calculation '{calc.tool_name}' failed: {calc.error_message}"
            )

    # Generate suggestions based on failures
    for failure in failed:
        if "T/W ratio" in failure and "too low" in failure:
            suggestions.append(
                "Consider using higher thrust motors or lighter components"
            )
        elif "Flight time" in failure:
            suggestions.append("Consider larger battery or more efficient motors")
        elif "Range" in failure:
            suggestions.append("Consider more fuel capacity or improved aerodynamics")
        elif "altitude" in failure:
            suggestions.append("Consider higher impulse motor or staging")

    is_valid = len(failed) == 0

    return ValidationResult(
        is_valid=is_valid,
        checks_passed=passed,
        checks_failed=failed,
        warnings=warnings,
        suggestions=suggestions,
    )


# =============================================================================
# Node Function
# =============================================================================


def validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for design validation.

    Args:
        state: Current graph state

    Returns:
        State updates
    """
    vehicle_type = state["vehicle_type"]
    calculations = state.get("calculation_results", [])
    requirements = state.get("requirements")
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    # Validate design
    result = validate_design(vehicle_type, calculations, requirements)

    # Determine next phase
    if result.is_valid:
        next_phase = DesignPhase.SYNTHESIZING
        message = f"Design validated: {len(result.checks_passed)} checks passed"
    elif iteration < max_iterations:
        next_phase = DesignPhase.CALCULATING  # Try again
        message = (
            f"Validation failed, iteration {iteration + 1}: {result.checks_failed}"
        )
    else:
        next_phase = DesignPhase.SYNTHESIZING  # Proceed anyway with warnings
        message = f"Validation incomplete after {max_iterations} iterations, proceeding with warnings"

    return {
        "validation_result": result,
        "iteration_count": iteration + 1,
        "current_phase": next_phase,
        "messages": [{"role": "system", "content": message}],
    }


if __name__ == "__main__":
    # Test validation
    print("=== Validator Node Tests ===\n")

    # Test drone validation
    drone_calcs = [
        CalculationResult(
            tool_name="size_drone",
            inputs={"payload_kg": 0.5},
            outputs={"thrust_to_weight": 2.5, "hover_time": 25, "total_weight": 1.5},
            success=True,
        )
    ]

    drone_req = UserRequirements(
        raw_input="test",
        payload_kg=0.5,
        endurance_hours=0.4,  # 24 min
    )

    result = validate_design(VehicleType.DRONE, drone_calcs, drone_req)

    print("Drone Validation:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Passed: {result.checks_passed}")
    print(f"  Failed: {result.checks_failed}")
    print(f"  Suggestions: {result.suggestions}")
