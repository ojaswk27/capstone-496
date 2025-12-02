"""
State Schema for Aerospace Design Assistant

Defines the state structure for the LangGraph workflow including:
- User requirements
- Vehicle classification
- Search results
- Extracted formulas
- Calculations
- Final design output
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class VehicleType(str, Enum):
    """Supported vehicle types."""

    DRONE = "drone"
    FIXED_WING = "fixed_wing"
    HELICOPTER = "helicopter"
    ROCKET = "rocket"
    SATELLITE = "satellite"
    GLIDER = "glider"
    UNKNOWN = "unknown"


class DesignPhase(str, Enum):
    """Current phase of the design process."""

    INITIAL = "initial"
    CLASSIFYING = "classifying"
    PARSING = "parsing"
    SEARCHING = "searching"
    EXTRACTING = "extracting"
    CALCULATING = "calculating"
    VALIDATING = "validating"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    ERROR = "error"


# =============================================================================
# Pydantic Models for Structured Data
# =============================================================================


class UserRequirements(BaseModel):
    """Parsed user requirements."""

    raw_input: str = Field(description="Original user input")

    # Common requirements
    payload_kg: Optional[float] = Field(None, description="Payload mass in kg")
    range_km: Optional[float] = Field(None, description="Range in km")
    endurance_hours: Optional[float] = Field(None, description="Flight time in hours")
    speed_kmh: Optional[float] = Field(None, description="Cruise speed in km/h")
    altitude_m: Optional[float] = Field(None, description="Operating altitude in m")

    # Mission profile
    mission_type: Optional[str] = Field(None, description="Type of mission")
    environment: Optional[str] = Field(None, description="Operating environment")

    # Constraints
    max_weight_kg: Optional[float] = Field(
        None, description="Maximum weight constraint"
    )
    max_cost: Optional[float] = Field(None, description="Budget constraint")

    # Vehicle-specific (populated based on type)
    vehicle_specific: Dict[str, Any] = Field(default_factory=dict)


class DroneRequirements(BaseModel):
    """Drone-specific requirements."""

    flight_time_minutes: Optional[float] = None
    num_motors: int = 4
    application: str = "general"  # racing, photography, heavy_lift
    camera_payload: bool = False
    fpv: bool = False


class FixedWingRequirements(BaseModel):
    """Fixed-wing specific requirements."""

    passengers: int = 0
    stall_speed_kmh: Optional[float] = None
    runway_length_m: Optional[float] = None
    aircraft_type: str = "single_engine_ga"


class RocketRequirements(BaseModel):
    """Rocket-specific requirements."""

    target_altitude_m: Optional[float] = None
    target_orbit: bool = False
    motor_type: str = "solid"  # solid, liquid, hybrid
    recoverable: bool = True


class SatelliteRequirements(BaseModel):
    """Satellite-specific requirements."""

    orbit_altitude_km: Optional[float] = None
    orbit_type: str = "LEO"  # LEO, MEO, GEO
    mission_years: float = 5
    payload_power_w: Optional[float] = None


class HelicopterRequirements(BaseModel):
    """Helicopter-specific requirements."""

    hover_ceiling_m: Optional[float] = None
    useful_load_kg: Optional[float] = None


class GliderRequirements(BaseModel):
    """Glider-specific requirements."""

    glider_class: str = "standard"  # club, standard, 15m, 18m, open
    target_glide_ratio: Optional[float] = None
    pilot_weight_kg: float = 80


class SearchResult(BaseModel):
    """A single search result from RAG."""

    content: str
    source: str
    vehicle_type: str
    relevance_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedFormula(BaseModel):
    """An extracted formula from documents."""

    name: str
    formula: str
    variables: Dict[str, str]
    source: str
    applicable_to: List[str]  # Vehicle types
    confidence: float


class CalculationResult(BaseModel):
    """Result of a calculation."""

    tool_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class DesignComponent(BaseModel):
    """A component of the design."""

    name: str
    category: str  # propulsion, structure, power, etc.
    specifications: Dict[str, Any]
    rationale: str
    source_citations: List[str]


class ValidationResult(BaseModel):
    """Result of design validation."""

    passed: bool
    checks: Dict[str, bool]
    warnings: List[str]
    errors: List[str]
    suggestions: List[str]


class DesignOutput(BaseModel):
    """Final design output."""

    vehicle_type: VehicleType
    summary: str

    # Key specifications
    specifications: Dict[str, Any]

    # Performance predictions
    performance: Dict[str, Any]

    # Components
    components: List[DesignComponent]

    # Weight breakdown
    weight_breakdown: Dict[str, float]

    # Validation
    validation: ValidationResult

    # Citations
    citations: List[str]

    # Confidence
    confidence_score: float


# =============================================================================
# Main State Class
# =============================================================================


class DesignState(BaseModel):
    """
    Main state for the aerospace design workflow.

    This state is passed between nodes in the LangGraph.
    """

    # Session info
    session_id: str = ""

    # Current phase
    phase: DesignPhase = DesignPhase.INITIAL

    # User input
    raw_input: str = ""
    requirements: Optional[UserRequirements] = None

    # Vehicle classification
    vehicle_type: VehicleType = VehicleType.UNKNOWN
    classification_confidence: float = 0.0
    classification_reasoning: str = ""

    # Search results
    search_queries: List[str] = Field(default_factory=list)
    search_results: List[SearchResult] = Field(default_factory=list)

    # Extracted information
    extracted_formulas: List[ExtractedFormula] = Field(default_factory=list)
    extracted_data: Dict[str, Any] = Field(default_factory=dict)

    # Calculations
    calculations: List[CalculationResult] = Field(default_factory=list)
    intermediate_results: Dict[str, Any] = Field(default_factory=dict)

    # Validation
    validation_result: Optional[ValidationResult] = None
    iteration_count: int = 0
    max_iterations: int = 3

    # Final output
    design_output: Optional[DesignOutput] = None

    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


# =============================================================================
# State Update Functions
# =============================================================================


def update_phase(state: DesignState, new_phase: DesignPhase) -> DesignState:
    """Update the current phase."""
    state.phase = new_phase
    return state


def add_search_result(state: DesignState, result: SearchResult) -> DesignState:
    """Add a search result."""
    state.search_results.append(result)
    return state


def add_calculation(state: DesignState, calc: CalculationResult) -> DesignState:
    """Add a calculation result."""
    state.calculations.append(calc)
    return state


def add_error(state: DesignState, error: str) -> DesignState:
    """Add an error message."""
    state.errors.append(error)
    return state


def add_warning(state: DesignState, warning: str) -> DesignState:
    """Add a warning message."""
    state.warnings.append(warning)
    return state


def set_vehicle_type(
    state: DesignState, vehicle_type: VehicleType, confidence: float, reasoning: str
) -> DesignState:
    """Set the classified vehicle type."""
    state.vehicle_type = vehicle_type
    state.classification_confidence = confidence
    state.classification_reasoning = reasoning
    return state


def should_iterate(state: DesignState) -> bool:
    """Check if design iteration is needed."""
    if state.validation_result is None:
        return False

    if state.iteration_count >= state.max_iterations:
        return False

    return not state.validation_result.passed


# =============================================================================
# State Serialization
# =============================================================================


def state_to_dict(state: DesignState) -> Dict[str, Any]:
    """Convert state to dictionary for storage/transmission."""
    return state.dict()


def state_from_dict(data: Dict[str, Any]) -> DesignState:
    """Create state from dictionary."""
    return DesignState(**data)


def state_to_json(state: DesignState) -> str:
    """Convert state to JSON string."""
    return state.json(indent=2)


def state_from_json(json_str: str) -> DesignState:
    """Create state from JSON string."""
    return DesignState.parse_raw(json_str)


# =============================================================================
# Initial State Factory
# =============================================================================


def create_initial_state(user_input: str, session_id: str = "") -> DesignState:
    """
    Create initial state from user input.

    Args:
        user_input: Raw user input string
        session_id: Optional session identifier

    Returns:
        Initialized DesignState
    """
    import uuid

    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    return DesignState(
        session_id=session_id,
        phase=DesignPhase.INITIAL,
        raw_input=user_input,
        requirements=UserRequirements(raw_input=user_input),
    )


def dict_to_design_state(data: dict) -> DesignState:
    """
    Convert a dictionary to DesignState, handling enum conversions.

    LangGraph returns state as dict with enum values as strings.
    This helper properly reconstructs the DesignState with enums.

    Args:
        data: Dictionary representation of DesignState

    Returns:
        Properly typed DesignState with enums
    """
    # Convert enum string values back to enum objects
    if "vehicle_type" in data and isinstance(data["vehicle_type"], str):
        try:
            data["vehicle_type"] = VehicleType(data["vehicle_type"])
        except (ValueError, KeyError):
            data["vehicle_type"] = VehicleType.UNKNOWN

    if "phase" in data and isinstance(data["phase"], str):
        try:
            data["phase"] = DesignPhase(data["phase"])
        except (ValueError, KeyError):
            data["phase"] = DesignPhase.INITIAL

    # Handle nested design_output with vehicle_type enum
    if "design_output" in data and data["design_output"] is not None:
        if isinstance(data["design_output"], dict):
            if "vehicle_type" in data["design_output"] and isinstance(
                data["design_output"]["vehicle_type"], str
            ):
                try:
                    data["design_output"]["vehicle_type"] = VehicleType(
                        data["design_output"]["vehicle_type"]
                    )
                except (ValueError, KeyError):
                    data["design_output"]["vehicle_type"] = VehicleType.UNKNOWN
            data["design_output"] = DesignOutput(**data["design_output"])

    # Create DesignState from dict
    return DesignState(**data)


if __name__ == "__main__":
    # Test state creation
    print("=== State Schema Test ===\n")

    test_input = (
        "Design a surveillance drone with 60 minute flight time and 2kg camera payload"
    )

    state = create_initial_state(test_input)
    print(f"Session ID: {state.session_id}")
    print(f"Phase: {state.phase}")
    print(f"Vehicle Type: {state.vehicle_type}")

    # Simulate state updates
    state = update_phase(state, DesignPhase.CLASSIFYING)
    state = set_vehicle_type(
        state, VehicleType.DRONE, 0.95, "Keywords: drone, flight time, camera payload"
    )

    print(f"\nAfter classification:")
    print(f"  Phase: {state.phase}")
    print(f"  Vehicle: {state.vehicle_type}")
    print(f"  Confidence: {state.classification_confidence}")

    # Test serialization
    json_str = state_to_json(state)
    print(f"\nJSON preview:\n{json_str[:500]}...")

    # Test deserialization
    restored = state_from_json(json_str)
    print(f"\nRestored vehicle type: {restored.vehicle_type}")
