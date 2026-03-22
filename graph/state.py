"""
State schema for Aerospace Design Assistant.
Redesigned for RLM sub-agentic workflow — no RAG fields.
"""
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VehicleType(str, Enum):
    DRONE = "drone"
    FIXED_WING = "fixed_wing"
    HELICOPTER = "helicopter"
    ROCKET = "rocket"
    SATELLITE = "satellite"
    GLIDER = "glider"
    UNKNOWN = "unknown"


class DesignPhase(str, Enum):
    UNDERSTANDING = "understanding"
    PARAMETERIZING = "parameterizing"
    DESIGNING = "designing"
    VALIDATING = "validating"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    ERROR = "error"


class UserRequirements(BaseModel):
    raw_input: str = ""
    payload_kg: Optional[float] = None
    range_km: Optional[float] = None
    endurance_hours: Optional[float] = None
    speed_kmh: Optional[float] = None
    altitude_m: Optional[float] = None
    mission_type: Optional[str] = None
    max_weight_kg: Optional[float] = None
    vehicle_specific: Dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class ValidationResult(BaseModel):
    passed: bool
    checks: Dict[str, bool] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class DesignOutput(BaseModel):
    vehicle_type: str
    summary: str
    specifications: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    weight_breakdown: Dict[str, float] = Field(default_factory=dict)
    validation: Optional[ValidationResult] = None
    confidence_score: float = 0.0


class DesignState(BaseModel):
    """Main state passed between LangGraph nodes."""
    session_id: str = ""
    phase: DesignPhase = DesignPhase.UNDERSTANDING
    raw_input: str = ""
    requirements: Optional[UserRequirements] = None

    # Classification
    vehicle_type: VehicleType = VehicleType.UNKNOWN
    classification_confidence: float = 0.0
    classification_reasoning: str = ""

    # Agent communication
    agent_messages: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)

    # Design results
    intermediate_results: Dict[str, Any] = Field(default_factory=dict)
    validation_result: Optional[ValidationResult] = None
    validation_feedback: Optional[str] = None
    retry_count: int = 0

    # Output
    design_output: Optional[DesignOutput] = None

    # Diagnostics
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


def create_initial_state(user_input: str, session_id: str = "") -> DesignState:
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    return DesignState(
        session_id=session_id,
        raw_input=user_input,
        requirements=UserRequirements(raw_input=user_input),
    )
