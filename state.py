from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add


class DesignRequirement(TypedDict):
    """Stores a single user requirement."""
    parameter: str  # e.g., "flight_time"
    value: float  # e.g., 30.0
    unit: str  # e.g., "minutes"
    description: str  # e.g., "Maximum hover duration"


class CalculationResult(TypedDict):
    """Stores a calculated value."""
    variable: str  # e.g., "thrust_hover"
    value: float  # e.g., 76.56
    unit: str  # e.g., "N"
    formula_used: str  # e.g., "momentum_theory"


class AgentState(TypedDict):
    """
    The complete memory of the Aerospace Design Agent.
    This dict is passed to every node in the LangGraph workflow.
    """
    # 1. Conversation History
    # 'add' reducer allows appending messages instead of overwriting
    messages: Annotated[List[str], add]

    # 2. The Goal
    vehicle_type: Optional[str]  # "drone", "rocket", "glider", etc.
    design_goal: Optional[str]  # "Build a surveillance drone"

    # 3. Structured Data
    requirements: List[DesignRequirement]  # Constraints extracted from chat
    calculations: List[CalculationResult]  # Results computed by tools

    # 4. Workflow Control
    current_step: str  # "requirements", "research", "calculation", "review"
    is_complete: bool  # True when design is finished
    error: Optional[str]  # If something goes wrong
