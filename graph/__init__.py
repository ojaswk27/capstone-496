"""
LangGraph Workflow Package
==========================

This package contains the LangGraph workflow implementation
for the Aerospace Design Assistant.

Main components:
- state.py: Pydantic state models
- workflow.py: LangGraph workflow definition
- nodes.py: Node implementations
"""

from .nodes import (
    classify_vehicle,
    extract_formulas,
    parse_requirements,
    perform_calculations,
    search_documents,
    synthesize_output,
    validate_design,
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
    create_initial_state,
    should_iterate,
    update_phase,
)
from .workflow import (
    AerospaceDesignWorkflow,
    build_design_graph,
    design_vehicle,
    get_design_summary,
)

__all__ = [
    # State
    "DesignState",
    "DesignPhase",
    "VehicleType",
    "UserRequirements",
    "SearchResult",
    "ExtractedFormula",
    "CalculationResult",
    "DesignComponent",
    "ValidationResult",
    "DesignOutput",
    "create_initial_state",
    "update_phase",
    "add_error",
    "add_warning",
    "should_iterate",
    # Nodes
    "classify_vehicle",
    "parse_requirements",
    "search_documents",
    "extract_formulas",
    "perform_calculations",
    "validate_design",
    "synthesize_output",
    # Workflow
    "AerospaceDesignWorkflow",
    "build_design_graph",
    "design_vehicle",
    "get_design_summary",
]
