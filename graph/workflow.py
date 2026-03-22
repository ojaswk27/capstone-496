"""
LangGraph workflow for Aerospace Design Assistant.
Deterministic graph with LLM sub-agents.
"""
from langgraph.graph import END, StateGraph

from agents.design import design_agent
from agents.parameter import parameter_agent
from agents.understand import understand_agent
from agents.validate import validate_agent
from config import get_config
from graph.state import (
    DesignOutput,
    DesignPhase,
    DesignState,
    ValidationResult,
    create_initial_state,
)


def synthesize_output(state: DesignState) -> DesignState:
    """Format intermediate results into final DesignOutput. Pure code, no LLM."""
    state.phase = DesignPhase.SYNTHESIZING
    design = state.intermediate_results.get("design", {})
    req = state.requirements

    if not design:
        state.errors.append("No design data to synthesize")
        state.phase = DesignPhase.ERROR
        return state

    # Build summary
    vtype = state.vehicle_type
    payload_str = f"{req.payload_kg}kg payload" if req.payload_kg else "unspecified payload"
    summary = f"{vtype.replace('_', ' ').title()} design for {payload_str}."

    # Build weight breakdown from design data
    weight_breakdown = {}
    for key in design:
        if "weight" in key or "mass" in key:
            val = design[key]
            if isinstance(val, (int, float)):
                weight_breakdown[key] = float(val)

    validation = state.validation_result or ValidationResult(
        passed=True, warnings=state.warnings, errors=state.errors
    )

    state.design_output = DesignOutput(
        vehicle_type=vtype,
        summary=summary,
        specifications=design,
        performance={},
        weight_breakdown=weight_breakdown,
        validation=validation,
        confidence_score=state.classification_confidence,
    )
    state.phase = DesignPhase.COMPLETE
    return state


def route_after_validation(state: DesignState) -> str:
    """Route after validation: retry design or synthesize. Pure function — no state mutation."""
    cfg = get_config().llm
    if state.validation_result and not state.validation_result.passed:
        if state.retry_count <= cfg.max_validation_retries:
            return "design"
    return "synthesize"


def handle_error(state: DesignState) -> DesignState:
    """Terminal error handler."""
    state.phase = DesignPhase.ERROR
    if not state.errors:
        state.errors.append("Unknown error in workflow")
    return state


def route_after_understand(state: DesignState) -> str:
    """Route after understand: error or continue."""
    if state.phase == "error":
        return "error"
    return "parameter"


def build_design_graph():
    """Build the LangGraph workflow."""
    workflow = StateGraph(DesignState)

    # Nodes
    workflow.add_node("understand", understand_agent)
    workflow.add_node("parameter", parameter_agent)
    workflow.add_node("design", design_agent)
    workflow.add_node("validate", validate_agent)
    workflow.add_node("synthesize", synthesize_output)
    workflow.add_node("error", handle_error)

    # Entry
    workflow.set_entry_point("understand")

    # Edges
    workflow.add_conditional_edges(
        "understand",
        route_after_understand,
        {"parameter": "parameter", "error": "error"},
    )
    workflow.add_edge("parameter", "design")
    workflow.add_edge("design", "validate")
    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {"design": "design", "synthesize": "synthesize"},
    )
    workflow.add_edge("synthesize", END)
    workflow.add_edge("error", END)

    return workflow.compile()


class AerospaceDesignWorkflow:
    """High-level interface."""

    def __init__(self):
        self.graph = build_design_graph()

    def run(self, user_input: str, session_id: str = "") -> DesignState:
        initial = create_initial_state(user_input, session_id)
        result = self.graph.invoke(initial)
        if isinstance(result, dict):
            return DesignState(**result)
        return result
