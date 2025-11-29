from typing import TypedDict, cast
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes.workflow_nodes import (
    classify_vehicle_node,
    extract_requirements_node,
    plan_calculations_node,
    run_calculations_node,
    generate_design_node,
    review_design_node,
    refine_design_node,
    finalize_node,
)


# Define the routing logic
def route_after_classify(state: AgentState) -> str:
    """After classification, always go to requirement extraction."""
    return "extract_requirements_node"


def route_after_requirements(state: AgentState) -> str:
    """After requirements, always go to planning."""
    return "plan_calculations_node"


def route_after_plan(state: AgentState) -> str:
    """After planning, always go to calculation."""
    return "run_calculations_node"


def route_after_calculations(state: AgentState) -> str:
    """After calculations, always go to design generation."""
    return "generate_design_node"


def route_after_design(state: AgentState) -> str:
    """After design, always go to review."""
    return "review_design_node"


def route_after_review(state: AgentState) -> str:
    """
    Conditional routing: If the review found issues, go to refine.
    Otherwise, go to finalize.
    """
    # For this MVP, we'll just always refine once, then finalize
    # In a real system, you'd parse the review text for keywords like "issue", "problem", etc.
    if state.get("current_step") == "refine":
        # Already refined once, now finalize
        return "finalize_node"
    else:
        return "refine_design_node"


def route_after_refine(state: AgentState) -> str:
    """After refinement, loop back to review."""
    return "review_design_node"


def should_continue(state: AgentState) -> str:
    """Check if we should stop the graph."""
    if state.get("is_complete") or state.get("error"):
        return END
    return "continue"


# Build the graph
def build_graph() -> StateGraph:
    """Constructs and returns the LangGraph workflow."""

    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("classify_vehicle_node", classify_vehicle_node)
    workflow.add_node("extract_requirements_node", extract_requirements_node)
    workflow.add_node("plan_calculations_node", plan_calculations_node)
    workflow.add_node("run_calculations_node", run_calculations_node)
    workflow.add_node("generate_design_node", generate_design_node)
    workflow.add_node("review_design_node", review_design_node)
    workflow.add_node("refine_design_node", refine_design_node)
    workflow.add_node("finalize_node", finalize_node)

    # Set the entry point
    workflow.set_entry_point("classify_vehicle_node")

    # Add edges (the routing logic)
    workflow.add_conditional_edges(
        "classify_vehicle_node",
        route_after_classify,
        {
            "extract_requirements_node": "extract_requirements_node",
        },
    )

    workflow.add_conditional_edges(
        "extract_requirements_node",
        route_after_requirements,
        {
            "plan_calculations_node": "plan_calculations_node",
        },
    )

    workflow.add_conditional_edges(
        "plan_calculations_node",
        route_after_plan,
        {
            "run_calculations_node": "run_calculations_node",
        },
    )

    workflow.add_conditional_edges(
        "run_calculations_node",
        route_after_calculations,
        {
            "generate_design_node": "generate_design_node",
        },
    )

    workflow.add_conditional_edges(
        "generate_design_node",
        route_after_design,
        {
            "review_design_node": "review_design_node",
        },
    )

    workflow.add_conditional_edges(
        "review_design_node",
        route_after_review,
        {
            "refine_design_node": "refine_design_node",
            "finalize_node": "finalize_node",
        },
    )

    workflow.add_conditional_edges(
        "refine_design_node",
        route_after_refine,
        {
            "review_design_node": "review_design_node",
        },
    )

    # Finalize node leads to END
    workflow.add_edge("finalize_node", END)

    # Compile the graph
    return workflow.compile()


# Export the compiled graph
graph = build_graph()
