"""
LangGraph Workflow for Aerospace Design Assistant
LLM-first (Anthropic Claude Sonnet 4-5-20250929) with graceful fallback.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from nodes.llm_data_validator import validate_and_correct_data_llm
from nodes.llm_parameter_completer import complete_parameters_with_llm

# ---------- peer import fix ----------------------------------
_repo_root = Path(__file__).resolve().parent.parent  # graph -> repo root
sys.path.insert(0, str(_repo_root))
# -------------------------------------------------------------

# LangGraph imports (with fallback for testing)
try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"

# --------------------------------------------------------------
# legacy node fallbacks (still inside graph/nodes from your pic)
from graph.nodes import (
    classify_vehicle,
    extract_formulas,
    parse_requirements,
    perform_calculations,
    search_documents,
    synthesize_output,
    validate_design,
)
from graph.state import (  # note the package prefix
    DesignPhase,
    DesignState,
    VehicleType,
    create_initial_state,
    should_iterate,
)
from nodes.llm_formula_extractor import extract_formulas_llm
from nodes.llm_search_generator import generate_queries_llm

# -------------------- LLM nodes (peer import) -----------------
from nodes.llm_supervisor import llm_supervisor_node
from nodes.llm_validator import validate_with_llm

# =============================================================================
# Conditional Routing Functions
# =============================================================================


def route_after_supervisor(state: DesignState) -> str:  # CHANGED (renamed)
    """
    Route based on LLM supervisor result.
    If LLM flagged invalid -> error
    Otherwise proceed to parsing.
    """
    if state.phase == DesignPhase.ERROR:
        return "error_handler"
    return "parse_requirements"


def route_after_validation(state: DesignState) -> str:
    if state.validation_result is None:
        return "synthesize"
    if state.validation_result.passed:
        return "synthesize"
    if should_iterate(state):
        return "refine_design"
    return "synthesize"


# =============================================================================
# Error Handling Nodes
# =============================================================================


def handle_error(state: DesignState) -> DesignState:
    state.phase = DesignPhase.ERROR
    if not state.errors:
        state.errors.append("An unknown error occurred in the design workflow")
    return state


def handle_low_confidence(state: DesignState) -> DesignState:
    vtype = (
        state.vehicle_type.value
        if hasattr(state.vehicle_type, "value")
        else str(state.vehicle_type)
    )
    state.warnings.append(
        f"Vehicle type '{vtype}' classified with low confidence "
        f"({state.classification_confidence:.0%}). Results may be less accurate."
    )
    return state


def refine_design(state: DesignState) -> DesignState:
    if state.validation_result:
        for error in state.validation_result.errors:
            if "weight" in error.lower():
                state.intermediate_results.setdefault("design", {})[
                    "weight_reduction_factor"
                ] = 0.9
            if "thrust" in error.lower() or "power" in error.lower():
                state.intermediate_results.setdefault("design", {})[
                    "power_margin_factor"
                ] = 1.2
    return state


# =============================================================================
# Graph Builder
# =============================================================================


def build_design_graph():
    if not LANGGRAPH_AVAILABLE:
        return None
    workflow = StateGraph(DesignState)

    # -------------------- NEW ENTRY POINT --------------------
    workflow.add_node("llm_supervisor", llm_supervisor_node)
    workflow.set_entry_point("llm_supervisor")
    # ---------------------------------------------------------

    # legacy nodes
    # legacy nodes
    workflow.add_node("parse_requirements", parse_requirements)
    workflow.add_node("complete_parameters", complete_parameters_with_llm)  # NEW
    workflow.add_node("search_documents", search_documents)

    workflow.add_node("extract_formulas", extract_formulas)

    # ========== NEW: Data validation node ==========
    workflow.add_node("validate_data", validate_and_correct_data_llm)
    # ================================================

    workflow.add_node("perform_calculations", perform_calculations)
    workflow.add_node("validate_design", validate_design)
    workflow.add_node("synthesize", synthesize_output)
    workflow.add_node("error_handler", handle_error)
    workflow.add_node("refine_design", refine_design)

    # conditional edges from NEW supervisor
    workflow.add_conditional_edges(
        "llm_supervisor",
        route_after_supervisor,  # CHANGED (renamed func)
        {
            "parse_requirements": "parse_requirements",
            "error_handler": "error_handler",
        },
    )

    workflow.add_edge("error_handler", END)

    # linear flow - ADD validation step
    workflow.add_edge("parse_requirements", "complete_parameters")  # NEW
    workflow.add_edge("complete_parameters", "search_documents")  # NEW
    workflow.add_edge("search_documents", "extract_formulas")
    workflow.add_edge("extract_formulas", "validate_data")  # NEW
    workflow.add_edge("validate_data", "perform_calculations")  # NEW
    workflow.add_edge("perform_calculations", "validate_design")

    # validation loop
    workflow.add_conditional_edges(
        "validate_design",
        route_after_validation,
        {"synthesize": "synthesize", "refine_design": "refine_design"},
    )
    workflow.add_edge("refine_design", "perform_calculations")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


# =============================================================================
# Simple Sequential Fallback (LangGraph not installed)
# =============================================================================


def run_simple_workflow(user_input: str) -> DesignState:
    state = create_initial_state(user_input)
    # LLM supervisor first
    state = llm_supervisor_node(state)
    if state.phase == DesignPhase.ERROR:
        state = handle_error(state)
        return state
    # continue legacy chain
    state = parse_requirements(state)
    state = complete_parameters_with_llm(state)  # NEW: LLM completes missing params
    state = search_documents(state)
    state = extract_formulas(state)
    state = validate_and_correct_data_llm(state)  # Validate retrieved data


# =============================================================================
# High-Level Interface (unchanged)
# =============================================================================


class AerospaceDesignWorkflow:
    def __init__(self):
        self.graph = None
        self.use_langgraph = LANGGRAPH_AVAILABLE
        if self.use_langgraph:
            try:
                self.graph = build_design_graph()
            except Exception as e:
                print(f"Warning: Could not build LangGraph workflow: {e}")
                self.use_langgraph = False

    def run(self, user_input: str, session_id: str = "") -> DesignState:
        initial_state = create_initial_state(user_input, session_id)
        if self.use_langgraph and self.graph:
            return self.graph.invoke(initial_state)
        return run_simple_workflow(user_input)

    def stream(self, user_input: str, session_id: str = ""):
        initial_state = create_initial_state(user_input, session_id)
        if self.use_langgraph and self.graph:
            for state in self.graph.stream(initial_state):
                yield state
        else:
            yield run_simple_workflow(user_input)


# =============================================================================
# Convenience Exports (unchanged)
# =============================================================================


def design_vehicle(user_input: str) -> DesignState:
    return AerospaceDesignWorkflow().run(user_input)


def get_design_summary(state: DesignState) -> str:
    if state.phase == DesignPhase.ERROR:
        return f"Design failed: {'; '.join(state.errors)}"
    if state.design_output is None:
        return "Design not completed"
    out = state.design_output
    vtype = (
        out.vehicle_type.value
        if hasattr(out.vehicle_type, "value")
        else str(out.vehicle_type)
    )
    lines = [
        f"=== {vtype.upper()} DESIGN ===",
        "",
        f"Summary: {out.summary}",
        "",
        "Specifications:",
    ]
    for k, v in out.specifications.items():
        lines.append(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    if out.weight_breakdown:
        lines += ["", "Weight Breakdown:"]
        for k, v in out.weight_breakdown.items():
            lines.append(f"  {k}: {v:.2f} kg")
    if out.validation.warnings:
        lines += ["", "Warnings:"] + [f"  - {w}" for w in out.validation.warnings]
    if out.validation.errors:
        lines += ["", "Errors:"] + [f"  - {e}" for e in out.validation.errors]
    lines += ["", f"Confidence: {out.confidence_score:.0%}"]
    return "\n".join(lines)


__all__ = [
    "AerospaceDesignWorkflow",
    "build_design_graph",
    "run_simple_workflow",
    "design_vehicle",
    "get_design_summary",
    "LANGGRAPH_AVAILABLE",
]

# ------------------------------------------------------------------
# Quick self-test (only runs when file executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Aerospace Design Workflow Test ===")
    print(f"LangGraph available: {LANGGRAPH_AVAILABLE}")
    for txt in ["4 kg fixed-wing drone, 4 h endurance"]:
        st = AerospaceDesignWorkflow().run(txt)
        print("\nInput :", txt)
        vtype = (
            st.vehicle_type.value
            if hasattr(st.vehicle_type, "value")
            else str(st.vehicle_type)
        )
        print("Vehicle:", vtype)
        print("Phase  :", st.phase)
        print("Confidence:", f"{st.classification_confidence:.0%}")
        if st.design_output:
            print(
                "Total weight:",
                st.intermediate_results.get("design", {}).get("total_weight_kg"),
            )
        if st.errors:
            print("Errors:", st.errors)
