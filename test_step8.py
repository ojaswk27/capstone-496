from state import AgentState
from nodes.workflow_nodes import (
    classify_vehicle_node,
    extract_requirements_node,
    plan_calculations_node,
    run_calculations_node,
    generate_design_node,
)

def test_nodes_pipeline():
    state: AgentState = {
        "messages": ["User: I want a long-endurance photography drone that can carry a 2kg payload."],
        "vehicle_type": None,
        "design_goal": None,
        "requirements": [],
        "calculations": [],
        "current_step": "start",
        "is_complete": False,
        "error": None,
    }

    # Run a subset of nodes in sequence to confirm they work
    for node in [
        classify_vehicle_node,
        extract_requirements_node,
        plan_calculations_node,
        run_calculations_node,
        generate_design_node,
    ]:
        update = node(state)
        state.update(update)

    print("✅ Node pipeline ran. Current step:", state["current_step"])
    print("🔢 Requirements:", len(state["requirements"]))
    print("🧮 Calculations:", len(state["calculations"]))
    print("🗨️ Last message:", state["messages"][-1][:200], "...")

if __name__ == "__main__":
    test_nodes_pipeline()
