from graph.design_graph import graph
from state import AgentState


def test_graph_debug():
    print("🚀 Testing LangGraph Workflow with DEBUG...")

    initial_state: AgentState = {
        "messages": ["User: I want a long-endurance photography drone that can carry a 2kg payload."],
        "vehicle_type": None,
        "design_goal": None,
        "requirements": [],
        "calculations": [],
        "current_step": "start",
        "is_complete": False,
        "error": None,
    }

    print("🎯 Input State:", initial_state["messages"][0])

    # Track the full state manually
    current_state: AgentState = initial_state.copy()
    step_num = 0

    # stream() yields dicts where keys = NODE NAMES, values = state updates from that node
    # We need to accumulate these into current_state
    for step_output in graph.stream(initial_state):
        step_num += 1

        print(f"\n{'=' * 60}")
        print(f"STEP {step_num}")
        print(f"{'=' * 60}")

        # step_output is a dict with ONE key (the node name) and ONE value (the updates)
        # e.g., {"classify_vehicle_node": {"vehicle_type": "drone", "messages": [...]}}

        node_name = list(step_output.keys())[0]
        node_updates = step_output[node_name]

        print(f"Node executed: {node_name}")
        print(f"Fields updated: {list(node_updates.keys())}")

        # Merge the updates into our current_state
        for field_name, field_value in node_updates.items():
            if field_name == "messages":
                # "messages" uses add reducer, so field_value is a list of NEW messages only
                current_state["messages"] = current_state.get("messages", []) + field_value
            else:
                # Other fields just overwrite
                current_state[field_name] = field_value

        # Now display the current state after this step
        print(f"\nCurrent Step: {current_state.get('current_step')}")
        print(f"Vehicle Type: {current_state.get('vehicle_type')}")
        print(f"Design Goal: {current_state.get('design_goal')}")
        print(f"Total Messages: {len(current_state.get('messages', []))}")
        print(f"Total Requirements: {len(current_state.get('requirements', []))}")
        print(f"Total Calculations: {len(current_state.get('calculations', []))}")

        # Show what was added THIS step
        if "messages" in node_updates:
            print("\n--- MESSAGES ADDED THIS STEP ---")
            for msg in node_updates["messages"]:
                print(f"  + {str(msg)[:100]}")

        if "requirements" in node_updates and node_updates["requirements"]:
            print("\n--- REQUIREMENTS ADDED THIS STEP ---")
            for req in node_updates["requirements"]:
                print(f"  - {req}")

        if "calculations" in node_updates and node_updates["calculations"]:
            print("\n--- CALCULATIONS ADDED THIS STEP ---")
            for calc in node_updates["calculations"]:
                print(f"  - {calc}")

        # Print the final state for verification
        if current_state.get("is_complete"):
            print("\n✅ DESIGN SESSION COMPLETE")
            break


if __name__ == "__main__":
    test_graph_debug()