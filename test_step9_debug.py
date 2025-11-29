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

    # Run the graph step by step with manual iteration
    current_state = initial_state

    for i, step_output in enumerate(graph.stream(initial_state)):
        for node_name, updated_state in step_output.items():
            print(f"\n{'=' * 60}")
            print(f"STEP {i + 1}: NODE: {node_name}")
            print(f"{'=' * 60}")
            print(f"Current Step: {updated_state.get('current_step')}")
            print(f"Messages added: {len(updated_state.get('messages', []))}")
            print(f"Requirements: {len(updated_state.get('requirements', []))}")
            print(f"Calculations: {len(updated_state.get('calculations', []))}")

            # Show the actual messages being added
            if updated_state.get('messages'):
                print(f"\n--- NEW MESSAGES ---")
                for msg in updated_state['messages']:
                    print(f"  + {msg[:100]}")

            # Show requirements if any
            if updated_state.get('requirements'):
                print(f"\n--- REQUIREMENTS ---")
                for req in updated_state['requirements']:
                    print(f"  - {req}")

            # Show calculations if any
            if updated_state.get('calculations'):
                print(f"\n--- CALCULATIONS ---")
                for calc in updated_state['calculations']:
                    print(f"  - {calc}")

            # Update current_state for next iteration
            current_state = updated_state


if __name__ == "__main__":
    test_graph_debug()
