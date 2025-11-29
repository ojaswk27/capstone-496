from state import AgentState, DesignRequirement


def test_state_schema():
    print("🧠 Testing Agent Memory State...")

    # Simulate an empty state (Start of conversation)
    state: AgentState = {
        "messages": [],
        "vehicle_type": None,
        "design_goal": None,
        "requirements": [],
        "calculations": [],
        "current_step": "start",
        "is_complete": False,
        "error": None
    }

    print("✅ Empty state initialized successfully.")

    # Simulate updating state (User says "I want a drone")
    state["vehicle_type"] = "drone"
    state["design_goal"] = "Long range photography drone"
    state["messages"].append("User: I want a drone for photography.")

    print(f"📝 Goal Set: {state['design_goal']} ({state['vehicle_type']})")

    # Simulate adding a requirement
    req: DesignRequirement = {
        "parameter": "payload",
        "value": 2.5,
        "unit": "kg",
        "description": "Camera weight"
    }
    state["requirements"].append(req)

    print(f"📌 Requirement Added: {req['parameter']} = {req['value']} {req['unit']}")

    if len(state["requirements"]) == 1 and state["vehicle_type"] == "drone":
        print("\n🎉 Step 7 Complete: State Schema is valid and ready for LangGraph.")


if __name__ == "__main__":
    test_state_schema()
