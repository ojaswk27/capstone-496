import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tools.aerospace_tools import AerospaceAgentTools


def test_integration():
    # Initialize the "Brain"
    tools = AerospaceAgentTools()

    print("\n🤖 SCENARIO: User wants to calculate Quadcopter Thrust")
    print("----------------------------------------------------")

    # User Inputs (e.g. from a front-end form)
    user_inputs = {
        "rho": 1.225,  # Air density
        "A": 0.2,  # Rotor disk area (m^2)
        "v_i": 12.5  # Induced velocity (m/s)
    }

    # Run the "Auto-Solve" tool
    # Note: The query must be clear so RAG finds the specific "Momentum Theory" formula
    response = tools.solve_physics_problem(
        query="calculate thrust using momentum theory",
        vehicle_type="drones",
        inputs=user_inputs
    )

    # Display Results
    print("\n✅ FINAL OUTPUT:")
    print(f"Formula Found: {response.get('formula_name')}")
    print(f"Python Code:   {response.get('formula_code')}")
    print(f"Result:        {response.get('result')} Newtons")


if __name__ == "__main__":
    test_integration()
