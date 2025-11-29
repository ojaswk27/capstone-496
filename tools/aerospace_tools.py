from typing import Dict, List, Any, Optional
from rag.document_store import DocumentStore
from rag.retriever import AerospaceRetriever
from rag.extractor import AerospaceExtractor
from tools.calculator import AerospaceCalculator


class AerospaceAgentTools:
    """
    Unified interface for the Aerospace Design Agent.
    Combines RAG (Search), Extraction (LLM), and Calculation (Math).
    """

    def __init__(self):
        print("🚀 Loading Aerospace Tools...")
        self.store = DocumentStore()
        self.retriever = AerospaceRetriever(self.store)
        self.extractor = AerospaceExtractor(self.retriever)
        self.calculator = AerospaceCalculator()

    def get_design_formula(self, query: str, vehicle_type: str) -> List[Dict]:
        """
        Tool 1: Finds relevant formulas for a design problem.
        Usage: "I need a formula for quadcopter thrust" -> Returns JSON formulas
        """
        print(f"🛠️  Tool Call: get_design_formula('{query}')")
        return self.extractor.extract_formulas(query, vehicle_type)

    def calculate_design_parameter(self, formula_expression: str, inputs: Dict[str, float]) -> float:
        """
        Tool 2: Executes a specific formula string.
        """
        print(f"🛠️  Tool Call: calculate_design_parameter")
        return self.calculator.calculate(formula_expression, inputs)

    def solve_physics_problem(self, query: str, vehicle_type: str, inputs: Dict[str, float]) -> Dict[str, Any]:
        """
        Tool 3 (High Level): "Auto-Solve"
        Retrieves formula AND calculates result in one step.
        """
        print(f"🛠️  Tool Call: solve_physics_problem('{query}')")

        # Try to extract what we're solving for from the query
        # e.g., "hover thrust" -> output = "thrust"
        # e.g., "delta v" -> output = "delta_v"
        query_lower = query.lower()
        desired_output = None

        if "thrust" in query_lower:
            desired_output = "thrust"
        elif "delta" in query_lower and "v" in query_lower:
            desired_output = "delta_v"
        elif "velocity" in query_lower or "speed" in query_lower:
            desired_output = "velocity"
        elif "power" in query_lower:
            desired_output = "power"
        elif "lift" in query_lower:
            desired_output = "lift"
        elif "drag" in query_lower:
            desired_output = "drag"

        # 1. Find Formula
        formulas = self.extractor.extract_formulas(query, vehicle_type, desired_output=desired_output)
        if not formulas:
            return {"error": "No suitable formula found."}

        # Pick the first/best formula
        # (In a real app, you might ask the user to confirm which one)
        best_formula = formulas[0]
        expr = best_formula["expression"]
        name = best_formula.get("name", "Unknown")

        # 2. Calculate
        result = self.calculator.calculate(expr, inputs)

        return {
            "formula_name": name,
            "formula_code": expr,
            "inputs_used": inputs,
            "result": result,
            "units": best_formula.get("units", {})
        }