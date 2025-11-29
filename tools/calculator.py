import math
from typing import Dict, Any, Union


class AerospaceCalculator:
    """
    Safely executes formulas extracted by the RAG system.
    """

    def __init__(self):
        # Define allowed math functions for safety
        self.safe_globals = {
            "math": math,
            "__builtins__": {},  # Prevent access to open(), import, etc.
            "pi": math.pi,
            "e": math.e,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pow": math.pow,
            "abs": abs
        }

        # Default physical constants
        self.constants = {
            "g": 9.81,  # Gravity (m/s^2)
            "rho_sl": 1.225,  # Sea level air density (kg/m^3)
            "R": 287.05,  # Specific gas constant for dry air
            "SL_temp": 288.15  # Sea level standard temperature (Kelvin)
        }

    def calculate(self, expression: str, inputs: Dict[str, float]) -> Union[float, str]:
        """
        Evaluates a mathematical expression with given variable inputs.
        """
        try:
            # 1. Merge Defaults with User Inputs
            # We start with the constants (defaults)
            context = self.constants.copy()

            # We update with user inputs (so user can override 'g' if on Mars)
            context.update(inputs)

            # 2. Execute
            # We pass 'context' as the local scope
            result = eval(expression, self.safe_globals, context)
            return result

        except NameError as e:
            return f"Missing variable: {e}"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except SyntaxError:
            return "Error: Invalid formula syntax"
        except Exception as e:
            return f"Calculation Error: {e}"

    def batch_calculate(self, formulas: list, inputs: Dict[str, float]) -> Dict[str, Any]:
        """
        Runs multiple formulas against the same inputs.
        """
        results = {}
        for f in formulas:
            name = f.get("name", "Unknown Formula")
            expr = f.get("expression")

            # Check if we have all required vars for this specific formula
            required_vars = f.get("variables", {}).keys()

            # FIX: Check if variable is in inputs OR safe_globals OR constants
            missing = []
            for v in required_vars:
                if (v not in inputs) and (v not in self.safe_globals) and (v not in self.constants):
                    missing.append(v)

            if missing:
                results[name] = f"Skipped (Missing: {', '.join(missing)})"
            else:
                val = self.calculate(expr, inputs)
                results[name] = val

        return results