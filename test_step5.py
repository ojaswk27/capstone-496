from tools.calculator import AerospaceCalculator


def test_calculator():
    calc = AerospaceCalculator()

    print("🧮 Testing Calculator Module...\n")

    # Case 1: Simple Lift Calculation
    expr_lift = "0.5 * rho * v**2 * S * Cl"
    inputs_lift = {
        "rho": 1.225,  # Sea level density
        "v": 30,  # 30 m/s
        "S": 0.5,  # 0.5 m^2 wing
        "Cl": 1.2  # Lift coefficient
    }

    print(f"Formula: {expr_lift}")
    print(f"Inputs: {inputs_lift}")
    result = calc.calculate(expr_lift, inputs_lift)
    print(f"✅ Result: {result} Newtons\n")

    # Case 2: Rocket Delta-V
    expr_rocket = "Isp * 9.81 * math.log(m0 / mf)"
    inputs_rocket = {
        "Isp": 250,  # Seconds
        "m0": 100,  # Initial kg
        "mf": 20  # Final kg
    }

    print(f"Formula: {expr_rocket}")
    print(f"Inputs: {inputs_rocket}")
    result = calc.calculate(expr_rocket, inputs_rocket)
    print(f"✅ Result: {result:.2f} m/s\n")

    # Case 3: Error Handling (Missing Variable)
    print("Testing Error Handling:")
    err = calc.calculate("a * b", {"a": 5})  # Missing 'b'
    print(f"❌ Expected Error: {err}")


if __name__ == "__main__":
    test_calculator()
