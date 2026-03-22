#!/usr/bin/env python3
"""
Aerospace Design Assistant — Main Entry Point.

Usage:
    python main.py --design "surveillance drone, 2kg payload, 60min flight"
    python main.py --interactive
    python main.py --status
"""
import argparse
import sys

from config import get_config


def print_banner():
    print("""
    ========================================================
    AI-Powered Aerospace Design Assistant
    Vehicles: Drones | Aircraft | Helicopters | Rockets
              Satellites | Gliders
    ========================================================
    """)


def process_design(user_input: str):
    """Run the design workflow and print results."""
    print(f"\nProcessing: \"{user_input}\"\n")

    from graph.workflow import AerospaceDesignWorkflow

    workflow = AerospaceDesignWorkflow()
    result = workflow.run(user_input)

    if result.phase == "error":
        print(f"\nDesign failed:")
        for e in result.errors:
            print(f"  - {e}")
        return

    if result.design_output:
        do = result.design_output
        print(f"\nVehicle Type: {do.vehicle_type.upper()}")
        print(f"Confidence: {do.confidence_score:.0%}")
        print(f"\n{do.summary}")
        print(f"\nSpecifications:")
        print("-" * 50)
        for k, v in do.specifications.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            elif isinstance(v, dict):
                continue
            elif isinstance(v, list):
                continue
            else:
                print(f"  {k}: {v}")

        if do.weight_breakdown:
            print(f"\nWeight Breakdown:")
            for k, v in do.weight_breakdown.items():
                print(f"  {k}: {v:.2f} kg")

        if do.validation:
            if do.validation.warnings:
                print(f"\nWarnings:")
                for w in do.validation.warnings:
                    print(f"  - {w}")
            if do.validation.errors:
                print(f"\nErrors:")
                for e in do.validation.errors:
                    print(f"  - {e}")

    print()


def run_interactive():
    print_banner()
    print("Describe the vehicle you want to design. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Design > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            process_design(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Aerospace Design Assistant")
    parser.add_argument("--design", "-d", type=str, help="Single design request")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--status", "-s", action="store_true", help="Show config status")
    args = parser.parse_args()

    if args.status:
        print_banner()
        get_config().print_status()
    elif args.design:
        print_banner()
        process_design(args.design)
    elif args.interactive or len(sys.argv) == 1:
        run_interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
