#!/usr/bin/env python3
"""
Aerospace Design Assistant - Main Entry Point
==============================================

This module provides the command-line interface for the
AI-Powered Aerospace Design Assistant.

Usage:
    # Single design request
    python main.py --design "surveillance drone, 2kg payload, 60min flight"

    # Interactive mode
    python main.py --interactive

    # Batch processing
    python main.py --batch designs.json --output results/

    # Show configuration status
    python main.py --status
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    typer = None

from config import create_env_template, get_config

# Initialize Rich console for formatted output
console = Console() if RICH_AVAILABLE else None


def print_banner():
    """Print the application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║       AI-Powered Aerospace Design Assistant                   ║
    ║       MAT496 Capstone Project                                 ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Supported Vehicles:                                          ║
    ║    • Drones/Multicopters  • Fixed-Wing Aircraft               ║
    ║    • Helicopters          • Rockets                           ║
    ║    • Satellites           • Gliders                           ║
    ║    • Hybrid VTOL                                              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    if console:
        console.print(Panel(banner, style="bold blue"))
    else:
        print(banner)


def print_status():
    """Print configuration status."""
    config = get_config()
    config.print_status()


def run_interactive_mode():
    """Run the assistant in interactive mode."""
    print_banner()

    config = get_config()
    if config.fallback_mode:
        print("\n⚠️  Running in FALLBACK MODE (limited functionality)")
        print("   Set API keys in .env file for full features.\n")

    print("Welcome to the Aerospace Design Assistant!")
    print("Describe the flying vehicle you want to design.")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'help' for usage examples.\n")

    while True:
        try:
            user_input = input("🚀 Design Request > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nThank you for using Aerospace Design Assistant!")
                break

            if user_input.lower() == "help":
                print_help()
                continue

            if user_input.lower() == "status":
                print_status()
                continue

            # Process the design request
            process_design_request(user_input)

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def print_help():
    """Print help information with usage examples."""
    help_text = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    USAGE EXAMPLES                             ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  DRONE DESIGN:                                                ║
    ║    "Design a surveillance drone with 2kg camera payload       ║
    ║     and 60 minute flight time"                                ║
    ║                                                               ║
    ║  AIRCRAFT DESIGN:                                             ║
    ║    "Design a 2-person aircraft with 500km range and           ║
    ║     200 km/h cruise speed"                                    ║
    ║                                                               ║
    ║  ROCKET DESIGN:                                               ║
    ║    "Design a model rocket to reach 1km altitude with          ║
    ║     500g payload"                                             ║
    ║                                                               ║
    ║  SATELLITE DESIGN:                                            ║
    ║    "Design a 100kg Earth observation satellite for            ║
    ║     400km LEO orbit with 5-year mission"                      ║
    ║                                                               ║
    ║  HELICOPTER DESIGN:                                           ║
    ║    "Design a light helicopter for 4 passengers with           ║
    ║     300km range"                                              ║
    ║                                                               ║
    ║  GLIDER DESIGN:                                               ║
    ║    "Design a competition sailplane with 50:1 glide ratio"     ║
    ║                                                               ║
    ║  HYBRID VTOL:                                                 ║
    ║    "Design a VTOL aircraft with vertical takeoff and          ║
    ║     100km cruise range"                                       ║
    ║                                                               ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  COMMANDS:                                                    ║
    ║    help   - Show this help message                            ║
    ║    status - Show configuration status                         ║
    ║    quit   - Exit the application                              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(help_text)


def process_design_request(user_input: str):
    """Process a design request and generate output."""
    print(f"\n📋 Processing design request...")
    print(f'   Input: "{user_input}"\n')

    try:
        from graph.state import DesignPhase, DesignState, dict_to_design_state
        from graph.workflow import AerospaceDesignWorkflow, get_design_summary

        print("🔧 Running aerospace design workflow...\n")
        workflow = AerospaceDesignWorkflow()
        result = workflow.run(user_input)  # May return dict or DesignState

        # -------- FIX: Convert dict to DesignState if needed --------
        if isinstance(result, dict):
            result = dict_to_design_state(result)
        # ------------------------------------------------------------

        # ----------------  FIXED: Use Pydantic attributes not dict access  -----------------
        if (
            RICH_AVAILABLE
            and result.design_output
            and hasattr(result.design_output, "_rich_render")
        ):
            console.print(result.design_output._rich_render())
        else:
            # Fallback plain text output
            print("\n" + "=" * 60)
            print("DESIGN RESULT")
            print("=" * 60)

            # Handle both enum and string vehicle_type
            vtype = result.vehicle_type
            vtype_str = vtype.value if hasattr(vtype, "value") else str(vtype)
            print(f"\n🚀 Vehicle Type: {vtype_str.upper()}")

            print(
                f"📊 Classification Confidence: {result.classification_confidence:.0%}"
            )

            req = result.requirements
            if req:
                print(f"\n📋 Parsed Requirements:")
                print(f"   • Raw Input: {req.raw_input}")
                if req.payload_kg:
                    print(f"   • Payload: {req.payload_kg} kg")
                if req.endurance_hours:
                    print(f"   • Endurance: {req.endurance_hours} hours")
                if req.range_km:
                    print(f"   • Range: {req.range_km} km")
                if req.speed_kmh:
                    print(f"   • Speed: {req.speed_kmh} km/h")
                if req.altitude_m:
                    print(f"   • Altitude: {req.altitude_m} m")

            if result.design_output:
                do = result.design_output
                print(f"\n{do.summary}")
                print("\nSpecifications:")
                for k, v in do.specifications.items():
                    if isinstance(v, float):
                        print(f"   • {k}: {v:.2f}")
                    else:
                        print(f"   • {k}: {v}")

            if result.warnings:
                print("\n⚠️  Warnings:")
                for w in result.warnings:
                    print(f"   • {w}")

            if result.errors:
                print("\n❌ Errors:")
                for e in result.errors:
                    print(f"   • {e}")

            print("\n" + "=" * 60 + "\n")

    # -------------  FALLBACK  -------------
    except ImportError as e:
        print(f"⚠️  Import error: {e}")
        print("    Falling back to direct calculation mode...")
        return run_direct_calculation(user_input)

    except Exception as e:
        print(f"\n❌ Error during design: {e}")


def run_direct_calculation(user_input: str):
    """
    Run calculations directly without LangGraph workflow.
    Uses the calculation tools directly for reliable results.
    """
    # -------- FIX: Correct import path --------
    from graph.state import VehicleType

    # ------------------------------------------
    from nodes.classifier import classify_vehicle
    from nodes.parser import parse_requirements
    from tools import (
        design_glider,
        design_helicopter,
        design_rocket,
        design_satellite,
        size_aircraft,
        size_drone,
    )

    print("\n" + "=" * 60)


def run_single_design(design_request: str, output_path: Optional[str] = None):
    """Run a single design request from command line."""
    print_banner()
    process_design_request(design_request)

    # TODO: Save output to file if output_path specified


def run_batch_processing(batch_file: str, output_dir: str):
    """Process multiple design requests from a JSON file."""
    print_banner()
    print(f"📁 Processing batch file: {batch_file}")
    print(f"📂 Output directory: {output_dir}\n")

    # Load batch file
    batch_path = Path(batch_file)
    if not batch_path.exists():
        print(f"❌ Error: Batch file not found: {batch_file}")
        return

    with open(batch_path, "r") as f:
        designs = json.load(f)

    print(f"Found {len(designs)} design requests.\n")

    # Process each design
    results = []
    for i, design in enumerate(designs, 1):
        print(f"[{i}/{len(designs)}] Processing: {design.get('name', 'Unnamed')}")
        request = design.get("request", design.get("description", ""))
        result = fallback_design(request)  # Will be replaced with actual processing
        results.append({"input": design, "output": result})

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_path / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, indent=2, fp=f)

    print(f"\n✅ Results saved to: {output_file}")


# Main entry point
def main():
    """Main entry point for the application."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI-Powered Aerospace Design Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive
  python main.py --design "surveillance drone, 2kg payload, 60min"
  python main.py --batch designs.json --output results/
  python main.py --status
        """,
    )

    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )

    parser.add_argument("--design", "-d", type=str, help="Single design request")

    parser.add_argument(
        "--batch",
        "-b",
        type=str,
        help="Path to batch file (JSON) with multiple designs",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output",
        help="Output directory for results",
    )

    parser.add_argument(
        "--status", "-s", action="store_true", help="Show configuration status"
    )

    parser.add_argument("--init", action="store_true", help="Create .env template file")

    args = parser.parse_args()

    # Handle commands
    if args.init:
        create_env_template()
        return

    if args.status:
        print_banner()
        print_status()
        return

    if args.design:
        run_single_design(args.design, args.output)
        return

    if args.batch:
        run_batch_processing(args.batch, args.output)
        return

    if args.interactive or len(sys.argv) == 1:
        run_interactive_mode()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
