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

import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    typer = None

from config import get_config, create_env_template

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
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Aerospace Design Assistant!")
                break
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            if user_input.lower() == 'status':
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
    print(f"   Input: \"{user_input}\"\n")
    
    # This is a placeholder that will be replaced with actual graph execution
    # once the LangGraph workflow is implemented
    
    config = get_config()
    
    if config.fallback_mode:
        print("🔧 Using fallback mode (rule-based design)...")
        result = fallback_design(user_input)
    else:
        print("🤖 Running AI-powered design workflow...")
        # result = run_design_graph(user_input)  # Will be implemented later
        result = fallback_design(user_input)  # Placeholder
    
    # Display result
    print("\n" + "=" * 60)
    print("DESIGN RESULT")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60 + "\n")


def fallback_design(user_input: str) -> dict:
    """
    Fallback design function using rule-based logic.
    Used when LLM API is not available.
    """
    # Simple keyword-based vehicle classification
    user_lower = user_input.lower()
    
    vehicle_type = "unknown"
    if any(kw in user_lower for kw in ["drone", "quadcopter", "multirotor", "uav"]):
        vehicle_type = "drone"
    elif any(kw in user_lower for kw in ["airplane", "aircraft", "plane", "fixed-wing", "fixed wing"]):
        vehicle_type = "fixed_wing"
    elif any(kw in user_lower for kw in ["helicopter", "heli", "rotorcraft"]):
        vehicle_type = "helicopter"
    elif any(kw in user_lower for kw in ["rocket", "launch vehicle"]):
        vehicle_type = "rocket"
    elif any(kw in user_lower for kw in ["satellite", "spacecraft", "orbit"]):
        vehicle_type = "satellite"
    elif any(kw in user_lower for kw in ["glider", "sailplane", "soaring"]):
        vehicle_type = "glider"
    elif any(kw in user_lower for kw in ["vtol", "tiltrotor", "hybrid"]):
        vehicle_type = "hybrid_vtol"
    
    return {
        "status": "fallback_mode",
        "vehicle_type": vehicle_type,
        "input": user_input,
        "message": "Full design requires LLM API. Configure API keys in .env file.",
        "timestamp": datetime.now().isoformat(),
        "note": "This is a placeholder result. Implement LangGraph workflow for complete functionality."
    }


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
    
    with open(batch_path, 'r') as f:
        designs = json.load(f)
    
    print(f"Found {len(designs)} design requests.\n")
    
    # Process each design
    results = []
    for i, design in enumerate(designs, 1):
        print(f"[{i}/{len(designs)}] Processing: {design.get('name', 'Unnamed')}")
        request = design.get('request', design.get('description', ''))
        result = fallback_design(request)  # Will be replaced with actual processing
        results.append({
            "input": design,
            "output": result
        })
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
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
        """
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--design', '-d',
        type=str,
        help='Single design request'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Path to batch file (JSON) with multiple designs'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Show configuration status'
    )
    
    parser.add_argument(
        '--init',
        action='store_true',
        help='Create .env template file'
    )
    
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
