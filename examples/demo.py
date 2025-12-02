#!/usr/bin/env python3
"""
Aerospace Design Assistant - Demo Script

Demonstrates the full capabilities of the system with
example designs for each vehicle type.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 70
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)


def print_subheader(text: str):
    """Print a formatted subheader."""
    print(f"\n--- {text} ---\n")


def demo_tools():
    """Demonstrate the calculation tools."""
    print_header("CALCULATION TOOLS DEMONSTRATION")
    
    # Drone tools
    print_subheader("Drone Sizing")
    from tools.drone_tools import size_drone
    
    drone = size_drone(
        payload_kg=0.5,
        flight_time_minutes=25,
        num_motors=4,
        application="photography"
    )
    print(f"  Payload: 0.5 kg")
    print(f"  Target flight time: 25 minutes")
    print(f"  Results:")
    print(f"    Frame size: {drone.frame_size} mm")
    print(f"    Motors: {drone.num_motors}x {drone.motor_kv} KV")
    print(f"    Props: {drone.prop_diameter}x{drone.prop_pitch} inch")
    print(f"    Battery: {drone.battery_cells}S {drone.battery_capacity:.0f}mAh")
    print(f"    Total weight: {drone.total_weight:.2f} kg")
    print(f"    T/W ratio: {drone.thrust_to_weight:.2f}")
    print(f"    Hover time: {drone.hover_time:.1f} min")
    
    # Fixed-wing tools
    print_subheader("Aircraft Sizing")
    from tools.fixed_wing_tools import size_aircraft
    
    aircraft = size_aircraft(
        payload_kg=200,
        range_km=800,
        cruise_speed_kmh=250,
        aircraft_type="single_engine_ga"
    )
    print(f"  Payload: 200 kg")
    print(f"  Range: 800 km")
    print(f"  Cruise speed: 250 km/h")
    print(f"  Results:")
    print(f"    Wing span: {aircraft.wing.span:.1f} m")
    print(f"    Wing area: {aircraft.wing.area:.1f} m²")
    print(f"    MTOW: {aircraft.total_weight:.0f} kg")
    print(f"    Stall speed: {aircraft.performance.stall_speed:.1f} m/s")
    print(f"    Power required: {aircraft.power_required/1000:.0f} kW")
    
    # Rocket tools
    print_subheader("Rocket Design")
    from tools.rocket_tools import design_rocket
    
    rocket = design_rocket(
        payload_kg=0.5,
        target_altitude=2000,
        motor_type="solid"
    )
    print(f"  Payload: 0.5 kg")
    print(f"  Target altitude: 2000 m")
    print(f"  Results:")
    print(f"    Total mass: {rocket.total_mass:.2f} kg")
    print(f"    Stages: {len(rocket.stages)}")
    print(f"    Total delta-v: {rocket.total_delta_v:.0f} m/s")
    print(f"    Max altitude: {rocket.max_altitude:.0f} m")
    print(f"    Target achieved: {rocket.target_achieved}")
    
    # Satellite tools
    print_subheader("Satellite Design")
    from tools.satellite_tools import design_satellite
    
    satellite = design_satellite(
        payload_power=100,
        payload_mass=50,
        altitude=400000,
        mission_years=5
    )
    print(f"  Payload: 50 kg, 100 W")
    print(f"  Orbit: 400 km LEO")
    print(f"  Mission: 5 years")
    print(f"  Results:")
    print(f"    Orbital velocity: {satellite.orbit.velocity:.0f} m/s")
    print(f"    Period: {satellite.orbit.period/60:.0f} min")
    print(f"    Solar array: {satellite.power.solar_array_area:.1f} m²")
    print(f"    Battery: {satellite.power.battery_capacity:.0f} Wh")
    print(f"    Total mass: {satellite.total_mass:.0f} kg")
    
    # Glider tools
    print_subheader("Glider Design")
    from tools.glider_tools import design_glider
    
    glider = design_glider(
        pilot_weight=80,
        target_glide_ratio=40,
        glider_class="standard"
    )
    print(f"  Pilot: 80 kg")
    print(f"  Class: Standard (15m)")
    print(f"  Results:")
    print(f"    Wing span: {glider.wing_span:.1f} m")
    print(f"    Wing area: {glider.wing_area:.1f} m²")
    print(f"    Best L/D: {glider.best_glide_ratio:.0f}")
    print(f"    Min sink: {glider.min_sink_rate:.2f} m/s")
    print(f"    Stall speed: {glider.stall_speed:.1f} m/s")


def demo_workflow():
    """Demonstrate the full design workflow."""
    print_header("DESIGN WORKFLOW DEMONSTRATION")
    
    from graph.workflow import AerospaceDesignWorkflow, get_design_summary
    
    test_requests = [
        {
            "name": "Photography Drone",
            "request": "Design a quadcopter drone that can carry a 500g GoPro camera "
                      "for 25 minutes of flight time."
        },
        {
            "name": "High Altitude Rocket",
            "request": "Build a model rocket to reach 1.5 km altitude with a "
                      "500g science payload."
        },
        {
            "name": "LEO Satellite",
            "request": "LEO satellite at 400km orbit for Earth observation, "
                      "50kg payload, 5 year mission."
        },
    ]
    
    workflow = AerospaceDesignWorkflow()
    
    for test in test_requests:
        print_subheader(test["name"])
        print(f"Request: {test['request']}\n")
        
        state = workflow.run(test["request"])
        
        print(f"Vehicle Type: {state.vehicle_type.value.upper()}")
        print(f"Confidence: {state.classification_confidence:.0%}")
        print(f"Phase: {state.phase.value}")
        
        if state.design_output:
            print(f"\nDesign Summary:")
            print(f"  {state.design_output.summary}")
            
            if state.design_output.specifications:
                print(f"\nKey Specifications:")
                for key, value in list(state.design_output.specifications.items())[:5]:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
            
            print(f"\nConfidence: {state.design_output.confidence_score:.0%}")
        
        if state.warnings:
            print(f"\nWarnings: {state.warnings}")
        
        if state.errors:
            print(f"\nErrors: {state.errors}")


def demo_classification():
    """Demonstrate vehicle classification."""
    print_header("VEHICLE CLASSIFICATION DEMONSTRATION")
    
    from nodes.classifier import classify_vehicle
    
    test_inputs = [
        "Design a hexacopter drone for aerial photography with 2kg payload",
        "I need a small 2-seat airplane with 500km range for weekend trips",
        "Build a model rocket to reach 3000 feet with a camera payload",
        "Earth observation satellite in LEO at 400km with 5 year mission",
        "Competition glider with high L/D ratio for cross-country soaring",
        "Light helicopter for 4 passengers with 300km range",
        "Flying vehicle with vertical takeoff capability",  # Ambiguous
    ]
    
    print("\nClassification Results:\n")
    
    for text in test_inputs:
        vtype, conf, reason = classify_vehicle(text)
        print(f"Input: {text[:55]}...")
        print(f"  → {vtype.value.upper()} ({conf:.0%})")
        print(f"    Reason: {reason}")
        print()


def demo_parsing():
    """Demonstrate requirement parsing."""
    print_header("REQUIREMENT PARSING DEMONSTRATION")
    
    from nodes.parser import parse_requirements
    from nodes.state import VehicleType
    
    test_cases = [
        ("Drone with 2kg payload for 30 minutes at 100m altitude", VehicleType.DRONE),
        ("Aircraft for 4 passengers, 800km range, 300 km/h cruise", VehicleType.FIXED_WING),
        ("Rocket to reach 5km altitude with 1kg payload", VehicleType.ROCKET),
        ("400km LEO satellite, 100kg, 7 year mission", VehicleType.SATELLITE),
    ]
    
    print("\nParsed Requirements:\n")
    
    for text, vtype in test_cases:
        req = parse_requirements(text, vtype)
        print(f"Input: {text}")
        print(f"  Type: {vtype.value}")
        
        # Print non-None fields
        for field in ['payload_kg', 'range_km', 'endurance_hours', 'speed_kmh',
                      'altitude_m', 'target_altitude_m', 'orbit_altitude_km',
                      'num_passengers', 'mission_years']:
            value = getattr(req, field, None)
            if value is not None:
                print(f"  {field}: {value}")
        print()


def main():
    """Run all demonstrations."""
    print_header("AI-POWERED AEROSPACE DESIGN ASSISTANT", "═")
    print(f"\n  MAT496 Capstone Project")
    print(f"  Demo generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Run demos
    demo_classification()
    demo_parsing()
    demo_tools()
    demo_workflow()
    
    print_header("DEMO COMPLETE", "═")
    print("\nThe system successfully demonstrates:")
    print("  ✓ Vehicle type classification from natural language")
    print("  ✓ Requirement extraction and parsing")
    print("  ✓ 44 aerospace calculation tools")
    print("  ✓ Complete design workflow with validation")
    print("  ✓ Structured output generation")
    print()


if __name__ == "__main__":
    main()
