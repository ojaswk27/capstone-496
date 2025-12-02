"""
Test Suite for Aerospace Design Assistant

Comprehensive tests for all components:
- Unit tests for calculation tools
- Integration tests for workflow
- Example design scenarios
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from typing import Dict, Any


# =============================================================================
# Tool Tests
# =============================================================================

class TestCommonTools(unittest.TestCase):
    """Test common aerospace calculations."""
    
    def test_isa_atmosphere(self):
        """Test ISA atmosphere model."""
        from tools.common_tools import isa_atmosphere
        
        # Sea level
        atm = isa_atmosphere(0)
        self.assertAlmostEqual(atm.temperature, 288.15, places=1)
        self.assertAlmostEqual(atm.pressure, 101325, places=0)
        self.assertAlmostEqual(atm.density, 1.225, places=2)
        
        # 10000m
        atm = isa_atmosphere(10000)
        self.assertLess(atm.temperature, 250)
        self.assertLess(atm.density, 0.5)
    
    def test_reynolds_number(self):
        """Test Reynolds number calculation."""
        from tools.common_tools import reynolds_number
        
        Re = reynolds_number(50, 1.0, 0)
        self.assertGreater(Re, 3e6)
        self.assertLess(Re, 4e6)
    
    def test_dynamic_pressure(self):
        """Test dynamic pressure."""
        from tools.common_tools import dynamic_pressure
        
        q = dynamic_pressure(50, 0)  # 50 m/s at sea level
        self.assertAlmostEqual(q, 0.5 * 1.225 * 50**2, places=0)


class TestDroneTools(unittest.TestCase):
    """Test drone calculation tools."""
    
    def test_hover_thrust(self):
        """Test hover thrust calculation."""
        from tools.drone_tools import calculate_hover_thrust
        
        thrust = calculate_hover_thrust(1.0, 4)  # 1kg, 4 motors
        self.assertAlmostEqual(thrust, 1.0 * 9.81 / 4, places=2)
    
    def test_flight_time(self):
        """Test flight time calculation."""
        from tools.drone_tools import calculate_flight_time
        
        time = calculate_flight_time(
            battery_capacity_mah=5000,
            battery_voltage=14.8,
            hover_power=200,
            usable_capacity=0.8
        )
        self.assertGreater(time, 15)
        self.assertLess(time, 40)
    
    def test_size_drone(self):
        """Test complete drone sizing."""
        from tools.drone_tools import size_drone
        
        design = size_drone(
            payload_kg=0.5,
            flight_time_minutes=20,
            num_motors=4,
            application="photography"
        )
        
        self.assertGreater(design.total_weight, 0.5)
        self.assertGreater(design.thrust_to_weight, 1.0)
        self.assertGreater(design.hover_time, 10)


class TestFixedWingTools(unittest.TestCase):
    """Test fixed-wing aircraft calculations."""
    
    def test_lift_calculation(self):
        """Test lift force calculation."""
        from tools.fixed_wing_tools import calculate_lift
        
        lift = calculate_lift(
            velocity=50,
            wing_area=10,
            cl=0.5,
            altitude=0
        )
        # L = 0.5 * 1.225 * 50^2 * 10 * 0.5 = 7656 N
        self.assertAlmostEqual(lift, 7656, delta=100)
    
    def test_stall_speed(self):
        """Test stall speed calculation."""
        from tools.fixed_wing_tools import calculate_stall_speed
        
        vs = calculate_stall_speed(
            weight=1000,
            wing_area=15,
            cl_max=1.5,
            altitude=0
        )
        self.assertGreater(vs, 20)
        self.assertLess(vs, 40)


class TestRocketTools(unittest.TestCase):
    """Test rocket propulsion calculations."""
    
    def test_delta_v(self):
        """Test Tsiolkovsky rocket equation."""
        from tools.rocket_tools import tsiolkovsky_delta_v
        
        dv = tsiolkovsky_delta_v(
            isp=250,
            mass_initial=100,
            mass_final=40
        )
        # dv = 250 * 9.81 * ln(100/40) = 2246 m/s
        self.assertAlmostEqual(dv, 2246, delta=50)
    
    def test_thrust_to_weight(self):
        """Test T/W calculation."""
        from tools.rocket_tools import calculate_thrust_to_weight
        
        tw = calculate_thrust_to_weight(thrust=1000, mass=50)
        self.assertAlmostEqual(tw, 1000 / (50 * 9.81), places=2)


class TestSatelliteTools(unittest.TestCase):
    """Test satellite/orbital calculations."""
    
    def test_orbital_velocity(self):
        """Test orbital velocity at LEO."""
        from tools.satellite_tools import calculate_orbital_velocity
        
        v = calculate_orbital_velocity(400e3)  # 400km
        self.assertGreater(v, 7500)
        self.assertLess(v, 8000)
    
    def test_orbital_period(self):
        """Test orbital period."""
        from tools.satellite_tools import calculate_orbital_period
        
        T = calculate_orbital_period(400e3)
        self.assertGreater(T / 60, 90)  # > 90 minutes
        self.assertLess(T / 60, 95)  # < 95 minutes


# =============================================================================
# Node Tests
# =============================================================================

class TestClassifier(unittest.TestCase):
    """Test vehicle classifier."""
    
    def test_drone_classification(self):
        """Test drone classification."""
        from nodes.classifier import classify_vehicle
        from nodes.state import VehicleType
        
        vtype, conf, _ = classify_vehicle(
            "Design a quadcopter drone for aerial photography"
        )
        self.assertEqual(vtype, VehicleType.DRONE)
        self.assertGreater(conf, 0.5)
    
    def test_rocket_classification(self):
        """Test rocket classification."""
        from nodes.classifier import classify_vehicle
        from nodes.state import VehicleType
        
        vtype, conf, _ = classify_vehicle(
            "Build a model rocket to reach 1km altitude"
        )
        self.assertEqual(vtype, VehicleType.ROCKET)
        self.assertGreater(conf, 0.5)
    
    def test_satellite_classification(self):
        """Test satellite classification."""
        from nodes.classifier import classify_vehicle
        from nodes.state import VehicleType
        
        vtype, conf, _ = classify_vehicle(
            "LEO satellite for earth observation, 400km orbit"
        )
        self.assertEqual(vtype, VehicleType.SATELLITE)
        self.assertGreater(conf, 0.5)


class TestParser(unittest.TestCase):
    """Test requirement parser."""
    
    def test_payload_extraction(self):
        """Test payload weight extraction."""
        from nodes.parser import parse_requirements
        from nodes.state import VehicleType
        
        req = parse_requirements(
            "Carry 2kg payload",
            VehicleType.DRONE
        )
        self.assertEqual(req.payload_kg, 2.0)
    
    def test_range_extraction(self):
        """Test range extraction."""
        from nodes.parser import parse_requirements
        from nodes.state import VehicleType
        
        req = parse_requirements(
            "Aircraft with 500km range",
            VehicleType.FIXED_WING
        )
        self.assertEqual(req.range_km, 500)
    
    def test_flight_time_extraction(self):
        """Test flight time extraction."""
        from nodes.parser import parse_requirements
        from nodes.state import VehicleType
        
        req = parse_requirements(
            "30 minutes flight time",
            VehicleType.DRONE
        )
        self.assertAlmostEqual(req.endurance_hours, 0.5, places=2)


# =============================================================================
# Integration Tests
# =============================================================================

class TestWorkflow(unittest.TestCase):
    """Test complete workflow integration."""
    
    def test_drone_design_workflow(self):
        """Test complete drone design."""
        from graph.workflow import AerospaceDesignWorkflow
        
        workflow = AerospaceDesignWorkflow()
        state = workflow.run(
            "Design a quadcopter drone with 500g payload for 20 minutes"
        )
        
        self.assertIsNotNone(state)
        self.assertIsNotNone(state.design_output)
    
    def test_rocket_design_workflow(self):
        """Test complete rocket design."""
        from graph.workflow import AerospaceDesignWorkflow
        
        workflow = AerospaceDesignWorkflow()
        state = workflow.run(
            "Model rocket to reach 1000m with 500g payload"
        )
        
        self.assertIsNotNone(state)


# =============================================================================
# Example Scenarios
# =============================================================================

def run_example_scenarios():
    """Run example design scenarios for demonstration."""
    from graph.workflow import AerospaceDesignWorkflow, get_design_summary
    
    scenarios = [
        {
            "name": "Photography Drone",
            "request": "Design a quadcopter drone that can carry a 500g camera "
                      "for 25 minutes of flight time. It should be stable for "
                      "aerial photography.",
        },
        {
            "name": "Private Aircraft",
            "request": "I need a small fixed-wing aircraft for 2 passengers "
                      "with 500km range and 200 km/h cruise speed.",
        },
        {
            "name": "Model Rocket",
            "request": "Build a model rocket to reach 1km altitude with a "
                      "500g payload for a science experiment.",
        },
        {
            "name": "Earth Observation Satellite",
            "request": "LEO satellite at 400km orbit for Earth observation, "
                      "50kg payload, 5 year mission life.",
        },
        {
            "name": "Competition Glider",
            "request": "Standard class competition glider with good thermal "
                      "performance and 40:1 glide ratio.",
        },
    ]
    
    workflow = AerospaceDesignWorkflow()
    
    print("=" * 70)
    print("AEROSPACE DESIGN ASSISTANT - EXAMPLE SCENARIOS")
    print("=" * 70)
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario['name']}")
        print(f"{'='*70}")
        print(f"\nRequest: {scenario['request']}\n")
        
        try:
            state = workflow.run(scenario['request'])
            print(get_design_summary(state))
        except Exception as e:
            print(f"Error: {e}")
        
        print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aerospace Design Assistant Tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--examples", action="store_true", help="Run example scenarios")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.examples or args.all:
        print("\n" + "=" * 70)
        print("RUNNING EXAMPLE SCENARIOS")
        print("=" * 70 + "\n")
        run_example_scenarios()
    
    if args.unit or args.integration or args.all or not any(vars(args).values()):
        print("\n" + "=" * 70)
        print("RUNNING UNIT TESTS")
        print("=" * 70 + "\n")
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        if args.unit or args.all or not any(vars(args).values()):
            suite.addTests(loader.loadTestsFromTestCase(TestCommonTools))
            suite.addTests(loader.loadTestsFromTestCase(TestDroneTools))
            suite.addTests(loader.loadTestsFromTestCase(TestFixedWingTools))
            suite.addTests(loader.loadTestsFromTestCase(TestRocketTools))
            suite.addTests(loader.loadTestsFromTestCase(TestSatelliteTools))
            suite.addTests(loader.loadTestsFromTestCase(TestClassifier))
            suite.addTests(loader.loadTestsFromTestCase(TestParser))
        
        if args.integration or args.all:
            suite.addTests(loader.loadTestsFromTestCase(TestWorkflow))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Summary
        print("\n" + "=" * 70)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print("=" * 70)
