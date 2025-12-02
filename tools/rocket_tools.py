"""
Rocket Calculation Tools

Specialized calculations for rocket design:
- Tsiolkovsky rocket equation (delta-v)
- Thrust and specific impulse
- Staging optimization
- Trajectory analysis
- Recovery system sizing
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .common_tools import MU_EARTH, R_EARTH, G, air_density

# =============================================================================
# Constants
# =============================================================================

# Standard gravitational acceleration
G0 = 9.80665  # m/s²


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DeltaVResult:
    """Delta-V calculation result."""

    delta_v: float  # m/s
    mass_ratio: float
    propellant_mass: float  # kg
    burnout_mass: float  # kg


@dataclass
class ThrustResult:
    """Thrust calculation result."""

    thrust: float  # N
    mass_flow_rate: float  # kg/s
    exhaust_velocity: float  # m/s
    specific_impulse: float  # s


@dataclass
class StageResult:
    """Single stage analysis result."""

    stage_number: int
    propellant_mass: float  # kg
    structural_mass: float  # kg
    delta_v: float  # m/s
    burn_time: float  # s
    thrust: float  # N


@dataclass
class RocketDesignResult:
    """Complete rocket design result."""

    stages: List[StageResult]
    total_delta_v: float  # m/s
    total_mass: float  # kg
    payload_mass: float  # kg
    payload_fraction: float
    max_altitude: float  # m (for suborbital)
    target_achieved: bool


@dataclass
class RecoveryResult:
    """Recovery system design result."""

    parachute_diameter: float  # m
    descent_rate: float  # m/s
    drogue_diameter: float  # m (optional)
    deploy_altitude: float  # m


# =============================================================================
# Propulsion Calculations
# =============================================================================


def tsiolkovsky_delta_v(isp: float, mass_initial: float, mass_final: float) -> float:
    """
    Calculate delta-v using Tsiolkovsky rocket equation.

    Δv = Isp × g₀ × ln(m₀/m_f)

    Args:
        isp: Specific impulse (s)
        mass_initial: Initial mass (kg)
        mass_final: Final/burnout mass (kg)

    Returns:
        Delta-v (m/s)
    """
    if mass_final <= 0 or mass_initial <= mass_final:
        return 0

    return isp * G0 * math.log(mass_initial / mass_final)


def calculate_delta_v(
    isp: float, propellant_mass: float, dry_mass: float, payload_mass: float = 0
) -> DeltaVResult:
    """
    Calculate delta-v and mass ratio.

    Args:
        isp: Specific impulse (s)
        propellant_mass: Propellant mass (kg)
        dry_mass: Dry mass of stage (kg)
        payload_mass: Payload mass (kg)

    Returns:
        DeltaVResult with delta-v and mass data
    """
    mass_initial = propellant_mass + dry_mass + payload_mass
    mass_final = dry_mass + payload_mass

    delta_v = tsiolkovsky_delta_v(isp, mass_initial, mass_final)
    mass_ratio = mass_initial / mass_final

    return DeltaVResult(
        delta_v=delta_v,
        mass_ratio=mass_ratio,
        propellant_mass=propellant_mass,
        burnout_mass=mass_final,
    )


def calculate_propellant_mass(
    delta_v_required: float, isp: float, dry_mass: float, payload_mass: float = 0
) -> float:
    """
    Calculate propellant mass required for given delta-v.

    Rearranging Tsiolkovsky:
    m_prop = m_f × (exp(Δv/(Isp×g₀)) - 1)

    Args:
        delta_v_required: Required delta-v (m/s)
        isp: Specific impulse (s)
        dry_mass: Dry mass (kg)
        payload_mass: Payload mass (kg)

    Returns:
        Required propellant mass (kg)
    """
    mass_final = dry_mass + payload_mass
    mass_ratio = math.exp(delta_v_required / (isp * G0))

    return mass_final * (mass_ratio - 1)


def calculate_thrust(
    mass_flow_rate: float,
    exhaust_velocity: float,
    exit_pressure: float = 0,
    ambient_pressure: float = 101325,
    exit_area: float = 0,
) -> ThrustResult:
    """
    Calculate thrust from propulsion parameters.

    F = ṁ × v_e + (P_e - P_a) × A_e

    Args:
        mass_flow_rate: Mass flow rate (kg/s)
        exhaust_velocity: Exhaust velocity (m/s)
        exit_pressure: Nozzle exit pressure (Pa)
        ambient_pressure: Ambient pressure (Pa)
        exit_area: Nozzle exit area (m²)

    Returns:
        ThrustResult with thrust data
    """
    # Momentum thrust
    momentum_thrust = mass_flow_rate * exhaust_velocity

    # Pressure thrust
    pressure_thrust = (exit_pressure - ambient_pressure) * exit_area

    # Total thrust
    thrust = momentum_thrust + pressure_thrust

    # Specific impulse
    isp = exhaust_velocity / G0

    return ThrustResult(
        thrust=thrust,
        mass_flow_rate=mass_flow_rate,
        exhaust_velocity=exhaust_velocity,
        specific_impulse=isp,
    )


def calculate_isp_from_thrust(thrust: float, mass_flow_rate: float) -> float:
    """
    Calculate specific impulse from thrust and mass flow.

    Isp = F / (ṁ × g₀)

    Args:
        thrust: Thrust (N)
        mass_flow_rate: Mass flow rate (kg/s)

    Returns:
        Specific impulse (s)
    """
    return thrust / (mass_flow_rate * G0)


def calculate_burn_time(propellant_mass: float, mass_flow_rate: float) -> float:
    """
    Calculate burn time.

    t_burn = m_prop / ṁ

    Args:
        propellant_mass: Propellant mass (kg)
        mass_flow_rate: Mass flow rate (kg/s)

    Returns:
        Burn time (s)
    """
    return propellant_mass / mass_flow_rate


def calculate_thrust_to_weight(thrust: float, mass: float) -> float:
    """
    Calculate thrust-to-weight ratio.

    T/W = F / (m × g₀)

    Args:
        thrust: Thrust (N)
        mass: Mass (kg)

    Returns:
        Thrust-to-weight ratio
    """
    return thrust / (mass * G0)


# =============================================================================
# Staging Calculations
# =============================================================================


def analyze_stage(
    propellant_mass: float,
    structural_fraction: float,
    isp: float,
    thrust: float,
    payload_mass: float,
    stage_number: int = 1,
) -> StageResult:
    """
    Analyze a single rocket stage.

    Args:
        propellant_mass: Propellant mass (kg)
        structural_fraction: Structure mass / (structure + propellant)
        isp: Specific impulse (s)
        thrust: Stage thrust (N)
        payload_mass: Payload mass for this stage (kg)
        stage_number: Stage number (1, 2, etc.)

    Returns:
        StageResult with stage analysis
    """
    # Structural mass from structural fraction
    # ε = m_s / (m_s + m_p) → m_s = ε × m_p / (1 - ε)
    structural_mass = structural_fraction * propellant_mass / (1 - structural_fraction)

    # Calculate delta-v
    dv_result = calculate_delta_v(isp, propellant_mass, structural_mass, payload_mass)

    # Mass flow rate from thrust and Isp
    mass_flow_rate = thrust / (isp * G0)

    # Burn time
    burn_time = calculate_burn_time(propellant_mass, mass_flow_rate)

    return StageResult(
        stage_number=stage_number,
        propellant_mass=propellant_mass,
        structural_mass=structural_mass,
        delta_v=dv_result.delta_v,
        burn_time=burn_time,
        thrust=thrust,
    )


# REPLACE the entire `optimize_staging` function in tools/rocket_tools.py starting at line 316


def optimize_staging(
    total_delta_v: float,
    payload_mass: float,
    num_stages: int,
    isp_stages: List[float],
    structural_fractions: List[float],
) -> List[StageResult]:
    """
    Optimize staging for minimum total mass.

    Uses correct rocket equation with realistic structural fractions.
    """
    stages = []
    dv_per_stage = total_delta_v / num_stages
    current_payload_mass = payload_mass

    for i in range(num_stages - 1, -1, -1):
        isp = isp_stages[i] if i < len(isp_stages) else isp_stages[-1]
        eps = (
            structural_fractions[i]
            if i < len(structural_fractions)
            else structural_fractions[-1]
        )

        # Mass ratio from rocket equation
        mass_ratio = math.exp(dv_per_stage / (isp * G0))

        # Correct formula for propellant mass:
        # m_prop = m_payload × (MR-1) × (1-eps) / (1-eps×MR)
        numerator = current_payload_mass * (mass_ratio - 1) * (1 - eps)
        denominator = 1 - eps * mass_ratio

        if denominator <= 0 or numerator < 0:
            print(f"⚠️  Stage {i + 1} not feasible (MR={mass_ratio:.2f}, eps={eps:.2f})")
            # Fallback to safer calculation
            m_prop = current_payload_mass * (mass_ratio - 1) * 0.7
            m_struct = m_prop * 0.2
        else:
            m_prop = numerator / denominator
            m_struct = eps * m_prop / (1 - eps)

        # Enforce minimum structural mass (can't build rocket case for less)
        min_struct_mass = max(0.050, payload_mass * 0.02, m_prop * 0.10)
        if m_struct < min_struct_mass:
            print(
                f"   Adjusting stage {i + 1} structure from {m_struct:.3f}kg to {min_struct_mass:.3f}kg (minimum)"
            )
            m_struct = min_struct_mass
            # Recalculate propellant with realistic structure
            m_prop = (current_payload_mass + m_struct) * (mass_ratio - 1)

        # Thrust: T/W = 5-8 for solid motors, 1.2-2 for liquid
        total_stage_mass = m_prop + m_struct + current_payload_mass
        if isp < 250:  # Solid motor
            thrust_to_weight = 6.0
        else:  # Liquid motor
            thrust_to_weight = 1.5
        thrust = thrust_to_weight * total_stage_mass * G0

        # Burn time
        mass_flow_rate = thrust / (isp * G0)
        burn_time = m_prop / mass_flow_rate if mass_flow_rate > 0 else 1.0
        burn_time = max(0.5, min(burn_time, 60.0))  # Clamp to reasonable range

        stage = StageResult(
            stage_number=i + 1,
            propellant_mass=m_prop,
            structural_mass=m_struct,
            payload_mass=current_payload_mass,
            delta_v=dv_per_stage,
            mass_ratio=mass_ratio,
            isp=isp,
            thrust=thrust,
            burn_time=burn_time,
        )

        stages.insert(0, stage)
        current_payload_mass = total_stage_mass

    return stages


def estimate_burn_time(propellant_mass: float, thrust: float, isp: float) -> float:
    """Estimate burn time from propellant mass and thrust."""
    # mdot = Thrust / (Isp × g0)
    mass_flow_rate = thrust / (isp * G0)
    if mass_flow_rate <= 0:
        return 0.0
    burn_time = propellant_mass / mass_flow_rate
    return max(0.5, min(burn_time, 30.0))  # Clamp to reasonable range


# =============================================================================
# Trajectory Calculations
# =============================================================================


def calculate_gravity_loss(burn_time: float, thrust_to_weight: float) -> float:
    """
    Estimate gravity loss during vertical ascent.

    Δv_gravity ≈ g₀ × t_burn × (1 - T/W factor)

    Args:
        burn_time: Total burn time (s)
        thrust_to_weight: Initial T/W ratio

    Returns:
        Gravity loss (m/s)
    """
    # Simplified gravity loss model
    # Actual loss depends on trajectory
    if thrust_to_weight <= 1:
        return burn_time * G0  # Maximum loss

    return burn_time * G0 * 0.5  # Approximate for typical trajectories


def calculate_drag_loss(
    velocity_max: float,
    cd: float = 0.5,
    area: float = 0.01,
    mass: float = 10,
    altitude_scale: float = 8500,
) -> float:
    """
    Estimate drag loss during ascent.

    Simplified model using average conditions.

    Args:
        velocity_max: Maximum velocity (m/s)
        cd: Drag coefficient
        area: Reference area (m²)
        mass: Mass (kg)
        altitude_scale: Atmospheric scale height (m)

    Returns:
        Drag loss (m/s)
    """
    # Very simplified drag loss estimate
    # Actual requires trajectory integration
    rho = air_density(0)
    q_max = 0.5 * rho * (velocity_max * 0.5) ** 2  # Approximate max q
    drag = q_max * cd * area

    return drag / mass * 30  # Rough estimate


def calculate_max_altitude(
    delta_v: float,
    launch_altitude: float = 0,
    gravity_loss: float = 0,
    drag_loss: float = 0,
) -> float:
    """
    Calculate maximum altitude for vertical launch.

    h_max = v² / (2g) for ideal case

    Args:
        delta_v: Available delta-v (m/s)
        launch_altitude: Starting altitude (m)
        gravity_loss: Gravity loss (m/s)
        drag_loss: Drag loss (m/s)

    Returns:
        Maximum altitude (m)
    """
    effective_dv = delta_v - gravity_loss - drag_loss

    if effective_dv <= 0:
        return launch_altitude

    # Ballistic apex altitude
    # Using h = v²/(2g) with g varying with altitude
    # Simplified to constant g for low altitudes
    coast_altitude = effective_dv**2 / (2 * G0)

    return launch_altitude + coast_altitude


def calculate_orbital_velocity(altitude: float) -> float:
    """
    Calculate circular orbital velocity at given altitude.

    v = √(μ/r)

    Args:
        altitude: Altitude above Earth surface (m)

    Returns:
        Orbital velocity (m/s)
    """
    r = R_EARTH + altitude
    return math.sqrt(MU_EARTH / r)


def delta_v_to_orbit(target_altitude: float, launch_latitude: float = 28.5) -> float:
    """
    Estimate delta-v required to reach orbit.

    Args:
        target_altitude: Target orbital altitude (m)
        launch_latitude: Launch site latitude (degrees)

    Returns:
        Required delta-v (m/s)
    """
    # Orbital velocity at target altitude
    v_orbit = calculate_orbital_velocity(target_altitude)

    # Earth's rotation velocity at launch site
    v_earth = 465 * math.cos(math.radians(launch_latitude))  # m/s at equator is 465

    # Gravity loss (typical 1200-1800 m/s)
    gravity_loss = 1500

    # Drag loss (typical 50-150 m/s)
    drag_loss = 100

    # Total delta-v needed
    return v_orbit - v_earth + gravity_loss + drag_loss


# =============================================================================
# Recovery System Calculations
# =============================================================================


def calculate_descent_rate(
    weight: float, parachute_diameter: float, cd: float = 0.8, altitude: float = 0
) -> float:
    """
    Calculate parachute descent rate.

    V_d = √(2W / (ρ × C_D × A))

    Args:
        weight: Weight (kg)
        parachute_diameter: Parachute diameter (m)
        cd: Drag coefficient (0.75-0.85 typical)
        altitude: Altitude for density (m)

    Returns:
        Descent rate (m/s)
    """
    rho = air_density(altitude)
    area = math.pi * (parachute_diameter / 2) ** 2
    weight_n = weight * G0

    return math.sqrt(2 * weight_n / (rho * cd * area))


def size_parachute(
    weight: float, target_descent_rate: float, cd: float = 0.8, altitude: float = 0
) -> float:
    """
    Size parachute for target descent rate.

    D = √(8W / (π × ρ × C_D × V_d²))

    Args:
        weight: Weight (kg)
        target_descent_rate: Target descent rate (m/s)
        cd: Drag coefficient
        altitude: Altitude for density (m)

    Returns:
        Required parachute diameter (m)
    """
    rho = air_density(altitude)
    weight_n = weight * G0

    area = 2 * weight_n / (rho * cd * target_descent_rate**2)
    diameter = 2 * math.sqrt(area / math.pi)

    return diameter


def design_recovery_system(
    weight: float,
    main_descent_rate: float = 5.0,
    drogue_descent_rate: float = 20.0,
    main_deploy_altitude: float = 300,
) -> RecoveryResult:
    """
    Design complete recovery system.

    Args:
        weight: Rocket weight (kg)
        main_descent_rate: Target main chute descent (m/s)
        drogue_descent_rate: Drogue descent rate (m/s)
        main_deploy_altitude: Main deployment altitude (m)

    Returns:
        RecoveryResult with parachute sizes
    """
    main_diameter = size_parachute(weight, main_descent_rate)
    drogue_diameter = size_parachute(weight, drogue_descent_rate)

    # Verify descent rates
    actual_main_descent = calculate_descent_rate(weight, main_diameter)

    return RecoveryResult(
        parachute_diameter=main_diameter,
        descent_rate=actual_main_descent,
        drogue_diameter=drogue_diameter,
        deploy_altitude=main_deploy_altitude,
    )


# =============================================================================
# Complete Rocket Design
# =============================================================================


def design_rocket(
    payload_kg: float, target_altitude: float, motor_type: str = "solid"
) -> RocketDesignResult:
    """
    Complete rocket design for target altitude.

    Args:
        payload_kg: Payload mass (kg)
        target_altitude: Target altitude (m)
        motor_type: "solid", "liquid", "hybrid"

    Returns:
        RocketDesignResult with complete design
    """
    # Isp based on motor type
    isp_values = {
        "solid": 220,
        "hybrid": 250,
        "liquid": 300,
    }
    isp = isp_values.get(motor_type, 220)

    # Structural fraction based on motor type
    struct_fractions = {
        "solid": 0.15,
        "hybrid": 0.20,
        "liquid": 0.10,
    }
    struct_frac = struct_fractions.get(motor_type, 0.15)

    # Estimate required delta-v (simplified)
    # For suborbital: v = √(2gh) ideally, plus losses
    ideal_velocity = math.sqrt(2 * G0 * target_altitude)
    gravity_loss = ideal_velocity * 0.3  # Rough estimate
    drag_loss = min(100, ideal_velocity * 0.05)
    required_dv = ideal_velocity + gravity_loss + drag_loss

    # Determine staging
    if target_altitude < 3000:
        num_stages = 1
    elif target_altitude < 30000:
        num_stages = 1
    else:
        num_stages = 2

    # Design stages
    stages = optimize_staging(
        total_delta_v=required_dv,
        payload_mass=payload_kg,
        num_stages=num_stages,
        isp_stages=[isp] * num_stages,
        structural_fractions=[struct_frac] * num_stages,
    )

    # Calculate totals
    total_dv = sum(s.delta_v for s in stages)
    total_mass = payload_kg
    for stage in stages:
        total_mass += stage.propellant_mass + stage.structural_mass

    payload_fraction = payload_kg / total_mass

    # Predicted altitude
    max_alt = calculate_max_altitude(
        total_dv, gravity_loss=gravity_loss, drag_loss=drag_loss
    )

    return RocketDesignResult(
        stages=stages,
        total_delta_v=total_dv,
        total_mass=total_mass,
        payload_mass=payload_kg,
        payload_fraction=payload_fraction,
        max_altitude=max_alt,
        target_achieved=max_alt >= target_altitude * 0.9,
    )


# =============================================================================
# Tool Registry for LangGraph
# =============================================================================

ROCKET_TOOLS = {
    "tsiolkovsky_delta_v": {
        "function": tsiolkovsky_delta_v,
        "description": "Calculate delta-v using rocket equation",
        "parameters": {
            "isp": "Specific impulse (s)",
            "mass_initial": "Initial mass (kg)",
            "mass_final": "Final mass (kg)",
        },
        "returns": "Delta-v (m/s)",
    },
    "calculate_delta_v": {
        "function": calculate_delta_v,
        "description": "Calculate delta-v with mass breakdown",
        "parameters": {
            "isp": "Specific impulse (s)",
            "propellant_mass": "Propellant mass (kg)",
            "dry_mass": "Dry mass (kg)",
            "payload_mass": "Payload mass (kg)",
        },
        "returns": "DeltaVResult",
    },
    "calculate_thrust": {
        "function": calculate_thrust,
        "description": "Calculate thrust from propulsion parameters",
        "parameters": {
            "mass_flow_rate": "Mass flow (kg/s)",
            "exhaust_velocity": "Exhaust velocity (m/s)",
        },
        "returns": "ThrustResult",
    },
    "calculate_thrust_to_weight": {
        "function": calculate_thrust_to_weight,
        "description": "Calculate T/W ratio",
        "parameters": {"thrust": "Thrust (N)", "mass": "Mass (kg)"},
        "returns": "T/W ratio",
    },
    "size_parachute": {
        "function": size_parachute,
        "description": "Size parachute for target descent rate",
        "parameters": {
            "weight": "Weight (kg)",
            "target_descent_rate": "Target descent (m/s)",
        },
        "returns": "Parachute diameter (m)",
    },
    "design_rocket": {
        "function": design_rocket,
        "description": "Complete rocket design for altitude target",
        "parameters": {
            "payload_kg": "Payload (kg)",
            "target_altitude": "Target altitude (m)",
            "motor_type": "Motor type",
        },
        "returns": "RocketDesignResult",
    },
    "delta_v_to_orbit": {
        "function": delta_v_to_orbit,
        "description": "Estimate delta-v for orbital insertion",
        "parameters": {
            "target_altitude": "Orbital altitude (m)",
            "launch_latitude": "Launch latitude (deg)",
        },
        "returns": "Required delta-v (m/s)",
    },
}


if __name__ == "__main__":
    print("=== Rocket Tools Test ===\n")

    # Test Tsiolkovsky equation
    print("Tsiolkovsky equation test:")
    dv = tsiolkovsky_delta_v(isp=250, mass_initial=100, mass_final=40)
    print(f"  Isp=250s, MR=2.5 → Δv = {dv:.0f} m/s")

    # Design a model rocket
    print("\nDesigning model rocket:")
    print("  Payload: 0.5 kg")
    print("  Target: 1000 m altitude")

    design = design_rocket(payload_kg=0.5, target_altitude=1000, motor_type="solid")

    print(f"\n  Results:")
    print(f"    Stages: {len(design.stages)}")
    print(f"    Total mass: {design.total_mass:.2f} kg")
    print(f"    Total Δv: {design.total_delta_v:.0f} m/s")
    print(f"    Predicted altitude: {design.max_altitude:.0f} m")
    print(f"    Payload fraction: {design.payload_fraction:.1%}")

    # Recovery system
    print("\nRecovery system for 1kg rocket:")
    recovery = design_recovery_system(weight=1.0, main_descent_rate=5.0)
    print(f"  Main chute: {recovery.parachute_diameter:.2f} m diameter")
    print(f"  Drogue: {recovery.drogue_diameter:.2f} m diameter")
    print(f"  Descent rate: {recovery.descent_rate:.1f} m/s")
