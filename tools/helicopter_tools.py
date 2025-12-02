"""
Helicopter Calculation Tools

Specialized calculations for rotorcraft design:
- Rotor sizing and performance
- Hover power
- Forward flight
- Autorotation
- Weight estimation
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .common_tools import G, RHO_SL, air_density


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RotorResult:
    """Rotor design result."""
    diameter: float  # m
    num_blades: int
    chord: float  # m
    solidity: float
    disk_area: float  # m²
    tip_speed: float  # m/s
    rpm: float


@dataclass
class HoverResult:
    """Hover performance result."""
    power_ideal: float  # W
    power_actual: float  # W
    figure_of_merit: float
    induced_velocity: float  # m/s
    disk_loading: float  # N/m²
    thrust: float  # N


@dataclass
class ForwardFlightResult:
    """Forward flight performance."""
    power_required: float  # W
    power_induced: float  # W
    power_profile: float  # W
    power_parasite: float  # W
    max_speed: float  # m/s
    best_range_speed: float  # m/s


@dataclass
class HelicopterDesignResult:
    """Complete helicopter design result."""
    rotor: RotorResult
    hover: HoverResult
    forward_flight: ForwardFlightResult
    tail_rotor_diameter: float  # m
    engine_power: float  # W
    fuel_consumption: float  # kg/hr
    total_weight: float  # kg


# =============================================================================
# Rotor Design
# =============================================================================

def calculate_disk_loading(
    thrust: float,
    disk_area: float
) -> float:
    """
    Calculate disk loading.
    
    DL = T / A
    
    Args:
        thrust: Thrust (N)
        disk_area: Disk area (m²)
        
    Returns:
        Disk loading (N/m²)
    """
    return thrust / disk_area


def calculate_solidity(
    num_blades: int,
    chord: float,
    radius: float
) -> float:
    """
    Calculate rotor solidity.
    
    σ = N_b × c / (π × R)
    
    Args:
        num_blades: Number of blades
        chord: Blade chord (m)
        radius: Rotor radius (m)
        
    Returns:
        Solidity (dimensionless)
    """
    return num_blades * chord / (math.pi * radius)


def calculate_tip_speed(
    rpm: float,
    radius: float
) -> float:
    """
    Calculate rotor tip speed.
    
    V_tip = Ω × R = (2π × RPM / 60) × R
    
    Args:
        rpm: Rotor RPM
        radius: Rotor radius (m)
        
    Returns:
        Tip speed (m/s)
    """
    omega = 2 * math.pi * rpm / 60
    return omega * radius


def design_rotor(
    gross_weight: float,
    disk_loading_target: float = 300,
    num_blades: int = 4,
    tip_speed_target: float = 200,
    solidity_target: float = 0.08
) -> RotorResult:
    """
    Design main rotor from requirements.
    
    Args:
        gross_weight: Helicopter gross weight (kg)
        disk_loading_target: Target disk loading (N/m²)
        num_blades: Number of blades
        tip_speed_target: Target tip speed (m/s)
        solidity_target: Target solidity
        
    Returns:
        RotorResult with rotor parameters
    """
    thrust = gross_weight * G
    
    # Disk area from disk loading
    disk_area = thrust / disk_loading_target
    
    # Radius from area
    radius = math.sqrt(disk_area / math.pi)
    diameter = 2 * radius
    
    # RPM from tip speed
    omega = tip_speed_target / radius
    rpm = omega * 60 / (2 * math.pi)
    
    # Chord from solidity
    chord = solidity_target * math.pi * radius / num_blades
    
    # Actual solidity
    solidity = calculate_solidity(num_blades, chord, radius)
    
    return RotorResult(
        diameter=diameter,
        num_blades=num_blades,
        chord=chord,
        solidity=solidity,
        disk_area=disk_area,
        tip_speed=tip_speed_target,
        rpm=rpm
    )


# =============================================================================
# Hover Performance
# =============================================================================

def calculate_induced_velocity(
    thrust: float,
    disk_area: float,
    altitude: float = 0
) -> float:
    """
    Calculate induced velocity in hover using momentum theory.
    
    v_i = √(T / (2ρA))
    
    Args:
        thrust: Thrust (N)
        disk_area: Disk area (m²)
        altitude: Altitude (m)
        
    Returns:
        Induced velocity (m/s)
    """
    rho = air_density(altitude)
    return math.sqrt(thrust / (2 * rho * disk_area))


def calculate_ideal_hover_power(
    thrust: float,
    disk_area: float,
    altitude: float = 0
) -> float:
    """
    Calculate ideal hover power (momentum theory).
    
    P_ideal = T × v_i = T^(3/2) / √(2ρA)
    
    Args:
        thrust: Thrust (N)
        disk_area: Disk area (m²)
        altitude: Altitude (m)
        
    Returns:
        Ideal power (W)
    """
    v_i = calculate_induced_velocity(thrust, disk_area, altitude)
    return thrust * v_i


def calculate_hover_power(
    gross_weight: float,
    rotor: RotorResult,
    altitude: float = 0,
    figure_of_merit: float = 0.75
) -> HoverResult:
    """
    Calculate actual hover power.
    
    P_actual = P_ideal / FM
    
    Args:
        gross_weight: Gross weight (kg)
        rotor: RotorResult from rotor design
        altitude: Altitude (m)
        figure_of_merit: FM (0.70-0.80 typical)
        
    Returns:
        HoverResult with hover performance
    """
    thrust = gross_weight * G
    rho = air_density(altitude)
    
    # Disk loading
    disk_loading = thrust / rotor.disk_area
    
    # Induced velocity
    v_i = calculate_induced_velocity(thrust, rotor.disk_area, altitude)
    
    # Ideal power
    p_ideal = thrust * v_i
    
    # Actual power
    p_actual = p_ideal / figure_of_merit
    
    return HoverResult(
        power_ideal=p_ideal,
        power_actual=p_actual,
        figure_of_merit=figure_of_merit,
        induced_velocity=v_i,
        disk_loading=disk_loading,
        thrust=thrust
    )


def hover_ceiling(
    gross_weight: float,
    power_available: float,
    rotor: RotorResult,
    figure_of_merit: float = 0.75
) -> float:
    """
    Calculate hover ceiling (altitude where power required = power available).
    
    Uses iterative solution.
    
    Args:
        gross_weight: Weight (kg)
        power_available: Available power (W)
        rotor: RotorResult
        figure_of_merit: FM
        
    Returns:
        Hover ceiling (m)
    """
    thrust = gross_weight * G
    
    # Iterate to find altitude
    for alt in range(0, 10000, 100):
        rho = air_density(alt)
        v_i = math.sqrt(thrust / (2 * rho * rotor.disk_area))
        p_required = thrust * v_i / figure_of_merit
        
        if p_required > power_available:
            return alt - 100
    
    return 10000


# =============================================================================
# Forward Flight
# =============================================================================

def calculate_advance_ratio(
    forward_speed: float,
    tip_speed: float
) -> float:
    """
    Calculate advance ratio.
    
    μ = V / ΩR
    
    Args:
        forward_speed: Forward speed (m/s)
        tip_speed: Rotor tip speed (m/s)
        
    Returns:
        Advance ratio
    """
    return forward_speed / tip_speed


def calculate_forward_flight_power(
    gross_weight: float,
    rotor: RotorResult,
    forward_speed: float,
    flat_plate_area: float = 2.0,
    altitude: float = 0
) -> ForwardFlightResult:
    """
    Calculate power required in forward flight.
    
    P_total = P_induced + P_profile + P_parasite
    
    Args:
        gross_weight: Weight (kg)
        rotor: RotorResult
        forward_speed: Forward speed (m/s)
        flat_plate_area: Equivalent flat plate area (m²)
        altitude: Altitude (m)
        
    Returns:
        ForwardFlightResult with power breakdown
    """
    rho = air_density(altitude)
    thrust = gross_weight * G
    
    # Advance ratio
    mu = calculate_advance_ratio(forward_speed, rotor.tip_speed)
    
    # Induced velocity in forward flight (Glauert's formula)
    v_h = calculate_induced_velocity(thrust, rotor.disk_area, altitude)
    if forward_speed > 0:
        v_i = v_h ** 2 / math.sqrt(forward_speed ** 2 + v_h ** 2)
    else:
        v_i = v_h
    
    # Induced power
    p_induced = thrust * v_i
    
    # Profile power (blade drag)
    cd0 = 0.008  # Blade drag coefficient
    p_profile = (rho * rotor.disk_area * rotor.tip_speed ** 3 * 
                 rotor.solidity * cd0 / 8 * (1 + 4.65 * mu ** 2))
    
    # Parasite power (fuselage drag)
    p_parasite = 0.5 * rho * forward_speed ** 3 * flat_plate_area
    
    # Total power
    p_total = p_induced + p_profile + p_parasite
    
    # Maximum speed (where power available = required)
    # Simplified estimate
    max_speed = forward_speed * 1.5 if forward_speed > 0 else 80
    
    # Best range speed (minimum P/V)
    best_range_speed = forward_speed if forward_speed > 0 else 50
    
    return ForwardFlightResult(
        power_required=p_total,
        power_induced=p_induced,
        power_profile=p_profile,
        power_parasite=p_parasite,
        max_speed=max_speed,
        best_range_speed=best_range_speed
    )


# =============================================================================
# Tail Rotor
# =============================================================================

def calculate_tail_rotor_thrust(
    main_rotor_torque: float,
    tail_arm: float
) -> float:
    """
    Calculate tail rotor thrust required for torque balance.
    
    T_tr × l_tr = Q_mr
    
    Args:
        main_rotor_torque: Main rotor torque (N·m)
        tail_arm: Distance from main rotor to tail rotor (m)
        
    Returns:
        Tail rotor thrust (N)
    """
    return main_rotor_torque / tail_arm


def design_tail_rotor(
    main_rotor_power: float,
    main_rotor_rpm: float,
    tail_arm: float,
    disk_loading_factor: float = 3.0
) -> float:
    """
    Design tail rotor diameter.
    
    Args:
        main_rotor_power: Main rotor power (W)
        main_rotor_rpm: Main rotor RPM
        tail_arm: Distance to tail rotor (m)
        disk_loading_factor: Ratio of tail to main rotor disk loading
        
    Returns:
        Tail rotor diameter (m)
    """
    # Main rotor torque
    omega_mr = 2 * math.pi * main_rotor_rpm / 60
    torque = main_rotor_power / omega_mr
    
    # Tail rotor thrust
    thrust_tr = calculate_tail_rotor_thrust(torque, tail_arm)
    
    # Tail rotor disk loading (typically 3x main rotor)
    dl_tr = disk_loading_factor * 300  # Assuming 300 N/m² main rotor
    
    # Tail rotor area and diameter
    area_tr = thrust_tr / dl_tr
    diameter_tr = 2 * math.sqrt(area_tr / math.pi)
    
    return diameter_tr


# =============================================================================
# Autorotation
# =============================================================================

def calculate_autorotation_descent_rate(
    gross_weight: float,
    rotor: RotorResult,
    altitude: float = 0
) -> float:
    """
    Calculate descent rate in autorotation.
    
    Simplified model using momentum theory.
    
    Args:
        gross_weight: Weight (kg)
        rotor: RotorResult
        altitude: Altitude (m)
        
    Returns:
        Descent rate (m/s)
    """
    rho = air_density(altitude)
    thrust = gross_weight * G
    
    # Ideal induced velocity
    v_h = calculate_induced_velocity(thrust, rotor.disk_area, altitude)
    
    # In autorotation, descent rate is approximately 1.7-2.0 × v_h
    descent_rate = 1.8 * v_h
    
    return descent_rate


def calculate_autorotation_index(
    rotor: RotorResult,
    gross_weight: float
) -> float:
    """
    Calculate autorotation index.
    
    AI = (I_rotor × Ω²) / (W × R)
    
    Higher is better for autorotation.
    
    Args:
        rotor: RotorResult
        gross_weight: Weight (kg)
        
    Returns:
        Autorotation index
    """
    # Estimate rotor inertia (simplified)
    blade_mass = gross_weight * 0.01  # Rough estimate per blade
    radius = rotor.diameter / 2
    i_blade = blade_mass * radius ** 2 / 3  # Rod approximation
    i_rotor = rotor.num_blades * i_blade
    
    omega = rotor.tip_speed / radius
    weight = gross_weight * G
    
    return (i_rotor * omega ** 2) / (weight * radius)


# =============================================================================
# Complete Helicopter Design
# =============================================================================

def design_helicopter(
    payload_kg: float,
    range_km: float,
    cruise_speed_kmh: float
) -> HelicopterDesignResult:
    """
    Complete helicopter preliminary design.
    
    Args:
        payload_kg: Payload mass (kg)
        range_km: Required range (km)
        cruise_speed_kmh: Cruise speed (km/h)
        
    Returns:
        HelicopterDesignResult
    """
    cruise_speed = cruise_speed_kmh / 3.6  # m/s
    
    # Initial weight estimate (payload ~25% of GW)
    gross_weight = payload_kg / 0.25
    
    # Iterate on weight
    for _ in range(5):
        # Design rotor
        rotor = design_rotor(gross_weight, disk_loading_target=300)
        
        # Hover power
        hover = calculate_hover_power(gross_weight, rotor)
        
        # Forward flight power
        flat_plate_area = 0.02 * gross_weight ** 0.67  # Empirical
        forward = calculate_forward_flight_power(
            gross_weight, rotor, cruise_speed, flat_plate_area
        )
        
        # Engine sizing (with margin)
        engine_power = max(hover.power_actual, forward.power_required) * 1.25
        
        # Fuel consumption (SFC ~ 0.3 kg/kW-hr for turboshaft)
        sfc = 0.3e-3  # kg/W-hr
        fuel_rate = sfc * forward.power_required * 1000 / 1000  # kg/hr
        
        # Flight time and fuel
        flight_time = range_km / cruise_speed_kmh  # hours
        fuel_weight = fuel_rate * flight_time * 1.1  # 10% reserve
        
        # Weight breakdown
        empty_fraction = 0.55
        empty_weight = gross_weight * empty_fraction
        
        new_gw = empty_weight + payload_kg + fuel_weight + 70  # 70kg crew
        
        if abs(new_gw - gross_weight) < 10:
            break
        
        gross_weight = 0.5 * gross_weight + 0.5 * new_gw
    
    # Tail rotor
    tail_arm = rotor.diameter * 0.55
    tail_rotor_diameter = design_tail_rotor(
        hover.power_actual, rotor.rpm, tail_arm
    )
    
    return HelicopterDesignResult(
        rotor=rotor,
        hover=hover,
        forward_flight=forward,
        tail_rotor_diameter=tail_rotor_diameter,
        engine_power=engine_power,
        fuel_consumption=fuel_rate,
        total_weight=gross_weight
    )


# =============================================================================
# Tool Registry
# =============================================================================

HELICOPTER_TOOLS = {
    "design_rotor": {
        "function": design_rotor,
        "description": "Design main rotor from requirements",
        "parameters": {
            "gross_weight": "Helicopter weight (kg)",
            "disk_loading_target": "Target disk loading (N/m²)"
        },
        "returns": "RotorResult"
    },
    "calculate_hover_power": {
        "function": calculate_hover_power,
        "description": "Calculate hover power requirement",
        "parameters": {
            "gross_weight": "Weight (kg)",
            "rotor": "RotorResult",
            "altitude": "Altitude (m)"
        },
        "returns": "HoverResult"
    },
    "calculate_forward_flight_power": {
        "function": calculate_forward_flight_power,
        "description": "Calculate forward flight power",
        "parameters": {
            "gross_weight": "Weight (kg)",
            "rotor": "RotorResult",
            "forward_speed": "Speed (m/s)"
        },
        "returns": "ForwardFlightResult"
    },
    "calculate_autorotation_descent_rate": {
        "function": calculate_autorotation_descent_rate,
        "description": "Calculate autorotation descent rate",
        "parameters": {
            "gross_weight": "Weight (kg)",
            "rotor": "RotorResult"
        },
        "returns": "Descent rate (m/s)"
    },
    "design_helicopter": {
        "function": design_helicopter,
        "description": "Complete helicopter design",
        "parameters": {
            "payload_kg": "Payload (kg)",
            "range_km": "Range (km)",
            "cruise_speed_kmh": "Cruise speed (km/h)"
        },
        "returns": "HelicopterDesignResult"
    },
}


if __name__ == "__main__":
    print("=== Helicopter Tools Test ===\n")
    
    # Design a light helicopter
    print("Designing light helicopter:")
    print("  Payload: 400 kg")
    print("  Range: 500 km")
    print("  Cruise: 200 km/h")
    
    heli = design_helicopter(
        payload_kg=400,
        range_km=500,
        cruise_speed_kmh=200
    )
    
    print(f"\n  Main Rotor:")
    print(f"    Diameter: {heli.rotor.diameter:.2f} m")
    print(f"    Blades: {heli.rotor.num_blades}")
    print(f"    RPM: {heli.rotor.rpm:.0f}")
    print(f"    Tip speed: {heli.rotor.tip_speed:.0f} m/s")
    
    print(f"\n  Performance:")
    print(f"    Hover power: {heli.hover.power_actual/1000:.0f} kW")
    print(f"    Cruise power: {heli.forward_flight.power_required/1000:.0f} kW")
    print(f"    Engine: {heli.engine_power/1000:.0f} kW")
    
    print(f"\n  Weight:")
    print(f"    Gross weight: {heli.total_weight:.0f} kg")
    print(f"    Fuel rate: {heli.fuel_consumption:.1f} kg/hr")
