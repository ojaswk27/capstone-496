"""
Satellite/Spacecraft Calculation Tools

Specialized calculations for satellite design:
- Orbital mechanics
- Power system sizing
- Thermal analysis
- Communication link budget
- Attitude control
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .common_tools import G, MU_EARTH, R_EARTH


# =============================================================================
# Constants
# =============================================================================

# Solar constant at 1 AU
SOLAR_CONSTANT = 1361  # W/m²

# Earth parameters
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s

# GEO altitude
GEO_ALTITUDE = 35786e3  # m


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class OrbitResult:
    """Orbital parameters result."""
    altitude: float  # m
    semi_major_axis: float  # m
    velocity: float  # m/s
    period: float  # s
    inclination: float  # degrees
    orbit_type: str


@dataclass
class PowerResult:
    """Power system sizing result."""
    solar_array_area: float  # m²
    battery_capacity: float  # Wh
    average_power: float  # W
    eclipse_duration: float  # s
    solar_array_power: float  # W


@dataclass
class ThermalResult:
    """Thermal analysis result."""
    hot_case_temp: float  # K
    cold_case_temp: float  # K
    heater_power: float  # W
    radiator_area: float  # m²


@dataclass
class LinkBudgetResult:
    """Communication link budget result."""
    eirp: float  # dBW
    path_loss: float  # dB
    received_power: float  # dBW
    snr: float  # dB
    margin: float  # dB
    data_rate: float  # bps


@dataclass
class SatelliteDesignResult:
    """Complete satellite design result."""
    orbit: OrbitResult
    power: PowerResult
    thermal: ThermalResult
    total_mass: float  # kg
    dimensions: Tuple[float, float, float]  # m
    design_life: float  # years


# =============================================================================
# Orbital Mechanics
# =============================================================================

def calculate_orbital_velocity(altitude: float) -> float:
    """
    Calculate circular orbital velocity.
    
    v = √(μ/r)
    
    Args:
        altitude: Altitude above Earth surface (m)
        
    Returns:
        Orbital velocity (m/s)
    """
    r = R_EARTH + altitude
    return math.sqrt(MU_EARTH / r)


def calculate_orbital_period(altitude: float) -> float:
    """
    Calculate orbital period.
    
    T = 2π × √(a³/μ)
    
    Args:
        altitude: Altitude (m)
        
    Returns:
        Orbital period (s)
    """
    a = R_EARTH + altitude
    return 2 * math.pi * math.sqrt(a ** 3 / MU_EARTH)


def calculate_orbit_parameters(
    altitude: float,
    inclination: float = 0
) -> OrbitResult:
    """
    Calculate all orbital parameters.
    
    Args:
        altitude: Orbital altitude (m)
        inclination: Orbital inclination (degrees)
        
    Returns:
        OrbitResult with orbital parameters
    """
    a = R_EARTH + altitude
    v = calculate_orbital_velocity(altitude)
    period = calculate_orbital_period(altitude)
    
    # Determine orbit type
    if altitude < 2000e3:
        orbit_type = "LEO"
    elif altitude < 35000e3:
        orbit_type = "MEO"
    elif abs(altitude - GEO_ALTITUDE) < 100e3:
        orbit_type = "GEO"
    else:
        orbit_type = "HEO"
    
    return OrbitResult(
        altitude=altitude,
        semi_major_axis=a,
        velocity=v,
        period=period,
        inclination=inclination,
        orbit_type=orbit_type
    )


def calculate_eclipse_duration(altitude: float) -> float:
    """
    Calculate maximum eclipse duration for circular orbit.
    
    Args:
        altitude: Orbital altitude (m)
        
    Returns:
        Eclipse duration (s)
    """
    r = R_EARTH + altitude
    period = calculate_orbital_period(altitude)
    
    # Eclipse half-angle
    rho = math.asin(R_EARTH / r)
    
    # Eclipse fraction of orbit
    eclipse_fraction = rho / math.pi
    
    return period * eclipse_fraction


def calculate_ground_track_repeat(
    altitude: float,
    inclination: float = 98.0
) -> int:
    """
    Calculate ground track repeat period for sun-synchronous orbit.
    
    Args:
        altitude: Orbital altitude (m)
        inclination: Inclination (degrees)
        
    Returns:
        Number of orbits for ground track repeat
    """
    period = calculate_orbital_period(altitude)
    orbits_per_day = 86400 / period
    
    # For sun-synchronous, typically 14-16 orbits/day
    return round(orbits_per_day)


def hohmann_transfer_delta_v(
    altitude_1: float,
    altitude_2: float
) -> Tuple[float, float]:
    """
    Calculate delta-v for Hohmann transfer.
    
    Args:
        altitude_1: Initial altitude (m)
        altitude_2: Final altitude (m)
        
    Returns:
        (delta_v_1, delta_v_2) in m/s
    """
    r1 = R_EARTH + altitude_1
    r2 = R_EARTH + altitude_2
    
    # Transfer orbit semi-major axis
    a_transfer = (r1 + r2) / 2
    
    # Velocities
    v1_initial = math.sqrt(MU_EARTH / r1)
    v1_transfer = math.sqrt(MU_EARTH * (2 / r1 - 1 / a_transfer))
    
    v2_transfer = math.sqrt(MU_EARTH * (2 / r2 - 1 / a_transfer))
    v2_final = math.sqrt(MU_EARTH / r2)
    
    delta_v_1 = abs(v1_transfer - v1_initial)
    delta_v_2 = abs(v2_final - v2_transfer)
    
    return delta_v_1, delta_v_2


def calculate_station_keeping_delta_v(
    altitude: float,
    years: float,
    orbit_type: str = "LEO"
) -> float:
    """
    Estimate station-keeping delta-v budget.
    
    Args:
        altitude: Altitude (m)
        years: Mission duration (years)
        orbit_type: "LEO", "GEO", etc.
        
    Returns:
        Delta-v budget (m/s)
    """
    # Typical annual station-keeping budgets
    annual_budgets = {
        "LEO": 25,  # m/s per year (mostly drag makeup)
        "MEO": 5,
        "GEO": 50,  # North-south and east-west
        "HEO": 10,
    }
    
    annual_dv = annual_budgets.get(orbit_type, 20)
    return annual_dv * years


# =============================================================================
# Power System
# =============================================================================

def calculate_solar_array_power(
    array_area: float,
    efficiency: float = 0.28,
    degradation_per_year: float = 0.02,
    years: float = 5,
    sun_angle: float = 0
) -> float:
    """
    Calculate solar array power output.
    
    P = S × A × η × cos(θ) × (1 - degradation)^years
    
    Args:
        array_area: Solar array area (m²)
        efficiency: Cell efficiency (0.20-0.32)
        degradation_per_year: Annual degradation rate
        years: Time since deployment
        sun_angle: Sun incidence angle (degrees)
        
    Returns:
        Power output (W)
    """
    degradation_factor = (1 - degradation_per_year) ** years
    angle_factor = math.cos(math.radians(sun_angle))
    
    return SOLAR_CONSTANT * array_area * efficiency * degradation_factor * angle_factor


def size_solar_array(
    average_power: float,
    efficiency: float = 0.28,
    mission_years: float = 5,
    eclipse_fraction: float = 0.35,
    sun_angle_max: float = 23.5
) -> float:
    """
    Size solar array for power requirement.
    
    Args:
        average_power: Average power required (W)
        efficiency: Solar cell efficiency
        mission_years: Mission duration for degradation
        eclipse_fraction: Fraction of orbit in eclipse
        sun_angle_max: Maximum sun angle (degrees)
        
    Returns:
        Required array area (m²)
    """
    # Account for eclipse - must generate extra during sunlight
    sunlight_fraction = 1 - eclipse_fraction
    required_generation = average_power / sunlight_fraction
    
    # End-of-life degradation
    degradation = (1 - 0.02) ** mission_years
    
    # Sun angle factor
    angle_factor = math.cos(math.radians(sun_angle_max))
    
    # Required area
    area = required_generation / (SOLAR_CONSTANT * efficiency * degradation * angle_factor)
    
    return area


def size_battery(
    average_power: float,
    eclipse_duration: float,
    depth_of_discharge: float = 0.3,
    battery_efficiency: float = 0.9
) -> float:
    """
    Size battery for eclipse power.
    
    Args:
        average_power: Power during eclipse (W)
        eclipse_duration: Eclipse duration (s)
        depth_of_discharge: Allowable DOD (0.2-0.4)
        battery_efficiency: Round-trip efficiency
        
    Returns:
        Battery capacity (Wh)
    """
    eclipse_hours = eclipse_duration / 3600
    energy_required = average_power * eclipse_hours
    
    # Account for efficiency and DOD
    capacity = energy_required / (depth_of_discharge * battery_efficiency)
    
    return capacity


def design_power_system(
    average_power: float,
    altitude: float,
    mission_years: float = 5
) -> PowerResult:
    """
    Complete power system design.
    
    Args:
        average_power: Average power requirement (W)
        altitude: Orbital altitude (m)
        mission_years: Mission duration (years)
        
    Returns:
        PowerResult with power system sizing
    """
    # Eclipse duration
    eclipse_duration = calculate_eclipse_duration(altitude)
    
    # Eclipse fraction
    period = calculate_orbital_period(altitude)
    eclipse_fraction = eclipse_duration / period
    
    # Size solar array
    array_area = size_solar_array(
        average_power,
        eclipse_fraction=eclipse_fraction,
        mission_years=mission_years
    )
    
    # Size battery
    battery_capacity = size_battery(
        average_power,
        eclipse_duration
    )
    
    # Beginning of life solar array power
    bol_power = calculate_solar_array_power(array_area, years=0)
    
    return PowerResult(
        solar_array_area=array_area,
        battery_capacity=battery_capacity,
        average_power=average_power,
        eclipse_duration=eclipse_duration,
        solar_array_power=bol_power
    )


# =============================================================================
# Thermal Analysis
# =============================================================================

def calculate_thermal_equilibrium(
    solar_absorptivity: float,
    ir_emissivity: float,
    internal_power: float,
    surface_area: float,
    solar_flux: float = SOLAR_CONSTANT,
    albedo: float = 0.3,
    earth_ir: float = 237  # W/m²
) -> float:
    """
    Calculate thermal equilibrium temperature.
    
    Energy balance: Q_in = Q_out
    α×S×A_s + P_int = ε×σ×A×T⁴
    
    Args:
        solar_absorptivity: Solar absorptivity (α)
        ir_emissivity: IR emissivity (ε)
        internal_power: Internal heat dissipation (W)
        surface_area: Total surface area (m²)
        solar_flux: Solar flux (W/m²)
        albedo: Earth albedo
        earth_ir: Earth IR radiation (W/m²)
        
    Returns:
        Equilibrium temperature (K)
    """
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    
    # Projected area for solar input (1/4 of surface for sphere)
    projected_area = surface_area / 4
    
    # Heat inputs
    q_solar = solar_absorptivity * solar_flux * projected_area
    q_albedo = solar_absorptivity * albedo * solar_flux * projected_area * 0.5
    q_earth_ir = ir_emissivity * earth_ir * projected_area * 0.5
    q_internal = internal_power
    
    q_total_in = q_solar + q_albedo + q_earth_ir + q_internal
    
    # Equilibrium temperature
    T = (q_total_in / (ir_emissivity * sigma * surface_area)) ** 0.25
    
    return T


def design_thermal_system(
    internal_power: float,
    surface_area: float,
    hot_limit: float = 323,  # 50°C
    cold_limit: float = 263  # -10°C
) -> ThermalResult:
    """
    Design thermal control system.
    
    Args:
        internal_power: Internal dissipation (W)
        surface_area: Spacecraft surface area (m²)
        hot_limit: Maximum temperature (K)
        cold_limit: Minimum temperature (K)
        
    Returns:
        ThermalResult with thermal design
    """
    # Hot case: full sun, max internal power
    hot_temp = calculate_thermal_equilibrium(
        solar_absorptivity=0.3,
        ir_emissivity=0.8,
        internal_power=internal_power,
        surface_area=surface_area
    )
    
    # Cold case: eclipse, minimum power
    cold_temp = calculate_thermal_equilibrium(
        solar_absorptivity=0.3,
        ir_emissivity=0.8,
        internal_power=internal_power * 0.3,
        surface_area=surface_area,
        solar_flux=0
    )
    
    # Heater power if cold case too cold
    heater_power = 0
    if cold_temp < cold_limit:
        # Need to add heat
        sigma = 5.67e-8
        q_needed = 0.8 * sigma * surface_area * (cold_limit ** 4 - cold_temp ** 4)
        heater_power = max(0, q_needed)
    
    # Radiator sizing if hot case too hot
    radiator_area = 0
    if hot_temp > hot_limit:
        sigma = 5.67e-8
        q_excess = 0.8 * sigma * surface_area * (hot_temp ** 4 - hot_limit ** 4)
        radiator_area = q_excess / (0.9 * sigma * hot_limit ** 4)
    
    return ThermalResult(
        hot_case_temp=hot_temp,
        cold_case_temp=cold_temp,
        heater_power=heater_power,
        radiator_area=radiator_area
    )


# =============================================================================
# Communications
# =============================================================================

def calculate_path_loss(
    frequency: float,
    distance: float
) -> float:
    """
    Calculate free space path loss.
    
    L = 20×log10(4πd/λ)
    
    Args:
        frequency: Frequency (Hz)
        distance: Distance (m)
        
    Returns:
        Path loss (dB)
    """
    c = 3e8  # Speed of light
    wavelength = c / frequency
    
    loss = 20 * math.log10(4 * math.pi * distance / wavelength)
    return loss


def calculate_link_budget(
    transmit_power_w: float,
    transmit_gain_dbi: float,
    frequency_hz: float,
    distance_m: float,
    receive_gain_dbi: float,
    system_noise_temp_k: float = 500,
    bandwidth_hz: float = 1e6,
    required_snr_db: float = 10
) -> LinkBudgetResult:
    """
    Calculate communication link budget.
    
    Args:
        transmit_power_w: Transmit power (W)
        transmit_gain_dbi: Transmit antenna gain (dBi)
        frequency_hz: Carrier frequency (Hz)
        distance_m: Link distance (m)
        receive_gain_dbi: Receive antenna gain (dBi)
        system_noise_temp_k: System noise temperature (K)
        bandwidth_hz: Bandwidth (Hz)
        required_snr_db: Required SNR (dB)
        
    Returns:
        LinkBudgetResult
    """
    k = 1.38e-23  # Boltzmann constant
    
    # EIRP
    transmit_power_dbw = 10 * math.log10(transmit_power_w)
    eirp = transmit_power_dbw + transmit_gain_dbi
    
    # Path loss
    path_loss = calculate_path_loss(frequency_hz, distance_m)
    
    # Received power
    received_power = eirp - path_loss + receive_gain_dbi
    
    # Noise power
    noise_power = 10 * math.log10(k * system_noise_temp_k * bandwidth_hz)
    
    # SNR
    snr = received_power - noise_power
    
    # Margin
    margin = snr - required_snr_db
    
    # Achievable data rate (Shannon)
    data_rate = bandwidth_hz * math.log2(1 + 10 ** (snr / 10))
    
    return LinkBudgetResult(
        eirp=eirp,
        path_loss=path_loss,
        received_power=received_power,
        snr=snr,
        margin=margin,
        data_rate=data_rate
    )


# =============================================================================
# Complete Satellite Design
# =============================================================================

def design_satellite(
    payload_power: float,
    payload_mass: float,
    altitude: float,
    mission_years: float = 5
) -> SatelliteDesignResult:
    """
    Complete satellite design.
    
    Args:
        payload_power: Payload power requirement (W)
        payload_mass: Payload mass (kg)
        altitude: Orbital altitude (m)
        mission_years: Mission duration (years)
        
    Returns:
        SatelliteDesignResult
    """
    # Orbit parameters
    orbit = calculate_orbit_parameters(altitude)
    
    # Total power (payload + housekeeping)
    total_power = payload_power * 1.5  # 50% overhead for bus
    
    # Power system
    power = design_power_system(total_power, altitude, mission_years)
    
    # Mass estimation
    power_mass = power.solar_array_area * 5 + power.battery_capacity / 150  # kg
    structure_mass = payload_mass * 0.3
    thermal_mass = payload_mass * 0.05
    adcs_mass = payload_mass * 0.1
    propulsion_mass = payload_mass * 0.15
    
    total_mass = (payload_mass + power_mass + structure_mass + 
                  thermal_mass + adcs_mass + propulsion_mass)
    
    # Approximate dimensions (cube approximation)
    volume = total_mass / 500  # ~500 kg/m³ average density
    side = volume ** (1/3)
    dimensions = (side, side, side * 1.5)
    
    # Surface area for thermal
    surface_area = 2 * (dimensions[0] * dimensions[1] + 
                       dimensions[1] * dimensions[2] + 
                       dimensions[0] * dimensions[2])
    
    # Thermal design
    thermal = design_thermal_system(total_power * 0.8, surface_area)
    
    return SatelliteDesignResult(
        orbit=orbit,
        power=power,
        thermal=thermal,
        total_mass=total_mass,
        dimensions=dimensions,
        design_life=mission_years
    )


# =============================================================================
# Tool Registry
# =============================================================================

SATELLITE_TOOLS = {
    "calculate_orbital_velocity": {
        "function": calculate_orbital_velocity,
        "description": "Calculate circular orbital velocity",
        "parameters": {"altitude": "Altitude (m)"},
        "returns": "Orbital velocity (m/s)"
    },
    "calculate_orbital_period": {
        "function": calculate_orbital_period,
        "description": "Calculate orbital period",
        "parameters": {"altitude": "Altitude (m)"},
        "returns": "Period (s)"
    },
    "hohmann_transfer_delta_v": {
        "function": hohmann_transfer_delta_v,
        "description": "Calculate Hohmann transfer delta-v",
        "parameters": {
            "altitude_1": "Initial altitude (m)",
            "altitude_2": "Final altitude (m)"
        },
        "returns": "(dv1, dv2) in m/s"
    },
    "design_power_system": {
        "function": design_power_system,
        "description": "Design satellite power system",
        "parameters": {
            "average_power": "Average power (W)",
            "altitude": "Altitude (m)",
            "mission_years": "Mission duration (years)"
        },
        "returns": "PowerResult"
    },
    "calculate_link_budget": {
        "function": calculate_link_budget,
        "description": "Calculate communication link budget",
        "parameters": {
            "transmit_power_w": "TX power (W)",
            "frequency_hz": "Frequency (Hz)",
            "distance_m": "Distance (m)"
        },
        "returns": "LinkBudgetResult"
    },
    "design_satellite": {
        "function": design_satellite,
        "description": "Complete satellite design",
        "parameters": {
            "payload_power": "Payload power (W)",
            "payload_mass": "Payload mass (kg)",
            "altitude": "Altitude (m)"
        },
        "returns": "SatelliteDesignResult"
    },
}


if __name__ == "__main__":
    print("=== Satellite Tools Test ===\n")
    
    # LEO orbit analysis
    print("LEO Orbit at 400 km:")
    orbit = calculate_orbit_parameters(400e3, 51.6)
    print(f"  Velocity: {orbit.velocity:.0f} m/s")
    print(f"  Period: {orbit.period/60:.1f} min")
    print(f"  Type: {orbit.orbit_type}")
    
    # Design small satellite
    print("\nDesigning small satellite:")
    print("  Payload: 50W, 20kg")
    print("  Altitude: 500 km")
    
    sat = design_satellite(
        payload_power=50,
        payload_mass=20,
        altitude=500e3,
        mission_years=5
    )
    
    print(f"\n  Results:")
    print(f"    Total mass: {sat.total_mass:.1f} kg")
    print(f"    Solar array: {sat.power.solar_array_area:.2f} m²")
    print(f"    Battery: {sat.power.battery_capacity:.0f} Wh")
    print(f"    Hot case: {sat.thermal.hot_case_temp - 273:.0f}°C")
    print(f"    Cold case: {sat.thermal.cold_case_temp - 273:.0f}°C")
