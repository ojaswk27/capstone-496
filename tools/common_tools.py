"""
Common Aerospace Calculation Tools

Shared utilities and calculations used across all vehicle types:
- Unit conversions
- Atmospheric properties
- Reynolds number
- Weight and CG calculations
- Material properties
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


# =============================================================================
# Constants
# =============================================================================

# Physical constants
G = 9.80665  # Gravitational acceleration (m/s²)
R_AIR = 287.05  # Specific gas constant for air (J/kg·K)
GAMMA = 1.4  # Ratio of specific heats for air

# Standard atmosphere at sea level
RHO_SL = 1.225  # Air density (kg/m³)
T_SL = 288.15  # Temperature (K)
P_SL = 101325  # Pressure (Pa)

# Earth
MU_EARTH = 3.986004418e14  # Gravitational parameter (m³/s²)
R_EARTH = 6.371e6  # Mean radius (m)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AtmosphereResult:
    """Atmospheric properties at a given altitude."""
    altitude: float  # m
    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m³
    speed_of_sound: float  # m/s
    dynamic_viscosity: float  # Pa·s


@dataclass
class CGResult:
    """Center of gravity calculation result."""
    x_cg: float  # x position
    y_cg: float  # y position (optional)
    z_cg: float  # z position (optional)
    total_weight: float
    static_margin: Optional[float] = None


# =============================================================================
# Atmospheric Functions
# =============================================================================

def isa_atmosphere(altitude: float) -> AtmosphereResult:
    """
    Calculate International Standard Atmosphere (ISA) properties.
    
    Valid for altitudes from 0 to 86 km.
    
    Args:
        altitude: Altitude above sea level (m)
        
    Returns:
        AtmosphereResult with all atmospheric properties
    """
    if altitude < 0:
        altitude = 0
    
    # Troposphere (0 - 11 km)
    if altitude <= 11000:
        T = T_SL - 0.0065 * altitude
        P = P_SL * (T / T_SL) ** 5.2561
    # Lower stratosphere (11 - 20 km)
    elif altitude <= 20000:
        T = 216.65
        P = 22632 * math.exp(-G * (altitude - 11000) / (R_AIR * T))
    # Upper stratosphere (20 - 32 km)
    elif altitude <= 32000:
        T = 216.65 + 0.001 * (altitude - 20000)
        P = 5474.9 * (T / 216.65) ** -34.163
    # Mesosphere (32 - 47 km)
    elif altitude <= 47000:
        T = 228.65 + 0.0028 * (altitude - 32000)
        P = 868.02 * (T / 228.65) ** -12.201
    else:
        T = 270.65
        P = 110.91 * math.exp(-G * (altitude - 47000) / (R_AIR * T))
    
    # Calculate remaining properties
    rho = P / (R_AIR * T)
    a = math.sqrt(GAMMA * R_AIR * T)
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)  # Sutherland's law
    
    return AtmosphereResult(
        altitude=altitude,
        temperature=T,
        pressure=P,
        density=rho,
        speed_of_sound=a,
        dynamic_viscosity=mu
    )


def density_altitude(pressure_altitude: float, temperature: float) -> float:
    """
    Calculate density altitude from pressure altitude and actual temperature.
    
    Args:
        pressure_altitude: Pressure altitude (m)
        temperature: Actual outside air temperature (K)
        
    Returns:
        Density altitude (m)
    """
    # ISA temperature at pressure altitude
    isa = isa_atmosphere(pressure_altitude)
    
    # Temperature deviation
    delta_T = temperature - isa.temperature
    
    # Density altitude approximation
    return pressure_altitude + 120 * delta_T


def air_density(altitude: float, temperature_offset: float = 0) -> float:
    """
    Get air density at altitude with optional temperature offset.
    
    Args:
        altitude: Altitude (m)
        temperature_offset: Deviation from ISA temperature (K)
        
    Returns:
        Air density (kg/m³)
    """
    isa = isa_atmosphere(altitude)
    
    if temperature_offset != 0:
        T_actual = isa.temperature + temperature_offset
        return isa.pressure / (R_AIR * T_actual)
    
    return isa.density


# =============================================================================
# Aerodynamic Functions
# =============================================================================

def reynolds_number(velocity: float, length: float, altitude: float = 0) -> float:
    """
    Calculate Reynolds number.
    
    Re = ρVL/μ
    
    Args:
        velocity: Flow velocity (m/s)
        length: Characteristic length (m)
        altitude: Altitude for air properties (m)
        
    Returns:
        Reynolds number (dimensionless)
    """
    atm = isa_atmosphere(altitude)
    return atm.density * velocity * length / atm.dynamic_viscosity


def mach_number(velocity: float, altitude: float = 0) -> float:
    """
    Calculate Mach number.
    
    M = V/a
    
    Args:
        velocity: True airspeed (m/s)
        altitude: Altitude (m)
        
    Returns:
        Mach number (dimensionless)
    """
    atm = isa_atmosphere(altitude)
    return velocity / atm.speed_of_sound


def dynamic_pressure(velocity: float, altitude: float = 0) -> float:
    """
    Calculate dynamic pressure (q).
    
    q = 0.5 × ρ × V²
    
    Args:
        velocity: True airspeed (m/s)
        altitude: Altitude (m)
        
    Returns:
        Dynamic pressure (Pa)
    """
    rho = air_density(altitude)
    return 0.5 * rho * velocity ** 2


def tas_to_eas(tas: float, altitude: float) -> float:
    """
    Convert True Airspeed to Equivalent Airspeed.
    
    EAS = TAS × √(ρ/ρ₀)
    
    Args:
        tas: True airspeed (m/s)
        altitude: Altitude (m)
        
    Returns:
        Equivalent airspeed (m/s)
    """
    rho = air_density(altitude)
    return tas * math.sqrt(rho / RHO_SL)


def eas_to_tas(eas: float, altitude: float) -> float:
    """
    Convert Equivalent Airspeed to True Airspeed.
    
    TAS = EAS / √(ρ/ρ₀)
    
    Args:
        eas: Equivalent airspeed (m/s)
        altitude: Altitude (m)
        
    Returns:
        True airspeed (m/s)
    """
    rho = air_density(altitude)
    return eas / math.sqrt(rho / RHO_SL)


# =============================================================================
# Weight and CG Functions
# =============================================================================

def calculate_cg(
    components: List[Dict[str, float]],
    reference_point: Tuple[float, float, float] = (0, 0, 0)
) -> CGResult:
    """
    Calculate center of gravity from component weights and positions.
    
    Each component: {"weight": kg, "x": m, "y": m (optional), "z": m (optional)}
    
    Args:
        components: List of component dictionaries
        reference_point: Reference datum (x, y, z) in meters
        
    Returns:
        CGResult with CG location and total weight
    """
    total_weight = 0
    moment_x = 0
    moment_y = 0
    moment_z = 0
    
    for comp in components:
        w = comp.get("weight", 0)
        x = comp.get("x", 0) - reference_point[0]
        y = comp.get("y", 0) - reference_point[1]
        z = comp.get("z", 0) - reference_point[2]
        
        total_weight += w
        moment_x += w * x
        moment_y += w * y
        moment_z += w * z
    
    if total_weight == 0:
        return CGResult(0, 0, 0, 0)
    
    return CGResult(
        x_cg=moment_x / total_weight + reference_point[0],
        y_cg=moment_y / total_weight + reference_point[1],
        z_cg=moment_z / total_weight + reference_point[2],
        total_weight=total_weight
    )


def weight_from_volume(
    volume: float,
    material: str = "aluminum"
) -> float:
    """
    Estimate weight from volume and material.
    
    Args:
        volume: Volume in m³
        material: Material name
        
    Returns:
        Weight in kg
    """
    densities = {
        "aluminum": 2700,
        "steel": 7850,
        "titanium": 4500,
        "carbon_fiber": 1550,
        "fiberglass": 1900,
        "balsa": 160,
        "foam": 40,
        "wood": 600,
    }
    
    rho = densities.get(material.lower(), 2700)
    return volume * rho


# =============================================================================
# Unit Conversions
# =============================================================================

class UnitConverter:
    """Unit conversion utilities for aerospace calculations."""
    
    # Length
    @staticmethod
    def ft_to_m(ft: float) -> float:
        return ft * 0.3048
    
    @staticmethod
    def m_to_ft(m: float) -> float:
        return m / 0.3048
    
    @staticmethod
    def in_to_m(inches: float) -> float:
        return inches * 0.0254
    
    @staticmethod
    def m_to_in(m: float) -> float:
        return m / 0.0254
    
    @staticmethod
    def nm_to_km(nm: float) -> float:
        return nm * 1.852
    
    @staticmethod
    def km_to_nm(km: float) -> float:
        return km / 1.852
    
    # Speed
    @staticmethod
    def kt_to_ms(kt: float) -> float:
        return kt * 0.514444
    
    @staticmethod
    def ms_to_kt(ms: float) -> float:
        return ms / 0.514444
    
    @staticmethod
    def mph_to_ms(mph: float) -> float:
        return mph * 0.44704
    
    @staticmethod
    def ms_to_mph(ms: float) -> float:
        return ms / 0.44704
    
    @staticmethod
    def kmh_to_ms(kmh: float) -> float:
        return kmh / 3.6
    
    @staticmethod
    def ms_to_kmh(ms: float) -> float:
        return ms * 3.6
    
    # Mass
    @staticmethod
    def lb_to_kg(lb: float) -> float:
        return lb * 0.453592
    
    @staticmethod
    def kg_to_lb(kg: float) -> float:
        return kg / 0.453592
    
    @staticmethod
    def oz_to_kg(oz: float) -> float:
        return oz * 0.0283495
    
    @staticmethod
    def kg_to_oz(kg: float) -> float:
        return kg / 0.0283495
    
    # Force
    @staticmethod
    def lbf_to_n(lbf: float) -> float:
        return lbf * 4.44822
    
    @staticmethod
    def n_to_lbf(n: float) -> float:
        return n / 4.44822
    
    # Area
    @staticmethod
    def sqft_to_sqm(sqft: float) -> float:
        return sqft * 0.092903
    
    @staticmethod
    def sqm_to_sqft(sqm: float) -> float:
        return sqm / 0.092903
    
    # Temperature
    @staticmethod
    def c_to_k(c: float) -> float:
        return c + 273.15
    
    @staticmethod
    def k_to_c(k: float) -> float:
        return k - 273.15
    
    @staticmethod
    def f_to_k(f: float) -> float:
        return (f + 459.67) * 5 / 9
    
    @staticmethod
    def k_to_f(k: float) -> float:
        return k * 9 / 5 - 459.67
    
    # Power
    @staticmethod
    def hp_to_w(hp: float) -> float:
        return hp * 745.7
    
    @staticmethod
    def w_to_hp(w: float) -> float:
        return w / 745.7
    
    # Pressure
    @staticmethod
    def psi_to_pa(psi: float) -> float:
        return psi * 6894.76
    
    @staticmethod
    def pa_to_psi(pa: float) -> float:
        return pa / 6894.76
    
    @staticmethod
    def bar_to_pa(bar: float) -> float:
        return bar * 100000
    
    @staticmethod
    def pa_to_bar(pa: float) -> float:
        return pa / 100000


# =============================================================================
# Material Properties
# =============================================================================

MATERIALS = {
    "aluminum_2024_t3": {
        "density": 2780,  # kg/m³
        "yield_strength": 345e6,  # Pa
        "ultimate_strength": 483e6,  # Pa
        "elastic_modulus": 73.1e9,  # Pa
        "poisson_ratio": 0.33,
    },
    "aluminum_6061_t6": {
        "density": 2700,
        "yield_strength": 276e6,
        "ultimate_strength": 310e6,
        "elastic_modulus": 68.9e9,
        "poisson_ratio": 0.33,
    },
    "aluminum_7075_t6": {
        "density": 2810,
        "yield_strength": 503e6,
        "ultimate_strength": 572e6,
        "elastic_modulus": 71.7e9,
        "poisson_ratio": 0.33,
    },
    "steel_4130": {
        "density": 7850,
        "yield_strength": 460e6,
        "ultimate_strength": 560e6,
        "elastic_modulus": 205e9,
        "poisson_ratio": 0.29,
    },
    "titanium_6al4v": {
        "density": 4430,
        "yield_strength": 880e6,
        "ultimate_strength": 950e6,
        "elastic_modulus": 113.8e9,
        "poisson_ratio": 0.342,
    },
    "carbon_fiber_ud": {
        "density": 1550,
        "tensile_strength": 1500e6,
        "elastic_modulus": 135e9,
        "poisson_ratio": 0.30,
    },
    "fiberglass_e": {
        "density": 1900,
        "tensile_strength": 500e6,
        "elastic_modulus": 35e9,
        "poisson_ratio": 0.22,
    },
}


def get_material_properties(material_name: str) -> Dict[str, float]:
    """Get material properties by name."""
    key = material_name.lower().replace(" ", "_").replace("-", "_")
    return MATERIALS.get(key, MATERIALS["aluminum_6061_t6"])


# =============================================================================
# Tool Registry for LangGraph
# =============================================================================

COMMON_TOOLS = {
    "isa_atmosphere": {
        "function": isa_atmosphere,
        "description": "Calculate ISA atmospheric properties at a given altitude",
        "parameters": {"altitude": "Altitude in meters"},
        "returns": "AtmosphereResult with temperature, pressure, density, speed of sound"
    },
    "reynolds_number": {
        "function": reynolds_number,
        "description": "Calculate Reynolds number for aerodynamic analysis",
        "parameters": {
            "velocity": "Flow velocity (m/s)",
            "length": "Characteristic length (m)",
            "altitude": "Altitude (m), default 0"
        },
        "returns": "Reynolds number (dimensionless)"
    },
    "mach_number": {
        "function": mach_number,
        "description": "Calculate Mach number",
        "parameters": {
            "velocity": "True airspeed (m/s)",
            "altitude": "Altitude (m)"
        },
        "returns": "Mach number (dimensionless)"
    },
    "dynamic_pressure": {
        "function": dynamic_pressure,
        "description": "Calculate dynamic pressure q = 0.5*rho*V^2",
        "parameters": {
            "velocity": "True airspeed (m/s)",
            "altitude": "Altitude (m)"
        },
        "returns": "Dynamic pressure (Pa)"
    },
    "calculate_cg": {
        "function": calculate_cg,
        "description": "Calculate center of gravity from component weights and positions",
        "parameters": {
            "components": "List of dicts with weight, x, y, z",
            "reference_point": "Reference datum (x, y, z)"
        },
        "returns": "CGResult with CG location"
    },
    "air_density": {
        "function": air_density,
        "description": "Get air density at altitude",
        "parameters": {
            "altitude": "Altitude (m)",
            "temperature_offset": "Deviation from ISA (K)"
        },
        "returns": "Air density (kg/m³)"
    },
}


if __name__ == "__main__":
    # Test common tools
    print("=== Common Tools Test ===\n")
    
    # Atmosphere test
    print("ISA Atmosphere at 3000m:")
    atm = isa_atmosphere(3000)
    print(f"  Temperature: {atm.temperature:.2f} K ({atm.temperature - 273.15:.2f} °C)")
    print(f"  Pressure: {atm.pressure:.0f} Pa")
    print(f"  Density: {atm.density:.4f} kg/m³")
    print(f"  Speed of Sound: {atm.speed_of_sound:.1f} m/s")
    
    # Reynolds number test
    print("\nReynolds number for 50 m/s, 1m chord:")
    Re = reynolds_number(50, 1.0, 0)
    print(f"  Re = {Re:.2e}")
    
    # Mach number test
    print("\nMach number for 250 m/s at 10000m:")
    M = mach_number(250, 10000)
    print(f"  M = {M:.3f}")
    
    # CG calculation test
    print("\nCG calculation:")
    components = [
        {"weight": 100, "x": 0.5, "y": 0, "z": 0},
        {"weight": 50, "x": 1.5, "y": 0, "z": 0},
        {"weight": 30, "x": 2.5, "y": 0, "z": 0},
    ]
    cg = calculate_cg(components)
    print(f"  Total weight: {cg.total_weight} kg")
    print(f"  CG location: x={cg.x_cg:.3f} m")
    
    # Unit conversion test
    print("\nUnit conversions:")
    print(f"  100 kt = {UnitConverter.kt_to_ms(100):.2f} m/s")
    print(f"  10000 ft = {UnitConverter.ft_to_m(10000):.1f} m")
    print(f"  100 hp = {UnitConverter.hp_to_w(100):.0f} W")
