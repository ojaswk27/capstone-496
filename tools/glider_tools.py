"""
Glider Calculation Tools

Specialized calculations for glider/sailplane design:
- Glide performance
- Sink rate and polar
- Thermal soaring
- Speed-to-fly
- Weight and balance
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .common_tools import G, RHO_SL, air_density


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GlideResult:
    """Glide performance result."""
    glide_ratio: float
    sink_rate: float  # m/s
    airspeed: float  # m/s
    cl: float
    cd: float


@dataclass
class PolarPoint:
    """Point on the glide polar."""
    airspeed: float  # m/s
    sink_rate: float  # m/s
    glide_ratio: float
    cl: float


@dataclass
class ThermalResult:
    """Thermal soaring analysis result."""
    min_sink_speed: float  # m/s
    min_sink_rate: float  # m/s
    thermal_strength_required: float  # m/s
    climb_rate: float  # m/s
    bank_angle: float  # degrees
    turn_radius: float  # m


@dataclass
class SpeedToFlyResult:
    """Speed-to-fly calculation result."""
    optimal_speed: float  # m/s
    expected_sink: float  # m/s
    achieved_ld: float
    mc_setting: float  # m/s


@dataclass
class GliderDesignResult:
    """Complete glider design result."""
    wing_span: float  # m
    wing_area: float  # m²
    aspect_ratio: float
    empty_weight: float  # kg
    max_weight: float  # kg
    best_glide_ratio: float
    min_sink_rate: float  # m/s
    stall_speed: float  # m/s
    max_speed: float  # m/s


# =============================================================================
# Glide Performance
# =============================================================================

def calculate_glide_ratio(
    lift: float,
    drag: float
) -> float:
    """
    Calculate glide ratio (L/D).
    
    L/D = Lift / Drag = C_L / C_D
    
    Args:
        lift: Lift force (N) or lift coefficient
        drag: Drag force (N) or drag coefficient
        
    Returns:
        Glide ratio
    """
    if drag == 0:
        return float('inf')
    return lift / drag


def calculate_sink_rate(
    weight: float,
    wing_area: float,
    cd: float,
    cl: float,
    altitude: float = 0
) -> float:
    """
    Calculate sink rate.
    
    V_sink = V × sin(γ) ≈ V × (D/W) = V × (C_D/C_L)
    
    Args:
        weight: Weight (kg)
        wing_area: Wing area (m²)
        cd: Drag coefficient
        cl: Lift coefficient
        altitude: Altitude (m)
        
    Returns:
        Sink rate (m/s, positive downward)
    """
    if cl == 0:
        return float('inf')
    
    rho = air_density(altitude)
    weight_n = weight * G
    
    # Airspeed for this CL
    velocity = math.sqrt(2 * weight_n / (rho * wing_area * cl))
    
    # Sink rate
    sink = velocity * cd / cl
    
    return sink


def calculate_glide_performance(
    weight: float,
    wing_area: float,
    aspect_ratio: float,
    cd0: float = 0.01,
    e: float = 0.9,
    velocity: float = None,
    altitude: float = 0
) -> GlideResult:
    """
    Calculate glide performance at given conditions.
    
    Args:
        weight: Weight (kg)
        wing_area: Wing area (m²)
        aspect_ratio: Wing aspect ratio
        cd0: Zero-lift drag coefficient
        e: Oswald efficiency factor
        velocity: Airspeed (m/s), or None for best L/D
        altitude: Altitude (m)
        
    Returns:
        GlideResult with performance data
    """
    rho = air_density(altitude)
    weight_n = weight * G
    
    # If no velocity specified, calculate best L/D speed
    if velocity is None:
        # Best L/D when CD0 = CDi
        cl_best = math.sqrt(math.pi * e * aspect_ratio * cd0)
        velocity = math.sqrt(2 * weight_n / (rho * wing_area * cl_best))
    
    # CL at this velocity
    q = 0.5 * rho * velocity ** 2
    cl = weight_n / (q * wing_area)
    
    # Induced drag
    cdi = cl ** 2 / (math.pi * e * aspect_ratio)
    
    # Total drag coefficient
    cd = cd0 + cdi
    
    # L/D and sink rate
    ld = cl / cd
    sink = velocity * cd / cl
    
    return GlideResult(
        glide_ratio=ld,
        sink_rate=sink,
        airspeed=velocity,
        cl=cl,
        cd=cd
    )


def calculate_best_glide_speed(
    weight: float,
    wing_area: float,
    aspect_ratio: float,
    cd0: float = 0.01,
    e: float = 0.9,
    altitude: float = 0
) -> Tuple[float, float]:
    """
    Calculate speed for best glide ratio.
    
    At best L/D: C_L = √(π × e × AR × C_D0)
    
    Args:
        weight: Weight (kg)
        wing_area: Wing area (m²)
        aspect_ratio: Wing AR
        cd0: Zero-lift drag coefficient
        e: Oswald efficiency
        altitude: Altitude (m)
        
    Returns:
        (Best glide speed (m/s), Maximum L/D)
    """
    rho = air_density(altitude)
    weight_n = weight * G
    
    # CL for best L/D
    cl_opt = math.sqrt(math.pi * e * aspect_ratio * cd0)
    
    # Speed for best L/D
    v_best = math.sqrt(2 * weight_n / (rho * wing_area * cl_opt))
    
    # Maximum L/D
    cd_opt = 2 * cd0
    ld_max = cl_opt / cd_opt
    
    return v_best, ld_max


def calculate_min_sink_speed(
    weight: float,
    wing_area: float,
    aspect_ratio: float,
    cd0: float = 0.01,
    e: float = 0.9,
    altitude: float = 0
) -> Tuple[float, float]:
    """
    Calculate speed for minimum sink rate.
    
    At min sink: C_L = √(3 × π × e × AR × C_D0)
    
    Args:
        weight: Weight (kg)
        wing_area: Wing area (m²)
        aspect_ratio: Wing AR
        cd0: Zero-lift drag coefficient
        e: Oswald efficiency
        altitude: Altitude (m)
        
    Returns:
        (Min sink speed (m/s), Minimum sink rate (m/s))
    """
    rho = air_density(altitude)
    weight_n = weight * G
    
    # CL for minimum sink
    cl_ms = math.sqrt(3 * math.pi * e * aspect_ratio * cd0)
    
    # Speed for min sink
    v_ms = math.sqrt(2 * weight_n / (rho * wing_area * cl_ms))
    
    # CD at min sink
    cd_ms = cd0 + cl_ms ** 2 / (math.pi * e * aspect_ratio)
    
    # Minimum sink rate
    sink_min = v_ms * cd_ms / cl_ms
    
    return v_ms, sink_min


# =============================================================================
# Glide Polar
# =============================================================================

def generate_polar(
    weight: float,
    wing_area: float,
    aspect_ratio: float,
    cd0: float = 0.01,
    e: float = 0.9,
    v_min: float = 15,
    v_max: float = 60,
    num_points: int = 20,
    altitude: float = 0
) -> List[PolarPoint]:
    """
    Generate glide polar curve.
    
    Args:
        weight: Weight (kg)
        wing_area: Wing area (m²)
        aspect_ratio: Wing AR
        cd0: Zero-lift drag coefficient
        e: Oswald efficiency
        v_min: Minimum speed (m/s)
        v_max: Maximum speed (m/s)
        num_points: Number of points
        altitude: Altitude (m)
        
    Returns:
        List of PolarPoint
    """
    polar = []
    
    for i in range(num_points):
        v = v_min + (v_max - v_min) * i / (num_points - 1)
        
        result = calculate_glide_performance(
            weight, wing_area, aspect_ratio, cd0, e, v, altitude
        )
        
        polar.append(PolarPoint(
            airspeed=v,
            sink_rate=result.sink_rate,
            glide_ratio=result.glide_ratio,
            cl=result.cl
        ))
    
    return polar


def interpolate_polar(
    polar: List[PolarPoint],
    airspeed: float
) -> PolarPoint:
    """
    Interpolate polar to get sink rate at given airspeed.
    
    Args:
        polar: List of PolarPoint
        airspeed: Target airspeed (m/s)
        
    Returns:
        Interpolated PolarPoint
    """
    # Find bracketing points
    for i in range(len(polar) - 1):
        if polar[i].airspeed <= airspeed <= polar[i+1].airspeed:
            # Linear interpolation
            t = (airspeed - polar[i].airspeed) / (polar[i+1].airspeed - polar[i].airspeed)
            
            sink = polar[i].sink_rate + t * (polar[i+1].sink_rate - polar[i].sink_rate)
            ld = polar[i].glide_ratio + t * (polar[i+1].glide_ratio - polar[i].glide_ratio)
            cl = polar[i].cl + t * (polar[i+1].cl - polar[i].cl)
            
            return PolarPoint(airspeed, sink, ld, cl)
    
    # Outside range, return nearest
    if airspeed < polar[0].airspeed:
        return polar[0]
    return polar[-1]


# =============================================================================
# Thermal Soaring
# =============================================================================

def calculate_thermal_climb(
    weight: float,
    wing_area: float,
    aspect_ratio: float,
    thermal_strength: float,
    bank_angle: float = 30,
    cd0: float = 0.01,
    e: float = 0.9,
    altitude: float = 0
) -> ThermalResult:
    """
    Calculate climb rate in a thermal.
    
    Args:
        weight: Weight (kg)
        wing_area: Wing area (m²)
        aspect_ratio: Wing AR
        thermal_strength: Thermal updraft strength (m/s)
        bank_angle: Bank angle (degrees)
        cd0: Zero-lift drag
        e: Oswald efficiency
        altitude: Altitude (m)
        
    Returns:
        ThermalResult with thermal performance
    """
    # Min sink speed and rate
    v_ms, sink_min = calculate_min_sink_speed(
        weight, wing_area, aspect_ratio, cd0, e, altitude
    )
    
    # In a turn, sink rate increases
    # Load factor in turn: n = 1/cos(bank)
    bank_rad = math.radians(bank_angle)
    load_factor = 1 / math.cos(bank_rad)
    
    # Sink rate in turn (approximately)
    sink_turn = sink_min * load_factor ** 1.5
    
    # Turn radius
    turn_radius = v_ms ** 2 / (G * math.tan(bank_rad))
    
    # Net climb rate
    climb_rate = thermal_strength - sink_turn
    
    # Minimum thermal strength to stay aloft
    min_thermal = sink_turn
    
    return ThermalResult(
        min_sink_speed=v_ms,
        min_sink_rate=sink_min,
        thermal_strength_required=min_thermal,
        climb_rate=climb_rate,
        bank_angle=bank_angle,
        turn_radius=turn_radius
    )


def calculate_dolphin_flight(
    weight: float,
    wing_area: float,
    aspect_ratio: float,
    lift_strength: float,
    sink_strength: float,
    cd0: float = 0.01,
    e: float = 0.9
) -> float:
    """
    Calculate average climb in dolphin/ridge soaring.
    
    Args:
        weight: Weight (kg)
        wing_area: Wing area (m²)
        aspect_ratio: Wing AR
        lift_strength: Lift band strength (m/s)
        sink_strength: Sink band strength (m/s)
        cd0: Zero-lift drag
        e: Oswald efficiency
        
    Returns:
        Average vertical speed (m/s)
    """
    # Fly slow in lift, fast in sink
    v_slow, _ = calculate_min_sink_speed(weight, wing_area, aspect_ratio, cd0, e)
    v_best, _ = calculate_best_glide_speed(weight, wing_area, aspect_ratio, cd0, e)
    v_fast = v_best * 1.3
    
    # Get sink rates
    slow_result = calculate_glide_performance(
        weight, wing_area, aspect_ratio, cd0, e, v_slow
    )
    fast_result = calculate_glide_performance(
        weight, wing_area, aspect_ratio, cd0, e, v_fast
    )
    
    # Assuming equal time in each air mass
    net_in_lift = lift_strength - slow_result.sink_rate
    net_in_sink = sink_strength - fast_result.sink_rate
    
    # Average (simplified)
    return (net_in_lift + net_in_sink) / 2


# =============================================================================
# Speed-to-Fly
# =============================================================================

def calculate_speed_to_fly(
    weight: float,
    wing_area: float,
    aspect_ratio: float,
    mc_setting: float,
    headwind: float = 0,
    cd0: float = 0.01,
    e: float = 0.9,
    altitude: float = 0
) -> SpeedToFlyResult:
    """
    Calculate optimal speed-to-fly (MacCready theory).
    
    The optimal speed maximizes cross-country speed given
    expected climb rate in next thermal.
    
    Args:
        weight: Weight (kg)
        wing_area: Wing area (m²)
        aspect_ratio: Wing AR
        mc_setting: MacCready setting / expected climb (m/s)
        headwind: Headwind component (m/s, positive = headwind)
        cd0: Zero-lift drag
        e: Oswald efficiency
        altitude: Altitude (m)
        
    Returns:
        SpeedToFlyResult with optimal speed
    """
    # Best glide speed as baseline
    v_best, ld_max = calculate_best_glide_speed(
        weight, wing_area, aspect_ratio, cd0, e, altitude
    )
    
    # Min sink speed
    v_ms, sink_min = calculate_min_sink_speed(
        weight, wing_area, aspect_ratio, cd0, e, altitude
    )
    
    # Speed-to-fly increases with MC setting
    # Simplified: V_stf ≈ V_best × √(1 + MC/sink_min)
    if mc_setting > 0:
        v_stf = v_best * math.sqrt(1 + mc_setting / sink_min)
    else:
        v_stf = v_ms  # Zero MC = fly at min sink
    
    # Adjust for headwind
    v_stf += headwind * 0.5  # Fly faster into headwind
    
    # Get sink at this speed
    result = calculate_glide_performance(
        weight, wing_area, aspect_ratio, cd0, e, v_stf, altitude
    )
    
    # Achieved L/D considering headwind
    groundspeed = v_stf - headwind
    if groundspeed > 0:
        achieved_ld = groundspeed / result.sink_rate
    else:
        achieved_ld = 0
    
    return SpeedToFlyResult(
        optimal_speed=v_stf,
        expected_sink=result.sink_rate,
        achieved_ld=achieved_ld,
        mc_setting=mc_setting
    )


# =============================================================================
# Glider Design
# =============================================================================

def design_glider(
    pilot_weight: float = 80,
    target_glide_ratio: float = 40,
    glider_class: str = "standard"
) -> GliderDesignResult:
    """
    Design glider for target performance.
    
    Args:
        pilot_weight: Pilot weight (kg)
        target_glide_ratio: Target L/D
        glider_class: "club", "standard", "15m", "18m", "open"
        
    Returns:
        GliderDesignResult
    """
    # Class constraints
    class_specs = {
        "club": {"span_max": 15, "ar": 18, "cd0": 0.012, "e": 0.85},
        "standard": {"span_max": 15, "ar": 22, "cd0": 0.010, "e": 0.88},
        "15m": {"span_max": 15, "ar": 24, "cd0": 0.009, "e": 0.90},
        "18m": {"span_max": 18, "ar": 28, "cd0": 0.008, "e": 0.91},
        "open": {"span_max": 30, "ar": 35, "cd0": 0.007, "e": 0.92},
    }
    
    specs = class_specs.get(glider_class, class_specs["standard"])
    
    # Determine AR needed for target L/D
    # L/D_max = 0.5 × √(π × e × AR / CD0)
    # AR = (2 × L/D_max)² × CD0 / (π × e)
    ar_needed = (2 * target_glide_ratio) ** 2 * specs["cd0"] / (math.pi * specs["e"])
    ar = min(ar_needed, specs["ar"])
    
    # Empty weight based on class
    empty_weights = {
        "club": 280, "standard": 260, "15m": 280, "18m": 350, "open": 450
    }
    empty_weight = empty_weights.get(glider_class, 280)
    
    # Total weight
    max_weight = empty_weight + pilot_weight + 20  # 20kg for ballast/baggage
    
    # Wing sizing
    # Wing loading typically 30-40 kg/m² for gliders
    wing_loading = 35  # kg/m²
    wing_area = max_weight / wing_loading
    
    # Span from AR
    span = math.sqrt(ar * wing_area)
    span = min(span, specs["span_max"])
    
    # Recalculate area and AR
    wing_area = span ** 2 / ar
    
    # Performance calculations
    v_best, ld_max = calculate_best_glide_speed(
        max_weight, wing_area, ar, specs["cd0"], specs["e"]
    )
    
    v_ms, sink_min = calculate_min_sink_speed(
        max_weight, wing_area, ar, specs["cd0"], specs["e"]
    )
    
    # Stall speed (CL_max ≈ 1.3 for glider)
    rho = RHO_SL
    v_stall = math.sqrt(2 * max_weight * G / (rho * wing_area * 1.3))
    
    # Max speed (structural)
    v_max = 70 if glider_class in ["club", "standard"] else 80
    
    return GliderDesignResult(
        wing_span=span,
        wing_area=wing_area,
        aspect_ratio=ar,
        empty_weight=empty_weight,
        max_weight=max_weight,
        best_glide_ratio=ld_max,
        min_sink_rate=sink_min,
        stall_speed=v_stall,
        max_speed=v_max
    )


def calculate_glide_range(
    altitude: float,
    glide_ratio: float,
    headwind: float = 0,
    safety_factor: float = 0.8
) -> float:
    """
    Calculate glide range from altitude.
    
    Range = Altitude × L/D × safety_factor
    
    Adjusted for headwind.
    
    Args:
        altitude: Starting altitude AGL (m)
        glide_ratio: L/D ratio
        headwind: Headwind component (m/s)
        safety_factor: Safety margin (0.7-0.9)
        
    Returns:
        Glide range (km)
    """
    # Basic range
    range_still = altitude * glide_ratio / 1000  # km
    
    # Time to descend (rough estimate)
    # Assuming typical sink rate of 0.7 m/s
    time_s = altitude / 0.7
    
    # Distance lost to headwind
    headwind_loss = headwind * time_s / 1000  # km
    
    return (range_still - headwind_loss) * safety_factor


# =============================================================================
# Tool Registry
# =============================================================================

GLIDER_TOOLS = {
    "calculate_glide_performance": {
        "function": calculate_glide_performance,
        "description": "Calculate glide performance at given conditions",
        "parameters": {
            "weight": "Weight (kg)",
            "wing_area": "Wing area (m²)",
            "aspect_ratio": "Wing AR",
            "velocity": "Airspeed (m/s)"
        },
        "returns": "GlideResult"
    },
    "calculate_best_glide_speed": {
        "function": calculate_best_glide_speed,
        "description": "Calculate speed for best L/D",
        "parameters": {
            "weight": "Weight (kg)",
            "wing_area": "Wing area (m²)",
            "aspect_ratio": "Wing AR"
        },
        "returns": "(speed, L/D)"
    },
    "calculate_min_sink_speed": {
        "function": calculate_min_sink_speed,
        "description": "Calculate speed for minimum sink",
        "parameters": {
            "weight": "Weight (kg)",
            "wing_area": "Wing area (m²)",
            "aspect_ratio": "Wing AR"
        },
        "returns": "(speed, sink_rate)"
    },
    "calculate_thermal_climb": {
        "function": calculate_thermal_climb,
        "description": "Calculate climb rate in thermal",
        "parameters": {
            "weight": "Weight (kg)",
            "wing_area": "Wing area (m²)",
            "thermal_strength": "Thermal strength (m/s)"
        },
        "returns": "ThermalResult"
    },
    "calculate_speed_to_fly": {
        "function": calculate_speed_to_fly,
        "description": "Calculate optimal speed-to-fly (MacCready)",
        "parameters": {
            "weight": "Weight (kg)",
            "wing_area": "Wing area (m²)",
            "mc_setting": "MacCready setting (m/s)"
        },
        "returns": "SpeedToFlyResult"
    },
    "design_glider": {
        "function": design_glider,
        "description": "Design glider for target performance",
        "parameters": {
            "pilot_weight": "Pilot weight (kg)",
            "target_glide_ratio": "Target L/D",
            "glider_class": "Class: club/standard/15m/18m/open"
        },
        "returns": "GliderDesignResult"
    },
    "calculate_glide_range": {
        "function": calculate_glide_range,
        "description": "Calculate glide range from altitude",
        "parameters": {
            "altitude": "Altitude AGL (m)",
            "glide_ratio": "L/D ratio",
            "headwind": "Headwind (m/s)"
        },
        "returns": "Range (km)"
    },
}


if __name__ == "__main__":
    print("=== Glider Tools Test ===\n")
    
    # Design a standard class glider
    print("Designing standard class glider:")
    print("  Pilot: 80 kg")
    print("  Target L/D: 42")
    
    glider = design_glider(
        pilot_weight=80,
        target_glide_ratio=42,
        glider_class="standard"
    )
    
    print(f"\n  Wing:")
    print(f"    Span: {glider.wing_span:.1f} m")
    print(f"    Area: {glider.wing_area:.1f} m²")
    print(f"    AR: {glider.aspect_ratio:.1f}")
    
    print(f"\n  Performance:")
    print(f"    Best L/D: {glider.best_glide_ratio:.0f}")
    print(f"    Min sink: {glider.min_sink_rate:.2f} m/s")
    print(f"    Stall: {glider.stall_speed:.1f} m/s ({glider.stall_speed * 1.944:.0f} kt)")
    
    # Thermal performance
    print("\n  Thermal soaring (2 m/s thermal):")
    thermal = calculate_thermal_climb(
        weight=glider.max_weight,
        wing_area=glider.wing_area,
        aspect_ratio=glider.aspect_ratio,
        thermal_strength=2.0,
        bank_angle=35
    )
    print(f"    Climb rate: {thermal.climb_rate:.2f} m/s")
    print(f"    Turn radius: {thermal.turn_radius:.0f} m")
    
    # Glide range
    print("\n  Glide range from 1500m AGL:")
    range_km = calculate_glide_range(1500, glider.best_glide_ratio)
    print(f"    Range: {range_km:.1f} km")
