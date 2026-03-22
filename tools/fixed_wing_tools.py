"""
Fixed-Wing Aircraft Calculation Tools

Specialized calculations for airplane design:
- Lift and drag
- Performance (range, endurance, climb)
- Stability and control
- Weight estimation
- Propulsion matching
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .common_tools import RHO_SL, G, UnitConverter, air_density, isa_atmosphere

# =============================================================================
# Rule-based Aircraft Classification
# =============================================================================


def classify_aircraft(
    payload_kg: float, range_km: float, cruise_speed_kmh: float
) -> Dict[str, Any]:
    """
    Classify aircraft type from requirements using rule-based logic.
    In the new architecture, the Parameter Agent sets aircraft_type correctly,
    so this is a safety net for direct tool calls.
    """
    return _fallback_classification(payload_kg, range_km, cruise_speed_kmh)


def _fallback_classification(
    payload_kg: float, range_km: float, cruise_speed_kmh: float
) -> Dict[str, Any]:
    """Fallback rule-based classification."""
    if payload_kg < 10:
        return {
            "category": "uav_small",
            "is_manned": False,
            "propulsion_type": "electric",
            "design_philosophy": "efficiency",
            "reasoning": "Small payload suggests consumer/hobby UAV",
        }
    elif payload_kg < 100:
        return {
            "category": "uav_tactical",
            "is_manned": False,
            "propulsion_type": "piston" if range_km > 200 else "electric",
            "design_philosophy": "endurance",
            "reasoning": "Medium payload suggests tactical UAV",
        }
    elif payload_kg < 400:
        return {
            "category": "light_sport",
            "is_manned": True,
            "propulsion_type": "piston",
            "design_philosophy": "efficiency",
            "reasoning": "Payload and speed suggest light sport aircraft",
        }
    else:
        return {
            "category": "single_engine_ga",
            "is_manned": True,
            "propulsion_type": "piston",
            "design_philosophy": "performance",
            "reasoning": "Large payload suggests general aviation",
        }


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LiftDragResult:
    """Lift and drag calculation result."""

    lift: float  # N
    drag: float  # N
    cl: float
    cd: float
    ld_ratio: float
    aoa: float  # degrees


@dataclass
class PerformanceResult:
    """Aircraft performance result."""

    stall_speed: float  # m/s
    cruise_speed: float  # m/s
    max_speed: float  # m/s
    climb_rate: float  # m/s
    service_ceiling: float  # m
    range_km: float
    endurance_hours: float


@dataclass
class WingResult:
    """Wing design result."""

    span: float  # m
    area: float  # m²
    chord: float  # m (mean)
    aspect_ratio: float
    wing_loading: float  # N/m²
    taper_ratio: float
    sweep_angle: float  # degrees


@dataclass
class AircraftDesignResult:
    """Complete aircraft design result."""

    wing: WingResult
    performance: PerformanceResult
    weight_breakdown: Dict[str, float]
    total_weight: float  # kg
    power_required: float  # W
    fuel_weight: float  # kg


# =============================================================================
# Aerodynamic Calculations
# =============================================================================


def calculate_lift(
    velocity: float, wing_area: float, cl: float, altitude: float = 0
) -> float:
    """
    Calculate lift force.

    L = 0.5 × ρ × V² × S × C_L

    Args:
        velocity: True airspeed (m/s)
        wing_area: Wing area (m²)
        cl: Lift coefficient
        altitude: Altitude (m)

    Returns:
        Lift force (N)
    """
    rho = air_density(altitude)
    return 0.5 * rho * velocity**2 * wing_area * cl


def calculate_drag(
    velocity: float,
    wing_area: float,
    cd0: float,
    cl: float,
    aspect_ratio: float,
    e: float = 0.8,
    altitude: float = 0,
) -> Tuple[float, float]:
    """
    Calculate drag force.

    C_D = C_D0 + C_L² / (π × e × AR)
    D = 0.5 × ρ × V² × S × C_D

    Args:
        velocity: True airspeed (m/s)
        wing_area: Wing area (m²)
        cd0: Zero-lift drag coefficient
        cl: Lift coefficient
        aspect_ratio: Wing aspect ratio
        e: Oswald efficiency factor (0.7-0.9)
        altitude: Altitude (m)

    Returns:
        (Total drag (N), Induced drag coefficient)
    """
    rho = air_density(altitude)

    # Induced drag coefficient
    cdi = cl**2 / (math.pi * e * aspect_ratio)

    # Total drag coefficient
    cd = cd0 + cdi

    # Drag force
    drag = 0.5 * rho * velocity**2 * wing_area * cd

    return drag, cdi


def calculate_lift_drag(
    weight: float,
    velocity: float,
    wing_area: float,
    aspect_ratio: float,
    cd0: float = 0.025,
    e: float = 0.8,
    altitude: float = 0,
) -> LiftDragResult:
    """
    Calculate lift and drag in steady level flight.

    Args:
        weight: Aircraft weight (kg)
        velocity: True airspeed (m/s)
        wing_area: Wing area (m²)
        aspect_ratio: Wing AR
        cd0: Zero-lift drag coefficient
        e: Oswald efficiency
        altitude: Altitude (m)

    Returns:
        LiftDragResult with all parameters
    """
    rho = air_density(altitude)
    weight_n = weight * G

    # In level flight, L = W
    q = 0.5 * rho * velocity**2
    cl = weight_n / (q * wing_area)

    # Drag
    cdi = cl**2 / (math.pi * e * aspect_ratio)
    cd = cd0 + cdi
    drag = q * wing_area * cd

    # L/D ratio
    ld_ratio = cl / cd

    # Approximate AoA (linear region)
    cl_alpha = (
        2 * math.pi * aspect_ratio / (aspect_ratio + 2)
    )  # Finite wing approximation
    aoa = cl / cl_alpha * 180 / math.pi

    return LiftDragResult(
        lift=weight_n, drag=drag, cl=cl, cd=cd, ld_ratio=ld_ratio, aoa=aoa
    )


def calculate_stall_speed(
    weight: float, wing_area: float, cl_max: float = 1.5, altitude: float = 0
) -> float:
    """
    Calculate stall speed.

    V_stall = √(2W / (ρ × S × C_Lmax))

    Args:
        weight: Weight (kg)
        wing_area: Wing area (m²)
        cl_max: Maximum lift coefficient
        altitude: Altitude (m)

    Returns:
        Stall speed (m/s)
    """
    rho = air_density(altitude)
    weight_n = weight * G

    return math.sqrt(2 * weight_n / (rho * wing_area * cl_max))


def calculate_cl_required(
    weight: float, velocity: float, wing_area: float, altitude: float = 0
) -> float:
    """
    Calculate lift coefficient required for level flight.

    C_L = 2W / (ρ × V² × S)

    Args:
        weight: Weight (kg)
        velocity: True airspeed (m/s)
        wing_area: Wing area (m²)
        altitude: Altitude (m)

    Returns:
        Required lift coefficient
    """
    rho = air_density(altitude)
    weight_n = weight * G

    return 2 * weight_n / (rho * velocity**2 * wing_area)


# =============================================================================
# Performance Calculations
# =============================================================================


def calculate_range(
    fuel_weight: float, sfc: float, ld_ratio: float, weight_initial: float
) -> float:
    """
    Calculate range using Breguet equation.

    R = (V/SFC) × (L/D) × ln(W_i/W_f)

    For propeller aircraft (propeller efficiency included in SFC).

    Args:
        fuel_weight: Fuel weight (kg)
        sfc: Specific fuel consumption (kg/W/hr)
        ld_ratio: Lift-to-drag ratio
        weight_initial: Initial weight (kg)

    Returns:
        Range (km)
    """
    weight_final = weight_initial - fuel_weight

    if weight_final <= 0:
        return 0

    # Breguet range equation for propeller aircraft
    # R = (η_p / SFC) × (L/D) × ln(W_i/W_f)
    # Assuming η_p = 0.8
    eta_p = 0.8
    range_m = (eta_p / sfc) * ld_ratio * math.log(weight_initial / weight_final) * 3600

    return range_m / 1000


def calculate_endurance(fuel_weight: float, power: float, sfc: float) -> float:
    """
    Calculate endurance.

    E = W_fuel / (Power × SFC)

    Args:
        fuel_weight: Fuel weight (kg)
        power: Power consumption (W)
        sfc: Specific fuel consumption (kg/W/hr)

    Returns:
        Endurance (hours)
    """
    return fuel_weight / (power * sfc)


def calculate_rate_of_climb(
    thrust: float, drag: float, weight: float, velocity: float
) -> float:
    """
    Calculate steady rate of climb.

    ROC = V × (T - D) / W

    Args:
        thrust: Available thrust (N)
        drag: Drag force (N)
        weight: Weight (kg)
        velocity: Airspeed (m/s)

    Returns:
        Rate of climb (m/s)
    """
    weight_n = weight * G
    excess_thrust = thrust - drag

    return velocity * excess_thrust / weight_n


def calculate_power_required(
    weight: float,
    velocity: float,
    wing_area: float,
    cd0: float,
    aspect_ratio: float,
    e: float = 0.8,
    altitude: float = 0,
) -> float:
    """
    Calculate power required for level flight.

    P_req = D × V

    Args:
        weight: Weight (kg)
        velocity: True airspeed (m/s)
        wing_area: Wing area (m²)
        cd0: Zero-lift drag coefficient
        aspect_ratio: Wing AR
        e: Oswald efficiency
        altitude: Altitude (m)

    Returns:
        Power required (W)
    """
    ld = calculate_lift_drag(
        weight, velocity, wing_area, aspect_ratio, cd0, e, altitude
    )
    return ld.drag * velocity


def calculate_best_ld_speed(
    weight: float,
    wing_area: float,
    cd0: float,
    aspect_ratio: float,
    e: float = 0.8,
    altitude: float = 0,
) -> Tuple[float, float]:
    """
    Calculate speed for maximum L/D (best range).

    Occurs when C_D0 = C_Di, so C_L = √(π × e × AR × C_D0)

    Args:
        weight: Weight (kg)
        wing_area: Wing area (m²)
        cd0: Zero-lift drag coefficient
        aspect_ratio: Wing AR
        e: Oswald efficiency
        altitude: Altitude (m)

    Returns:
        (Best L/D speed (m/s), Maximum L/D ratio)
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


# =============================================================================
# Wing Design
# =============================================================================


def design_wing(
    weight: float,
    cruise_speed: float,
    stall_speed: float,
    aspect_ratio: float = 8.0,
    taper_ratio: float = 0.5,
    sweep_angle: float = 0,
    altitude: float = 0,
) -> WingResult:
    """
    Design wing from requirements.

    Args:
        weight: Design weight (kg)
        cruise_speed: Target cruise speed (m/s)
        stall_speed: Target stall speed (m/s)
        aspect_ratio: Desired AR
        taper_ratio: Tip chord / root chord
        sweep_angle: Quarter-chord sweep (degrees)
        altitude: Design altitude (m)

    Returns:
        WingResult with wing geometry
    """
    rho = air_density(altitude)
    weight_n = weight * G

    # CL_max from stall speed requirement
    cl_max = 2 * weight_n / (rho * stall_speed**2)

    # Wing area from stall speed (with safety factor)
    wing_area = 2 * weight_n / (rho * stall_speed**2 * cl_max * 0.9)

    # Wing loading
    wing_loading = weight_n / wing_area

    # Span and chord
    span = math.sqrt(wing_area * aspect_ratio)
    mean_chord = wing_area / span

    return WingResult(
        span=span,
        area=wing_area,
        chord=mean_chord,
        aspect_ratio=aspect_ratio,
        wing_loading=wing_loading,
        taper_ratio=taper_ratio,
        sweep_angle=sweep_angle,
    )


# =============================================================================
# Weight Estimation
# =============================================================================


def estimate_empty_weight(
    mtow: float, aircraft_type: str = "single_engine_ga"
) -> float:
    """
    Estimate empty weight from MTOW using historical data.

    W_e = A × W_0^C

    Args:
        mtow: Maximum takeoff weight (kg)
        aircraft_type: Type of aircraft

    Returns:
        Empty weight (kg)
    """
    # Coefficients from Raymer
    coefficients = {
        "homebuilt": (0.99, 0.99),
        "single_engine_ga": (2.36, 0.95),
        "twin_engine_ga": (1.51, 0.96),
        "agricultural": (0.74, 1.00),
        "twin_turboprop": (0.96, 0.95),
        "jet_trainer": (1.59, 0.91),
        "jet_fighter": (2.34, 0.95),
        "transport": (0.92, 0.95),
    }

    A, C = coefficients.get(aircraft_type, (1.0, 0.95))

    # Convert to lb for formula, then back to kg
    mtow_lb = mtow * 2.20462
    empty_lb = A * mtow_lb**C

    return empty_lb / 2.20462


def estimate_fuel_weight(
    range_km: float, cruise_speed: float, ld_ratio: float, sfc: float, weight: float
) -> float:
    """
    Estimate fuel weight required for mission.

    Uses inverse Breguet equation.

    Args:
        range_km: Required range (km)
        cruise_speed: Cruise speed (m/s)
        ld_ratio: L/D ratio at cruise
        sfc: Specific fuel consumption (kg/W/hr)
        weight: Aircraft weight (kg)

    Returns:
        Fuel weight (kg)
    """
    range_m = range_km * 1000
    eta_p = 0.8  # Propeller efficiency

    # Rearranging Breguet:
    # W_i/W_f = exp(R × SFC / (η_p × L/D))
    exponent = range_m * sfc / (eta_p * ld_ratio * 3600)
    weight_ratio = math.exp(exponent)

    # W_fuel = W_i - W_f = W_i × (1 - 1/weight_ratio)
    # But we don't know W_i yet... iterate
    fuel_fraction = 1 - 1 / weight_ratio

    return weight * fuel_fraction / (1 - fuel_fraction)


# =============================================================================
# Complete Aircraft Sizing
# =============================================================================


def size_aircraft(
    payload_kg: float,
    range_km: float,
    cruise_speed_kmh: float,
    aircraft_type: str = "single_engine_ga",
) -> AircraftDesignResult:
    """
    Complete aircraft sizing from requirements.

    Args:
        payload_kg: Payload mass (kg)
        range_km: Required range (km)
        cruise_speed_kmh: Cruise speed (km/h)
        aircraft_type: Type of aircraft

    Returns:
        AircraftDesignResult with complete design
    """
    cruise_speed = cruise_speed_kmh / 3.6  # m/s

    # ========== RULE-BASED CLASSIFICATION ==========
    classification = classify_aircraft(
        payload_kg,
        range_km,
        cruise_speed_kmh,
    )

    category = classification["category"]
    is_manned = classification["is_manned"]
    propulsion = classification["propulsion_type"]

    # ========== CATEGORY-SPECIFIC SIZING PARAMETERS ==========
    sizing_params = {
        "uav_small": {
            "mtow_multiplier": 2.5,
            "empty_weight_fraction": 0.50,
            "cd0": 0.018,
            "aspect_ratio": 14.0,
            "e": 0.88,
            "stall_factor": 0.25,
            "cruise_altitude": 500,
            "sfc": 0.0,
            "crew_weight": 0,
        },
        "uav_tactical": {
            "mtow_multiplier": 3.0,
            "empty_weight_fraction": 0.55,
            "cd0": 0.020,
            "aspect_ratio": 12.0,
            "e": 0.85,
            "stall_factor": 0.30,
            "cruise_altitude": 2000,
            "sfc": 0.0001 if propulsion == "piston" else 0.0,
            "crew_weight": 0,
        },
        "light_sport": {
            "mtow_multiplier": 4.0,
            "empty_weight_fraction": 0.60,
            "cd0": 0.023,
            "aspect_ratio": 10.0,
            "e": 0.82,
            "stall_factor": 0.35,
            "cruise_altitude": 2500,
            "sfc": 0.00009,
            "crew_weight": 80,
        },
        "single_engine_ga": {
            "mtow_multiplier": 4.5,
            "empty_weight_fraction": 0.65,
            "cd0": 0.025,
            "aspect_ratio": 8.0,
            "e": 0.80,
            "stall_factor": 0.40,
            "cruise_altitude": 3000,
            "sfc": 0.00007,
            "crew_weight": 80,
        },
        "twin_engine_ga": {
            "mtow_multiplier": 5.0,
            "empty_weight_fraction": 0.68,
            "cd0": 0.027,
            "aspect_ratio": 7.5,
            "e": 0.78,
            "stall_factor": 0.40,
            "cruise_altitude": 4000,
            "sfc": 0.00007,
            "crew_weight": 160,
        },
        "commuter": {
            "mtow_multiplier": 6.0,
            "empty_weight_fraction": 0.70,
            "cd0": 0.024,
            "aspect_ratio": 9.0,
            "e": 0.82,
            "stall_factor": 0.40,
            "cruise_altitude": 6000,
            "sfc": 0.00005,
            "crew_weight": 200,
        },
        "transport": {
            "mtow_multiplier": 7.0,
            "empty_weight_fraction": 0.72,
            "cd0": 0.022,
            "aspect_ratio": 9.5,
            "e": 0.85,
            "stall_factor": 0.40,
            "cruise_altitude": 10000,
            "sfc": 0.00004,
            "crew_weight": 400,
        },
    }

    params = sizing_params.get(category, sizing_params["single_engine_ga"])

    # ========== WEIGHT ITERATION ==========
    endurance_hours = range_km / cruise_speed_kmh
    mtow = payload_kg * params["mtow_multiplier"]

    # For electric UAVs, adjust initial guess based on endurance
    if propulsion == "electric" and endurance_hours > 1:
        mtow = payload_kg * (2.5 + endurance_hours * 0.5)

    # Iterate to converge on weight
    for iteration in range(15):
        empty_weight = mtow * params["empty_weight_fraction"]

        # Stall speed target
        stall_speed = cruise_speed * params["stall_factor"]

        # Design wing
        wing = design_wing(mtow, cruise_speed, stall_speed, params["aspect_ratio"])

        # L/D at cruise
        ld = calculate_lift_drag(
            mtow,
            cruise_speed,
            wing.area,
            params["aspect_ratio"],
            params["cd0"],
            params["e"],
            params["cruise_altitude"],
        )

        # Fuel/Battery estimate
        if propulsion == "electric":
            # Battery weight for electric aircraft
            # Energy density ~200 Wh/kg, discharge to 80%
            power_required = calculate_power_required(
                mtow,
                cruise_speed,
                wing.area,
                params["cd0"],
                params["aspect_ratio"],
                params["e"],
                params["cruise_altitude"],
            )
            energy_required_wh = (
                power_required * endurance_hours / 0.85
            )  # Motor efficiency
            battery_energy_density = 200  # Wh/kg
            fuel_weight = energy_required_wh / battery_energy_density / 0.8  # 80% DOD
        else:
            # Fuel weight for ICE aircraft
            fuel_weight = estimate_fuel_weight(
                range_km, cruise_speed, ld.ld_ratio, params["sfc"], mtow
            )

        # Check weight closure
        new_mtow = empty_weight + payload_kg + fuel_weight + params["crew_weight"]

        if abs(new_mtow - mtow) < 0.5:
            break

        # Relaxation factor for convergence
        alpha = 0.3 if iteration < 5 else 0.5
        mtow = (1 - alpha) * mtow + alpha * new_mtow

    # ========== FINAL PERFORMANCE CALCULATIONS ==========
    power_required = calculate_power_required(
        mtow,
        cruise_speed,
        wing.area,
        params["cd0"],
        params["aspect_ratio"],
        params["e"],
        params["cruise_altitude"],
    )

    v_stall = calculate_stall_speed(mtow, wing.area, 1.5, 0)
    _, ld_max = calculate_best_ld_speed(
        mtow, wing.area, params["cd0"], params["aspect_ratio"], params["e"]
    )

    performance = PerformanceResult(
        stall_speed=v_stall,
        cruise_speed=cruise_speed,
        max_speed=cruise_speed * 1.3,
        climb_rate=3.0 + (power_required / mtow / 100),  # Rough estimate
        service_ceiling=params["cruise_altitude"] * 2,
        range_km=range_km,
        endurance_hours=endurance_hours,
    )

    weight_breakdown = {
        "empty": empty_weight,
        "fuel" if propulsion != "electric" else "battery": fuel_weight,
        "payload": payload_kg,
        "crew": params["crew_weight"],
    }

    return AircraftDesignResult(
        wing=wing,
        performance=performance,
        weight_breakdown=weight_breakdown,
        total_weight=mtow,
        power_required=power_required,
        fuel_weight=fuel_weight,
    )


# =============================================================================
# Tool Registry for LangGraph
# =============================================================================

FIXED_WING_TOOLS = {
    "size_aircraft": {
        "function": size_aircraft,
        "description": "Complete aircraft sizing from requirements",
        "parameters": {
            "payload_kg": "Payload (kg)",
            "range_km": "Range (km)",
            "cruise_speed_kmh": "Cruise speed (km/h)",
            "aircraft_type": "Aircraft type",
        },
        "returns": "AircraftDesignResult",
    },
    "calculate_lift_drag": {
        "function": calculate_lift_drag,
        "description": "Calculate lift and drag in level flight",
        "parameters": {
            "weight": "Weight (kg)",
            "velocity": "Speed (m/s)",
            "wing_area": "Wing area (m²)",
            "aspect_ratio": "Wing AR",
            "cd0": "Zero-lift drag coefficient",
            "e": "Oswald efficiency",
            "altitude": "Altitude (m)",
        },
        "returns": "LiftDragResult",
    },
    "calculate_stall_speed": {
        "function": calculate_stall_speed,
        "description": "Calculate stall speed",
        "parameters": {
            "weight": "Weight (kg)",
            "wing_area": "Wing area (m²)",
            "cl_max": "Max lift coefficient",
            "altitude": "Altitude (m)",
        },
        "returns": "Stall speed (m/s)",
    },
    "design_wing": {
        "function": design_wing,
        "description": "Design wing from requirements",
        "parameters": {
            "weight": "Weight (kg)",
            "cruise_speed": "Cruise speed (m/s)",
            "stall_speed": "Stall speed (m/s)",
            "aspect_ratio": "Aspect ratio",
        },
        "returns": "WingResult",
    },
    "calculate_power_required": {
        "function": calculate_power_required,
        "description": "Calculate power required for level flight",
        "parameters": {
            "weight": "Weight (kg)",
            "velocity": "Speed (m/s)",
            "wing_area": "Wing area (m²)",
            "cd0": "Zero-lift drag",
            "aspect_ratio": "Wing AR",
            "e": "Oswald efficiency",
            "altitude": "Altitude (m)",
        },
        "returns": "Power (W)",
    },
}
