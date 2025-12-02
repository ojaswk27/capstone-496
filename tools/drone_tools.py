"""
Drone/Multicopter Calculation Tools

Specialized calculations for multirotor UAV design:
- Thrust and hover calculations
- Motor and propeller sizing
- Battery and endurance
- Power consumption
- Flight dynamics
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .common_tools import G, RHO_SL, air_density


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HoverResult:
    """Results from hover analysis."""
    thrust_per_motor: float  # N
    total_thrust: float  # N
    power_per_motor: float  # W
    total_power: float  # W
    disk_loading: float  # N/m²
    figure_of_merit: float
    induced_velocity: float  # m/s


@dataclass
class MotorResult:
    """Motor selection result."""
    kv: float  # RPM/V
    max_thrust: float  # N
    max_power: float  # W
    efficiency: float
    recommended_prop: str
    weight: float  # kg


@dataclass
class BatteryResult:
    """Battery sizing result."""
    capacity_mah: float
    voltage: float  # V
    cells: int  # S count
    weight: float  # kg
    energy_wh: float
    flight_time_minutes: float
    c_rating_required: float


@dataclass
class DroneDesignResult:
    """Complete drone design result."""
    frame_size: float  # mm diagonal
    num_motors: int
    motor_kv: float
    prop_diameter: float  # inches
    prop_pitch: float  # inches
    battery_cells: int
    battery_capacity: float  # mAh
    total_weight: float  # kg
    max_thrust: float  # N
    thrust_to_weight: float
    hover_time: float  # minutes
    max_speed: float  # m/s


# =============================================================================
# Thrust and Power Calculations
# =============================================================================

def calculate_hover_thrust(
    total_weight: float,
    num_motors: int = 4
) -> float:
    """
    Calculate thrust required per motor for hover.
    
    For hover: Total Thrust = Weight
    
    Args:
        total_weight: Total drone weight (kg)
        num_motors: Number of motors
        
    Returns:
        Thrust per motor (N)
    """
    weight_n = total_weight * G
    return weight_n / num_motors


def calculate_hover_power(
    total_weight: float,
    total_disk_area: float,
    altitude: float = 0,
    efficiency: float = 0.7
) -> HoverResult:
    """
    Calculate hover power using momentum theory.
    
    Ideal Power: P_ideal = T^(3/2) / √(2ρA)
    Actual Power: P_actual = P_ideal / FM
    
    Args:
        total_weight: Total weight (kg)
        total_disk_area: Total propeller disk area (m²)
        altitude: Altitude (m)
        efficiency: Figure of merit (0.5-0.8 typical)
        
    Returns:
        HoverResult with all hover parameters
    """
    rho = air_density(altitude)
    thrust = total_weight * G
    
    # Disk loading
    disk_loading = thrust / total_disk_area
    
    # Induced velocity (momentum theory)
    v_i = math.sqrt(thrust / (2 * rho * total_disk_area))
    
    # Ideal power
    p_ideal = thrust * v_i
    
    # Actual power with figure of merit
    p_actual = p_ideal / efficiency
    
    return HoverResult(
        thrust_per_motor=thrust / 4,  # Assuming 4 motors
        total_thrust=thrust,
        power_per_motor=p_actual / 4,
        total_power=p_actual,
        disk_loading=disk_loading,
        figure_of_merit=efficiency,
        induced_velocity=v_i
    )


def calculate_disk_loading(
    thrust: float,
    prop_diameter: float,
    num_motors: int = 4
) -> float:
    """
    Calculate disk loading.
    
    DL = T / A
    
    Args:
        thrust: Total thrust (N)
        prop_diameter: Propeller diameter (m)
        num_motors: Number of motors
        
    Returns:
        Disk loading (N/m²)
    """
    disk_area = num_motors * math.pi * (prop_diameter / 2) ** 2
    return thrust / disk_area


def calculate_thrust_from_power(
    power: float,
    prop_diameter: float,
    altitude: float = 0
) -> float:
    """
    Estimate thrust from power using momentum theory.
    
    Rearranging P = T^(3/2) / √(2ρA):
    T = (P² × 2ρA)^(1/3)
    
    Args:
        power: Motor power (W)
        prop_diameter: Propeller diameter (m)
        altitude: Altitude (m)
        
    Returns:
        Estimated thrust (N)
    """
    rho = air_density(altitude)
    area = math.pi * (prop_diameter / 2) ** 2
    
    return (power ** 2 * 2 * rho * area) ** (1/3)


# =============================================================================
# Motor Calculations
# =============================================================================

def calculate_motor_rpm(
    kv: float,
    voltage: float,
    load_factor: float = 0.85
) -> float:
    """
    Calculate motor RPM under load.
    
    RPM = KV × V × load_factor
    
    Args:
        kv: Motor KV rating (RPM/V)
        voltage: Battery voltage (V)
        load_factor: RPM reduction under load (0.8-0.9)
        
    Returns:
        Motor RPM under load
    """
    return kv * voltage * load_factor


def calculate_prop_tip_speed(
    rpm: float,
    diameter: float
) -> float:
    """
    Calculate propeller tip speed.
    
    V_tip = π × D × RPM / 60
    
    Args:
        rpm: Motor/prop RPM
        diameter: Propeller diameter (m)
        
    Returns:
        Tip speed (m/s)
    """
    return math.pi * diameter * rpm / 60


def estimate_static_thrust(
    prop_diameter: float,
    prop_pitch: float,
    rpm: float,
    altitude: float = 0
) -> float:
    """
    Estimate static thrust using empirical formula.
    
    Thrust ≈ 1.225 × (D/10)^3.5 × (pitch/D) × (RPM/1000)²
    
    Args:
        prop_diameter: Diameter (inches)
        prop_pitch: Pitch (inches)
        rpm: Motor RPM
        altitude: Altitude for density correction
        
    Returns:
        Estimated thrust (N)
    """
    rho = air_density(altitude)
    rho_ratio = rho / RHO_SL
    
    # Empirical formula (results in grams, convert to N)
    thrust_g = rho_ratio * ((prop_diameter / 10) ** 3.5) * \
               (prop_pitch / prop_diameter) * ((rpm / 1000) ** 2) * 0.15
    
    return thrust_g * G / 1000  # Convert grams to N


def select_motor(
    required_thrust: float,
    prop_diameter_inches: float,
    voltage: float,
    application: str = "general"
) -> MotorResult:
    """
    Recommend motor specifications based on requirements.
    
    Args:
        required_thrust: Required thrust per motor (N)
        prop_diameter_inches: Propeller diameter (inches)
        voltage: Battery voltage (V)
        application: "racing", "photography", "heavy_lift"
        
    Returns:
        MotorResult with motor specifications
    """
    # KV guidelines based on application and prop size
    kv_base = {
        "racing": 2500,
        "freestyle": 2000,
        "photography": 900,
        "heavy_lift": 400,
        "general": 1000
    }
    
    base_kv = kv_base.get(application, 1000)
    
    # Adjust KV for prop size (larger props need lower KV)
    kv = base_kv * (5 / prop_diameter_inches) ** 0.5
    kv = max(300, min(3000, kv))  # Clamp to reasonable range
    
    # Estimate max thrust (roughly 2x hover thrust for good performance)
    max_thrust = required_thrust * 2.5
    
    # Power estimate (using typical efficiency)
    max_power = max_thrust * 10  # Rough W/N ratio
    
    # Weight estimate based on power
    weight = max_power / 5000  # ~5000 W/kg for modern motors
    
    # Efficiency varies with load
    efficiency = 0.85 if application == "photography" else 0.80
    
    prop_pitch = prop_diameter_inches * 0.4  # Typical pitch ratio
    
    return MotorResult(
        kv=round(kv, -1),
        max_thrust=max_thrust,
        max_power=max_power,
        efficiency=efficiency,
        recommended_prop=f"{prop_diameter_inches}x{prop_pitch:.1f}",
        weight=weight
    )


# =============================================================================
# Battery Calculations
# =============================================================================

def calculate_battery_requirements(
    total_power: float,
    flight_time_minutes: float,
    voltage: float,
    discharge_rate: float = 0.8,
    safety_margin: float = 0.2
) -> BatteryResult:
    """
    Calculate battery requirements for target flight time.
    
    Args:
        total_power: Average power consumption (W)
        flight_time_minutes: Desired flight time (min)
        voltage: Nominal battery voltage (V)
        discharge_rate: Usable capacity fraction (0.8 typical for LiPo)
        safety_margin: Additional capacity margin
        
    Returns:
        BatteryResult with battery specifications
    """
    # Energy required
    energy_wh = (total_power * flight_time_minutes / 60) / discharge_rate
    energy_wh *= (1 + safety_margin)
    
    # Capacity in mAh
    capacity_mah = energy_wh * 1000 / voltage
    
    # Number of cells (3.7V nominal per cell)
    cells = round(voltage / 3.7)
    actual_voltage = cells * 3.7
    
    # Battery weight estimate (150-200 Wh/kg for LiPo)
    energy_density = 180  # Wh/kg
    weight = energy_wh / energy_density
    
    # C-rating required
    current = total_power / actual_voltage
    c_rating = current / (capacity_mah / 1000)
    
    return BatteryResult(
        capacity_mah=round(capacity_mah, -2),  # Round to nearest 100
        voltage=actual_voltage,
        cells=cells,
        weight=weight,
        energy_wh=energy_wh,
        flight_time_minutes=flight_time_minutes,
        c_rating_required=c_rating
    )


def calculate_flight_time(
    battery_capacity_mah: float,
    battery_voltage: float,
    hover_power: float,
    usable_capacity: float = 0.8
) -> float:
    """
    Calculate estimated flight time.
    
    Time = (Capacity × Voltage × Usable) / Power
    
    Args:
        battery_capacity_mah: Battery capacity (mAh)
        battery_voltage: Nominal voltage (V)
        hover_power: Power at hover (W)
        usable_capacity: Fraction of capacity usable (0.8 typical)
        
    Returns:
        Flight time in minutes
    """
    energy_wh = battery_capacity_mah * battery_voltage / 1000
    usable_energy = energy_wh * usable_capacity
    
    flight_time_hours = usable_energy / hover_power
    return flight_time_hours * 60


def calculate_current_draw(
    power: float,
    voltage: float
) -> float:
    """
    Calculate current draw.
    
    I = P / V
    
    Args:
        power: Power consumption (W)
        voltage: Battery voltage (V)
        
    Returns:
        Current draw (A)
    """
    return power / voltage


# =============================================================================
# Complete Drone Sizing
# =============================================================================

def size_drone(
    payload_kg: float,
    flight_time_minutes: float,
    num_motors: int = 4,
    application: str = "photography"
) -> DroneDesignResult:
    """
    Complete drone sizing based on requirements.
    
    Args:
        payload_kg: Payload mass (kg)
        flight_time_minutes: Target flight time (min)
        num_motors: Number of motors (4, 6, or 8)
        application: Use case ("racing", "photography", "heavy_lift")
        
    Returns:
        DroneDesignResult with complete design
    """
    # Initial weight estimate (iterate)
    # Payload is typically 20-40% of total weight
    payload_fraction = 0.25 if application == "heavy_lift" else 0.30
    initial_weight = payload_kg / payload_fraction
    
    # Battery cells based on application
    if application == "racing":
        cells = 4 if payload_kg < 0.5 else 6
    elif application == "heavy_lift":
        cells = 6
    else:
        cells = 4 if payload_kg < 1 else 6
    
    voltage = cells * 3.7
    
    # Prop sizing based on weight and motor count
    weight_per_motor = initial_weight / num_motors
    if weight_per_motor < 0.2:
        prop_diameter = 5
    elif weight_per_motor < 0.5:
        prop_diameter = 7
    elif weight_per_motor < 1.0:
        prop_diameter = 10
    elif weight_per_motor < 2.0:
        prop_diameter = 15
    else:
        prop_diameter = 18
    
    prop_pitch = prop_diameter * 0.45
    
    # Frame size (diagonal)
    frame_size = prop_diameter * 25.4 * 1.1 * math.sqrt(2) * (num_motors / 4)
    
    # Motor selection
    thrust_required = initial_weight * G / num_motors
    motor = select_motor(thrust_required, prop_diameter, voltage, application)
    
    # Hover power calculation
    disk_area = num_motors * math.pi * (prop_diameter * 0.0254 / 2) ** 2
    hover = calculate_hover_power(initial_weight, disk_area)
    
    # Battery sizing
    battery = calculate_battery_requirements(
        hover.total_power,
        flight_time_minutes,
        voltage
    )
    
    # Refined weight estimate
    frame_weight = (frame_size / 1000) ** 2 * 0.5  # Rough frame weight
    motor_weight = motor.weight * num_motors
    esc_weight = 0.05 * num_motors
    total_weight = (payload_kg + battery.weight + frame_weight + 
                   motor_weight + esc_weight)
    
    # Max thrust
    max_thrust = motor.max_thrust * num_motors
    
    # Thrust to weight ratio
    t_w = max_thrust / (total_weight * G)
    
    # Actual flight time with refined weight
    actual_hover_power = calculate_hover_power(total_weight, disk_area).total_power
    actual_flight_time = calculate_flight_time(
        battery.capacity_mah, voltage, actual_hover_power
    )
    
    # Max speed estimate (very rough)
    max_speed = 20 * t_w  # m/s
    
    return DroneDesignResult(
        frame_size=frame_size,
        num_motors=num_motors,
        motor_kv=motor.kv,
        prop_diameter=prop_diameter,
        prop_pitch=prop_pitch,
        battery_cells=cells,
        battery_capacity=battery.capacity_mah,
        total_weight=total_weight,
        max_thrust=max_thrust,
        thrust_to_weight=t_w,
        hover_time=actual_flight_time,
        max_speed=max_speed
    )


# =============================================================================
# Tool Registry for LangGraph
# =============================================================================

DRONE_TOOLS = {
    "calculate_hover_thrust": {
        "function": calculate_hover_thrust,
        "description": "Calculate thrust required per motor for hover",
        "parameters": {
            "total_weight": "Total drone weight (kg)",
            "num_motors": "Number of motors"
        },
        "returns": "Thrust per motor (N)"
    },
    "calculate_hover_power": {
        "function": calculate_hover_power,
        "description": "Calculate hover power using momentum theory",
        "parameters": {
            "total_weight": "Total weight (kg)",
            "total_disk_area": "Total propeller disk area (m²)",
            "altitude": "Altitude (m)",
            "efficiency": "Figure of merit"
        },
        "returns": "HoverResult with power and performance data"
    },
    "calculate_flight_time": {
        "function": calculate_flight_time,
        "description": "Calculate estimated flight time from battery specs",
        "parameters": {
            "battery_capacity_mah": "Battery capacity (mAh)",
            "battery_voltage": "Nominal voltage (V)",
            "hover_power": "Power at hover (W)",
            "usable_capacity": "Usable capacity fraction"
        },
        "returns": "Flight time in minutes"
    },
    "calculate_battery_requirements": {
        "function": calculate_battery_requirements,
        "description": "Size battery for target flight time",
        "parameters": {
            "total_power": "Average power (W)",
            "flight_time_minutes": "Target flight time (min)",
            "voltage": "Battery voltage (V)"
        },
        "returns": "BatteryResult with battery specifications"
    },
    "size_drone": {
        "function": size_drone,
        "description": "Complete drone sizing from requirements",
        "parameters": {
            "payload_kg": "Payload mass (kg)",
            "flight_time_minutes": "Target flight time (min)",
            "num_motors": "Number of motors (4, 6, or 8)",
            "application": "Use case"
        },
        "returns": "DroneDesignResult with complete design"
    },
    "estimate_static_thrust": {
        "function": estimate_static_thrust,
        "description": "Estimate propeller static thrust",
        "parameters": {
            "prop_diameter": "Diameter (inches)",
            "prop_pitch": "Pitch (inches)",
            "rpm": "Motor RPM"
        },
        "returns": "Estimated thrust (N)"
    },
    "select_motor": {
        "function": select_motor,
        "description": "Recommend motor based on requirements",
        "parameters": {
            "required_thrust": "Thrust per motor (N)",
            "prop_diameter_inches": "Prop diameter (inches)",
            "voltage": "Battery voltage (V)",
            "application": "Use case"
        },
        "returns": "MotorResult with specifications"
    },
}


if __name__ == "__main__":
    # Test drone tools
    print("=== Drone Tools Test ===\n")
    
    # Design a photography drone
    print("Designing photography drone:")
    print("  Payload: 0.5 kg (camera)")
    print("  Target flight time: 25 minutes")
    print("  4 motors")
    
    design = size_drone(
        payload_kg=0.5,
        flight_time_minutes=25,
        num_motors=4,
        application="photography"
    )
    
    print(f"\n  Results:")
    print(f"    Frame size: {design.frame_size:.0f} mm")
    print(f"    Motors: {design.num_motors}x {design.motor_kv:.0f} KV")
    print(f"    Props: {design.prop_diameter}x{design.prop_pitch:.1f} inches")
    print(f"    Battery: {design.battery_cells}S {design.battery_capacity:.0f} mAh")
    print(f"    Total weight: {design.total_weight:.2f} kg")
    print(f"    T/W ratio: {design.thrust_to_weight:.1f}")
    print(f"    Hover time: {design.hover_time:.1f} min")
    print(f"    Max speed: {design.max_speed:.1f} m/s")
    
    # Test hover calculation
    print("\n\nHover analysis for 2kg drone with 10\" props:")
    disk_area = 4 * math.pi * (0.254 / 2) ** 2
    hover = calculate_hover_power(2.0, disk_area)
    print(f"  Total thrust: {hover.total_thrust:.1f} N")
    print(f"  Total power: {hover.total_power:.1f} W")
    print(f"  Disk loading: {hover.disk_loading:.1f} N/m²")
    print(f"  Induced velocity: {hover.induced_velocity:.2f} m/s")
