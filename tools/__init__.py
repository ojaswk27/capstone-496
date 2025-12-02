"""
Aerospace Calculation Tools
===========================

This package contains specialized calculation tools for different
aerospace vehicle types. Each tool implements real physics formulas
and engineering calculations.

Tool Categories:
- common_tools: Shared utilities (weight, CG, Reynolds number, units)
- drone_tools: Multicopter calculations (thrust, hover, battery)
- fixed_wing_tools: Aircraft calculations (lift, drag, range)
- helicopter_tools: Rotorcraft calculations (disk loading, autorotation)
- rocket_tools: Propulsion calculations (delta-v, staging, burn time)
- satellite_tools: Orbital mechanics (velocity, period, power budget)
- glider_tools: Soaring calculations (glide ratio, sink rate, thermals)
"""

from typing import List, Dict, Any, Callable

# Import tool registries from each module
from .common_tools import COMMON_TOOLS, UnitConverter
from .drone_tools import DRONE_TOOLS
from .fixed_wing_tools import FIXED_WING_TOOLS
from .helicopter_tools import HELICOPTER_TOOLS
from .rocket_tools import ROCKET_TOOLS
from .satellite_tools import SATELLITE_TOOLS
from .glider_tools import GLIDER_TOOLS

# Import key functions for direct access
from .common_tools import (
    isa_atmosphere,
    reynolds_number,
    mach_number,
    dynamic_pressure,
    calculate_cg,
    air_density,
)

from .drone_tools import (
    calculate_hover_thrust,
    calculate_hover_power,
    calculate_flight_time,
    size_drone,
)

from .fixed_wing_tools import (
    calculate_lift,
    calculate_lift_drag,
    calculate_stall_speed,
    calculate_range,
    size_aircraft,
)

from .helicopter_tools import (
    design_rotor,
    calculate_hover_power as helicopter_hover_power,
    design_helicopter,
)

from .rocket_tools import (
    tsiolkovsky_delta_v,
    calculate_delta_v,
    calculate_thrust,
    design_rocket,
)

from .satellite_tools import (
    calculate_orbital_velocity,
    calculate_orbital_period,
    design_satellite,
)

from .glider_tools import (
    calculate_glide_performance,
    calculate_best_glide_speed,
    design_glider,
)


# Unified tool registry mapping vehicle types to their tools
VEHICLE_TOOLS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "drone": DRONE_TOOLS,
    "fixed_wing": FIXED_WING_TOOLS,
    "helicopter": HELICOPTER_TOOLS,
    "rocket": ROCKET_TOOLS,
    "satellite": SATELLITE_TOOLS,
    "glider": GLIDER_TOOLS,
}

# All tools combined
ALL_TOOLS: Dict[str, Dict[str, Any]] = {
    **COMMON_TOOLS,
    **DRONE_TOOLS,
    **FIXED_WING_TOOLS,
    **HELICOPTER_TOOLS,
    **ROCKET_TOOLS,
    **SATELLITE_TOOLS,
    **GLIDER_TOOLS,
}


def get_tools_for_vehicle(vehicle_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all tools applicable for a vehicle type.
    
    Args:
        vehicle_type: Type of vehicle
        
    Returns:
        Dict of tool name to tool info
    """
    tools = {**COMMON_TOOLS}  # Always include common tools
    
    if vehicle_type in VEHICLE_TOOLS:
        tools.update(VEHICLE_TOOLS[vehicle_type])
    
    return tools


def get_tool_function(tool_name: str) -> Callable:
    """
    Get a tool function by name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        The tool function
    """
    if tool_name in ALL_TOOLS:
        return ALL_TOOLS[tool_name]["function"]
    raise ValueError(f"Tool not found: {tool_name}")


def list_tools(vehicle_type: str = None) -> List[str]:
    """
    List available tools.
    
    Args:
        vehicle_type: Optional filter by vehicle type
        
    Returns:
        List of tool names
    """
    if vehicle_type:
        return list(get_tools_for_vehicle(vehicle_type).keys())
    return list(ALL_TOOLS.keys())


__all__ = [
    # Tool registries
    "COMMON_TOOLS",
    "DRONE_TOOLS",
    "FIXED_WING_TOOLS",
    "HELICOPTER_TOOLS",
    "ROCKET_TOOLS",
    "SATELLITE_TOOLS",
    "GLIDER_TOOLS",
    "VEHICLE_TOOLS",
    "ALL_TOOLS",
    # Utility functions
    "get_tools_for_vehicle",
    "get_tool_function",
    "list_tools",
    "UnitConverter",
    # Key functions
    "isa_atmosphere",
    "reynolds_number",
    "calculate_lift",
    "size_drone",
    "size_aircraft",
    "design_helicopter",
    "design_rocket",
    "design_satellite",
    "design_glider",
]
