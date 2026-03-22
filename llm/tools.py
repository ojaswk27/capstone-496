"""
Tool schema generation for Ollama function-calling.
Converts existing calculation tool functions into Ollama-compatible schemas
using function signature introspection.
"""
import inspect
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from tools import (
    size_drone, calculate_hover_thrust, calculate_flight_time,
    size_aircraft, calculate_lift, calculate_stall_speed,
    design_helicopter,
    design_rocket, tsiolkovsky_delta_v,
    design_satellite, calculate_orbital_velocity, calculate_orbital_period,
    design_glider, calculate_glide_performance, calculate_best_glide_speed,
)

# Python type -> JSON Schema type
TYPE_MAP = {
    float: "number",
    int: "integer",
    str: "string",
    bool: "boolean",
}

# Tool registry: vehicle_type -> list of (function, description)
VEHICLE_TOOL_REGISTRY: Dict[str, List[tuple]] = {
    "drone": [
        (size_drone, "Complete drone sizing from payload and flight time requirements"),
        (calculate_hover_thrust, "Calculate thrust required per motor for hover"),
        (calculate_flight_time, "Calculate estimated flight time from battery specs"),
    ],
    "fixed_wing": [
        (size_aircraft, "Complete aircraft sizing from payload, range, and speed"),
        (calculate_lift, "Calculate lift force at given speed, wing area, and lift coefficient"),
        (calculate_stall_speed, "Calculate stall speed for given weight and wing"),
    ],
    "helicopter": [
        (design_helicopter, "Complete helicopter design from payload, range, and speed"),
    ],
    "rocket": [
        (design_rocket, "Complete rocket design for target altitude"),
        (tsiolkovsky_delta_v, "Calculate delta-v using the Tsiolkovsky rocket equation"),
    ],
    "satellite": [
        (design_satellite, "Complete satellite design. Takes payload_power (electrical power in Watts), payload_mass (kg), altitude (meters), mission_years"),
        (calculate_orbital_velocity, "Calculate circular orbital velocity at altitude (meters)"),
        (calculate_orbital_period, "Calculate orbital period at altitude (meters)"),
    ],
    "glider": [
        (design_glider, "Design glider for target glide ratio and class"),
        (calculate_glide_performance, "Calculate glide performance at given conditions"),
        (calculate_best_glide_speed, "Calculate speed for best lift-to-drag ratio"),
    ],
}


def generate_tool_schema(func: Callable, description: str) -> Dict[str, Any]:
    """Generate an Ollama-compatible tool schema from a Python function."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        ann = hints.get(name, str)
        origin = getattr(ann, "__origin__", None)
        if origin is not None:
            args = getattr(ann, "__args__", ())
            ann = args[0] if args else str

        json_type = TYPE_MAP.get(ann, "string")
        properties[name] = {"type": json_type, "description": name.replace("_", " ")}

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def get_tools_for_vehicle_type(vehicle_type: str) -> List[Dict[str, Any]]:
    """Get Ollama tool schemas for a specific vehicle type."""
    entries = VEHICLE_TOOL_REGISTRY.get(vehicle_type, [])
    return [generate_tool_schema(func, desc) for func, desc in entries]


def get_tool_function(name: str) -> Optional[Callable]:
    """Look up a tool function by name across all vehicle types."""
    for entries in VEHICLE_TOOL_REGISTRY.values():
        for func, _ in entries:
            if func.__name__ == name:
                return func
    return None


def validate_tool_args(
    func_name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate and coerce tool call arguments against the function signature."""
    func = get_tool_function(func_name)
    if func is None:
        raise ValueError(f"Unknown tool: {func_name}")

    sig = inspect.signature(func)
    hints = get_type_hints(func)
    cleaned = {}

    for name, param in sig.parameters.items():
        if name in arguments:
            value = arguments[name]
            expected_type = hints.get(name, str)

            origin = getattr(expected_type, "__origin__", None)
            if origin is not None:
                args = getattr(expected_type, "__args__", ())
                expected_type = args[0] if args else str

            try:
                if expected_type == float and not isinstance(value, float):
                    value = float(value)
                elif expected_type == int and not isinstance(value, int):
                    value = int(float(value))
                elif expected_type == bool and not isinstance(value, bool):
                    value = str(value).lower() in ("true", "1", "yes")
                elif expected_type == str and not isinstance(value, str):
                    value = str(value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Parameter '{name}' expected {expected_type.__name__}, "
                    f"got {type(value).__name__}: {value}"
                ) from e

            cleaned[name] = value
        elif param.default is not inspect.Parameter.empty:
            pass
        else:
            raise ValueError(f"Missing required parameter: {name}")

    return cleaned
