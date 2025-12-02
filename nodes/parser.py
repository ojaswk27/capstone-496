"""
Requirement Parser Node

Parses user requirements from natural language into structured format.
"""

import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# FIXED: Import from graph.state
sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.state import DesignPhase, DesignState, UserRequirements, VehicleType

# ... rest of the file remains the same


# =============================================================================
# Unit Conversion Tables
# =============================================================================

MASS_CONVERSIONS = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "lb": 0.453592,
    "lbs": 0.453592,
    "pound": 0.453592,
    "pounds": 0.453592,
    "oz": 0.0283495,
    "ounce": 0.0283495,
    "ounces": 0.0283495,
}

DISTANCE_CONVERSIONS = {
    "km": 1.0,
    "kilometer": 1.0,
    "kilometers": 1.0,
    "m": 0.001,
    "meter": 0.001,
    "meters": 0.001,
    "mi": 1.60934,
    "mile": 1.60934,
    "miles": 1.60934,
    "nm": 1.852,
    "nmi": 1.852,
    "ft": 0.0003048,
    "feet": 0.0003048,
    "foot": 0.0003048,
}

ALTITUDE_CONVERSIONS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "ft": 0.3048,
    "feet": 0.3048,
    "foot": 0.3048,
}

SPEED_CONVERSIONS = {
    "km/h": 1.0,
    "kmh": 1.0,
    "kph": 1.0,
    "m/s": 3.6,
    "mph": 1.60934,
    "kt": 1.852,
    "kts": 1.852,
    "knot": 1.852,
    "knots": 1.852,
}

TIME_CONVERSIONS = {
    "h": 1.0,
    "hr": 1.0,
    "hrs": 1.0,
    "hour": 1.0,
    "hours": 1.0,
    "min": 1 / 60,
    "mins": 1 / 60,
    "minute": 1 / 60,
    "minutes": 1 / 60,
    "s": 1 / 3600,
    "sec": 1 / 3600,
    "second": 1 / 3600,
    "seconds": 1 / 3600,
}


# =============================================================================
# Parsing Functions
# =============================================================================


def extract_number_with_unit(
    text: str,
    keywords: List[str],
    conversions: Dict[str, float],
    default_unit: str = None,
) -> Optional[float]:
    """
    Extract a numerical value with unit from text.

    Args:
        text: Input text
        keywords: Keywords that precede or relate to the value
        conversions: Unit to base unit conversion factors
        default_unit: Default unit if none specified

    Returns:
        Value in base units, or None if not found
    """
    text_lower = text.lower()

    # Build pattern for keywords
    keyword_pattern = "|".join(re.escape(kw) for kw in keywords)
    unit_pattern = "|".join(re.escape(u) for u in conversions.keys())

    # Pattern: keyword ... number unit
    patterns = [
        # "payload of 2 kg", "range: 500 km"
        rf"(?:{keyword_pattern})\s*(?:of|:|\s)\s*(\d+(?:\.\d+)?)\s*({unit_pattern})?",
        # "2 kg payload", "500 km range"
        rf"(\d+(?:\.\d+)?)\s*({unit_pattern})?\s*(?:{keyword_pattern})",
        # "carry 2kg", "fly 500km"
        rf"(\d+(?:\.\d+)?)\s*({unit_pattern})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            value = float(match.group(1))
            unit = match.group(2) if len(match.groups()) > 1 else None

            if unit and unit in conversions:
                return value * conversions[unit]
            elif default_unit and default_unit in conversions:
                return value * conversions[default_unit]
            else:
                return value

    return None


def extract_payload(text: str) -> Optional[float]:
    """Extract payload weight in kg."""
    keywords = ["payload", "carry", "carrying", "load", "camera", "package"]
    return extract_number_with_unit(text, keywords, MASS_CONVERSIONS, "kg")


def extract_range(text: str) -> Optional[float]:
    """Extract range in km."""
    keywords = ["range", "distance", "travel", "fly", "cover"]
    return extract_number_with_unit(text, keywords, DISTANCE_CONVERSIONS, "km")


def extract_endurance(text: str) -> Optional[float]:
    """Extract endurance/flight time in hours."""
    keywords = [
        "endurance",
        "flight time",
        "duration",
        "fly for",
        "hover for",
        "airborne",
    ]

    # Special handling for minutes
    text_lower = text.lower()
    min_match = re.search(r"(\d+)\s*min(ute)?s?\s*(of\s+)?(flight|hover)?", text_lower)
    if min_match:
        return float(min_match.group(1)) / 60

    return extract_number_with_unit(text, keywords, TIME_CONVERSIONS, "h")


def extract_speed(text: str) -> Optional[float]:
    """Extract speed in km/h."""
    keywords = ["speed", "velocity", "cruise", "max speed", "travel at"]
    return extract_number_with_unit(text, keywords, SPEED_CONVERSIONS, "km/h")


def extract_altitude(text: str) -> Optional[float]:
    """Extract altitude in meters."""
    keywords = ["altitude", "height", "ceiling", "reach", "fly at", "elevation"]
    return extract_number_with_unit(text, keywords, ALTITUDE_CONVERSIONS, "m")


def extract_passengers(text: str) -> Optional[int]:
    """Extract number of passengers."""
    patterns = [
        r"(\d+)\s*(?:passenger|person|people|seat|pax)",
        r"(?:carry|transport|hold)\s*(\d+)\s*(?:passenger|person|people)",
        r"(\d+)-?seat",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))

    return None


def extract_orbit_altitude(text: str) -> Optional[float]:
    """Extract orbital altitude in km."""
    # LEO/MEO/GEO shortcuts
    text_lower = text.lower()

    if "geo" in text_lower or "geostationary" in text_lower:
        return 35786
    if "leo" in text_lower:
        # Default LEO altitude
        return 400
    if "meo" in text_lower:
        return 20000

    # Explicit altitude
    keywords = ["orbit", "orbital altitude", "altitude"]
    return extract_number_with_unit(text, keywords, {"km": 1.0, "m": 0.001}, "km")


def extract_mission_years(text: str) -> Optional[float]:
    """Extract mission duration in years."""
    patterns = [
        r"(\d+(?:\.\d+)?)\s*year\s*(?:mission|life|duration)?",
        r"(?:mission|life|duration)\s*(?:of)?\s*(\d+(?:\.\d+)?)\s*year",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1))

    return None


def extract_mission_type(text: str) -> Optional[str]:
    """Extract mission type from text."""
    mission_keywords = {
        "surveillance": ["surveillance", "monitoring", "observation", "watch"],
        "photography": ["photo", "camera", "filming", "video", "imaging"],
        "delivery": ["delivery", "transport", "carry", "logistics"],
        "racing": ["racing", "fpv", "speed", "competition"],
        "research": ["research", "scientific", "experiment", "study"],
        "communication": ["communication", "relay", "broadcast", "telecom"],
        "earth_observation": ["earth observation", "remote sensing", "mapping"],
        "passenger": ["passenger", "people", "travel", "transport people"],
        "cargo": ["cargo", "freight", "shipping", "goods"],
        "rescue": ["rescue", "emergency", "medical", "evacuation", "sar"],
        "training": ["training", "trainer", "practice", "learn"],
    }

    text_lower = text.lower()

    for mission_type, keywords in mission_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                return mission_type

    return None


def parse_requirements(text: str, vehicle_type: VehicleType) -> UserRequirements:
    """
    Parse all requirements from user input.

    Args:
        text: User input text
        vehicle_type: Classified vehicle type

    Returns:
        UserRequirements dataclass
    """
    req = UserRequirements(raw_input=text)

    # Common requirements
    req.payload_kg = extract_payload(text)
    req.mission_type = extract_mission_type(text)

    # Vehicle-specific parsing
    if vehicle_type in [VehicleType.DRONE]:
        req.endurance_hours = extract_endurance(text)
        req.speed_kmh = extract_speed(text)
        req.altitude_m = extract_altitude(text)

    elif vehicle_type == VehicleType.FIXED_WING:
        req.range_km = extract_range(text)
        req.speed_kmh = extract_speed(text)
        req.num_passengers = extract_passengers(text)
        req.altitude_m = extract_altitude(text)

    elif vehicle_type == VehicleType.HELICOPTER:
        req.range_km = extract_range(text)
        req.speed_kmh = extract_speed(text)
        req.num_passengers = extract_passengers(text)
        req.endurance_hours = extract_endurance(text)

    elif vehicle_type == VehicleType.ROCKET:
        req.target_altitude_m = extract_altitude(text)
        if not req.target_altitude_m:
            # Check for "reach X km"
            match = re.search(r"reach\s*(\d+)\s*(km|m)", text.lower())
            if match:
                val = float(match.group(1))
                unit = match.group(2)
                req.target_altitude_m = val * (1000 if unit == "km" else 1)

    elif vehicle_type == VehicleType.SATELLITE:
        req.orbit_altitude_km = extract_orbit_altitude(text)
        req.mission_years = extract_mission_years(text)

    elif vehicle_type == VehicleType.GLIDER:
        req.speed_kmh = extract_speed(text)
        # Look for glide ratio requirement
        match = re.search(r"(\d+):1|(\d+)/1|l/d\s*(?:of\s*)?(\d+)", text.lower())
        if match:
            ratio = match.group(1) or match.group(2) or match.group(3)
            req.constraints["target_ld"] = float(ratio)

    return req


def parser_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for requirement parsing.

    Args:
        state: Current graph state

    Returns:
        State updates
    """
    user_input = state["user_input"]
    vehicle_type = state["vehicle_type"]

    # Parse requirements
    requirements = parse_requirements(user_input, vehicle_type)

    # Build summary message
    summary_parts = []
    if requirements.payload_kg:
        summary_parts.append(f"Payload: {requirements.payload_kg} kg")
    if requirements.range_km:
        summary_parts.append(f"Range: {requirements.range_km} km")
    if requirements.endurance_hours:
        summary_parts.append(f"Endurance: {requirements.endurance_hours:.2f} hours")
    if requirements.speed_kmh:
        summary_parts.append(f"Speed: {requirements.speed_kmh} km/h")
    if requirements.target_altitude_m:
        summary_parts.append(f"Target altitude: {requirements.target_altitude_m} m")
    if requirements.orbit_altitude_km:
        summary_parts.append(f"Orbit: {requirements.orbit_altitude_km} km")

    summary = "; ".join(summary_parts) if summary_parts else "Basic requirements only"

    return {
        "requirements": requirements,
        "current_phase": DesignPhase.SEARCHING,
        "messages": [{"role": "system", "content": f"Parsed requirements: {summary}"}],
    }


if __name__ == "__main__":
    # Test parsing
    test_cases = [
        ("Design a drone that can carry 2kg payload for 30 minutes", VehicleType.DRONE),
        (
            "I need a small airplane for 2 passengers with 500km range and 250 km/h cruise",
            VehicleType.FIXED_WING,
        ),
        (
            "Design a helicopter for medical evacuation, 400km range",
            VehicleType.HELICOPTER,
        ),
        (
            "Build a model rocket to reach 1km altitude with 500g payload",
            VehicleType.ROCKET,
        ),
        (
            "LEO satellite for Earth observation, 400km orbit, 5 year mission, 50kg",
            VehicleType.SATELLITE,
        ),
        ("Competition glider with 50:1 glide ratio", VehicleType.GLIDER),
    ]

    print("=== Requirement Parsing Tests ===\n")

    for text, vtype in test_cases:
        print(f"Input: {text}")
        print(f"Type: {vtype.value}")

        req = parse_requirements(text, vtype)

        # Print non-None fields
        for field_name, value in asdict(req).items():
            if value is not None and field_name != "raw_input":
                if isinstance(value, dict) and not value:
                    continue
                print(f"  {field_name}: {value}")
        print()
