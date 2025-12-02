"""
LLM Parameter Completer - Intelligently fills in missing design parameters

Uses LLM reasoning to infer appropriate values for any missing parameters
based on vehicle type and specified requirements. No hardcoded defaults.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from anthropic import Anthropic

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import DesignState, VehicleType

# Initialize client
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key) if api_key else None


def _strip_markdown_json(text: str) -> str:
    """Strip markdown code block markers from JSON response."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def complete_parameters_with_llm(state: DesignState) -> DesignState:
    """
    Use LLM to intelligently complete any missing design parameters.

    Instead of using hardcoded defaults, the LLM reasons about what values
    make sense given the vehicle type and specified requirements.

    Args:
        state: Current DesignState with possibly incomplete requirements

    Returns:
        Updated DesignState with all parameters filled in
    """
    if not client or not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set, cannot complete parameters intelligently")
        return state

    req = state.requirements
    if not req:
        return state

    # Handle both enum and string vehicle_type
    vehicle_type = (
        state.vehicle_type.value
        if hasattr(state.vehicle_type, "value")
        else str(state.vehicle_type)
    )

    # Build current parameters dict
    current_params = {
        "payload_kg": req.payload_kg,
        "endurance_hours": req.endurance_hours,
        "range_km": req.range_km,
        "speed_kmh": req.speed_kmh,
        "altitude_m": req.altitude_m,
        "mission_type": req.mission_type,
    }

    # Add vehicle-specific parameters
    for key, value in req.vehicle_specific.items():
        current_params[key] = value

    # Remove None values to see what's missing
    specified = {k: v for k, v in current_params.items() if v is not None}

    prompt = f"""You are an aerospace engineer completing a vehicle design specification.

VEHICLE TYPE: {vehicle_type}

USER REQUEST: {req.raw_input}

CURRENTLY SPECIFIED PARAMETERS:
{json.dumps(specified, indent=2)}

TASK: Intelligently infer ALL missing parameters that would be needed to design this vehicle.
Base your decisions on:
1. The vehicle type and typical performance for that category
2. The specified parameters (use these as constraints)
3. Engineering relationships (e.g., range = endurance × speed)
4. Real-world examples of similar vehicles

Return ONLY a JSON object with ALL required parameters:
{{
  "payload_kg": float,
  "endurance_hours": float,
  "range_km": float,
  "speed_kmh": float (cruise speed),
  "altitude_m": float (operating altitude),
  "mission_type": string,
  "vehicle_specific": {{
    // Add any vehicle-specific parameters here
    // For drones: "num_motors", "application"
    // For fixed_wing: "propulsion_type"
    // For helicopters: "rotor_config"
    // For rockets: "target_altitude_m", "motor_type"
    // For satellites: "orbit_altitude_km", "mission_years"
    // For gliders: "pilot_weight_kg", "glider_class"
  }},
  "reasoning": "Explain your parameter choices and any assumptions"
}}

IMPORTANT GUIDELINES:
1. Keep existing specified values EXACTLY as given (don't change them)
2. Ensure parameters are internally consistent (e.g., range ≤ endurance × speed)
3. Use realistic values based on vehicle type:
   - Drone (small): 40-80 km/h, 10-60 min endurance
   - Fixed-wing UAV (<25kg): 50-90 km/h, 1-6 hour endurance
   - Fixed-wing UAV (tactical): 80-150 km/h, 4-12 hour endurance
   - Manned aircraft: 150-300 km/h, 2-6 hour endurance
   - Helicopter (small): 60-120 km/h, 1-3 hour endurance
   - Helicopter (manned): 150-250 km/h, 2-4 hour endurance
4. If endurance is specified but not range, calculate: range = endurance × appropriate_cruise_speed
5. If range is specified but not endurance, calculate: endurance = range / appropriate_cruise_speed
6. Choose conservative, proven values rather than theoretical maximums"""

    try:
        print("\n🤖 LLM completing missing parameters...")

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()
        json_text = _strip_markdown_json(response_text)
        data = json.loads(json_text)

        # Update state with completed parameters
        req.payload_kg = data.get("payload_kg", req.payload_kg)
        req.endurance_hours = data.get("endurance_hours", req.endurance_hours)
        req.range_km = data.get("range_km", req.range_km)
        req.speed_kmh = data.get("speed_kmh", req.speed_kmh)
        req.altitude_m = data.get("altitude_m", req.altitude_m)
        req.mission_type = data.get("mission_type", req.mission_type or "general")

        # Update vehicle-specific parameters
        if data.get("vehicle_specific"):
            for key, value in data["vehicle_specific"].items():
                req.vehicle_specific[key] = value

        # Print results
        print("   ✅ Parameters completed:")
        print(f"      Payload: {req.payload_kg} kg")
        print(f"      Endurance: {req.endurance_hours} hours")
        print(f"      Range: {req.range_km} km")
        print(f"      Cruise Speed: {req.speed_kmh} km/h")
        print(f"      Altitude: {req.altitude_m} m")
        if req.vehicle_specific:
            print(f"      Vehicle-specific: {req.vehicle_specific}")
        print(f"\n   💭 Reasoning: {data.get('reasoning', 'N/A')}")

        # Store reasoning in metadata
        state.metadata["parameter_completion"] = {
            "reasoning": data.get("reasoning", ""),
            "original_params": specified,
            "completed_params": {
                "payload_kg": req.payload_kg,
                "endurance_hours": req.endurance_hours,
                "range_km": req.range_km,
                "speed_kmh": req.speed_kmh,
                "altitude_m": req.altitude_m,
                "vehicle_specific": req.vehicle_specific,
            },
        }

        return state

    except json.JSONDecodeError as e:
        print(f"⚠️  LLM parameter completion JSON parse failed: {e}")
        return state
    except Exception as e:
        print(f"⚠️  LLM parameter completion failed: {e}")
        import traceback

        traceback.print_exc()
        return state
