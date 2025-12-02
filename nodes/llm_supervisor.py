"""
LLM Supervisor Node - Anthropic Claude 3.5 Sonnet
Works with the Pydantic DesignState used by the graph workflow.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from anthropic import Anthropic

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from graph.state (Pydantic)
from graph.state import DesignPhase, DesignState, UserRequirements, VehicleType

# Initialize Anthropic client
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key) if api_key else None

SYSTEM_PROMPT = """You are an aerospace engineering assistant helping to classify vehicle design requests.

Your job is to:
1. Identify what type of aerospace vehicle the user wants to design
2. Extract specific requirements (numbers, specifications)

VEHICLE TYPES:
- "drone" - Multirotor UAVs, quadcopters, octocopters, multirotors
- "fixed_wing" - Airplanes, general aviation aircraft, planes
- "helicopter" - Rotorcraft with main rotor and tail rotor
- "rocket" - Launch vehicles, model rockets, sounding rockets
- "satellite" - Spacecraft for orbit
- "glider" - Unpowered sailplanes
- "unknown" - If you cannot determine the type

RESPOND WITH ONLY A JSON OBJECT (no markdown, no explanation):
{
  "valid": true or false,
  "vehicle": "drone",
  "payload_kg": 2.0,
  "endurance_hours": 0.75,
  "range_km": null,
  "speed_kmh": null,
  "target_altitude_m": null,
  "orbit_altitude_km": null,
  "num_passengers": null,
  "mission_years": null,
  "mission_type": "surveillance",
  "reason": "User wants a surveillance drone with 2kg payload and 45 minute flight time"
}

RULES:
- Set "valid" to false only if the request is completely unclear or not about aerospace vehicles
- Convert flight time minutes to hours (e.g., 45 minutes = 0.75 hours)
- Use null for values not mentioned
- "drone" includes: quadcopter, multicopter, UAV, multirotor, quadrotor, octocopter
- Be generous in classification - if it flies/goes to space, classify it

EXAMPLES:
- "surveillance drone, 2kg payload, 45min flight" → vehicle: "drone", payload_kg: 2.0, endurance_hours: 0.75
- "quadcopter for photography" → vehicle: "drone", mission_type: "photography"
- "small airplane for 2 people" → vehicle: "fixed_wing", num_passengers: 2
- "rocket to 2km altitude" → vehicle: "rocket", target_altitude_m: 2000
- "design a drone with 1kg camera" → vehicle: "drone", payload_kg: 1.0, mission_type: "photography"
"""


def _strip_markdown_json(text: str) -> str:
    """Strip markdown code block markers from JSON response."""
    text = text.strip()
    # Remove ```json or ``` markers
    if text.startswith("```"):
        # Find first newline after opening ```
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        # Remove closing ```
        if text.endswith("```"):
            text = text[:-3].strip()
    return text.strip()


def llm_supervisor_node(state: DesignState) -> DesignState:
    """LangGraph node - uses LLM to classify vehicle and extract requirements."""
    user_text = state.raw_input

    if not client or not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set, using fallback")
        return _fallback_classifier(state)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=800,  # Increased from 500
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
        )

        # Extract text from response
        response_text = response.content[0].text.strip()

        # Debug: print what we got
        print(f"🤖 LLM Response (raw): {response_text[:300]}...")

        # Strip markdown code blocks if present
        json_text = _strip_markdown_json(response_text)
        print(f"🤖 LLM Response (cleaned): {json_text[:300]}...")

        # Parse JSON
        data = json.loads(json_text)
        print(
            f"✅ Parsed successfully: vehicle={data.get('vehicle')}, payload={data.get('payload_kg')}kg"
        )

        if not data.get("valid", True):
            state.phase = DesignPhase.ERROR
            state.errors.append(f"Invalid request: {data.get('reason', 'Unknown')}")
            return state

        # Set vehicle type
        vehicle_str = data.get("vehicle", "unknown").lower()
        try:
            state.vehicle_type = VehicleType(vehicle_str)
            print(f"✅ Classified as: {state.vehicle_type.value}")
        except ValueError:
            print(f"⚠️  Unknown vehicle type: {vehicle_str}, using UNKNOWN")
            state.vehicle_type = VehicleType.UNKNOWN

        state.classification_confidence = 0.9
        state.classification_reasoning = data.get("reason", "LLM classification")

        # Build requirements
        if not state.requirements:
            state.requirements = UserRequirements(raw_input=user_text)

        if data.get("payload_kg") is not None:
            state.requirements.payload_kg = float(data["payload_kg"])
            print(f"  📦 Payload: {state.requirements.payload_kg} kg")

        if data.get("endurance_hours") is not None:
            state.requirements.endurance_hours = float(data["endurance_hours"])
            print(f"  ⏱️  Endurance: {state.requirements.endurance_hours} hours")

        if data.get("range_km") is not None:
            state.requirements.range_km = float(data["range_km"])
            print(f"  🛫 Range: {state.requirements.range_km} km")

        if data.get("speed_kmh") is not None:
            state.requirements.speed_kmh = float(data["speed_kmh"])
            print(f"  ⚡ Speed: {state.requirements.speed_kmh} km/h")

        if data.get("target_altitude_m") is not None:
            state.requirements.altitude_m = float(data["target_altitude_m"])
            print(f"  📈 Altitude: {state.requirements.altitude_m} m")

        if data.get("mission_type") is not None:
            state.requirements.mission_type = data["mission_type"]
            print(f"  🎯 Mission: {state.requirements.mission_type}")

        # Vehicle-specific
        if data.get("orbit_altitude_km") is not None:
            state.requirements.vehicle_specific["orbit_altitude_km"] = float(
                data["orbit_altitude_km"]
            )
        if data.get("num_passengers") is not None:
            state.requirements.vehicle_specific["num_passengers"] = int(
                data["num_passengers"]
            )
        if data.get("mission_years") is not None:
            state.requirements.vehicle_specific["mission_years"] = float(
                data["mission_years"]
            )

        state.phase = DesignPhase.PARSING
        return state

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        if "response_text" in locals():
            print(f"   Original response: {response_text[:500]}...")
        if "json_text" in locals():
            print(f"   After stripping markdown: {json_text[:500]}...")
        return _fallback_classifier(state)
    except Exception as e:
        print(f"❌ LLM supervisor failed: {e}, using fallback")
        import traceback

        traceback.print_exc()
        return _fallback_classifier(state)


def _fallback_classifier(state: DesignState) -> DesignState:
    """Fallback to keyword-based classification."""
    try:
        from graph.nodes import classify_vehicle

        print("🔄 Using fallback keyword-based classifier...")
        return classify_vehicle(state)
    except Exception as e:
        state.phase = DesignPhase.ERROR
        state.errors.append(f"Both LLM and fallback failed: {e}")
        return state
