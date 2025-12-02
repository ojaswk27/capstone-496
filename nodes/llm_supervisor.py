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

SYSTEM_PROMPT = """You are an aerospace expert. Analyze the user request and return **only** a JSON object:
{
  "valid": bool,
  "vehicle": "drone"|"fixed_wing"|"helicopter"|"rocket"|"satellite"|"glider"|"unknown",
  "payload_kg": float or null,
  "endurance_hours": float or null,
  "range_km": float or null,
  "speed_kmh": float or null,
  "target_altitude_m": float or null,
  "orbit_altitude_km": float or null,
  "num_passengers": int or null,
  "mission_years": float or null,
  "mission_type": string or null,
  "reason": string
}

Extract numerical values and vehicle type. If invalid/unclear, set valid=false."""


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
            text = text[:-3]
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
            max_tokens=500,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_text}],
        )

        # Extract text from response
        response_text = response.content[0].text.strip()

        # Strip markdown code blocks if present
        json_text = _strip_markdown_json(response_text)

        # Parse JSON
        data = json.loads(json_text)

        if not data.get("valid", True):
            state.phase = DesignPhase.ERROR
            state.errors.append(f"Invalid request: {data.get('reason', 'Unknown')}")
            return state

        # Set vehicle type
        try:
            state.vehicle_type = VehicleType(data.get("vehicle", "unknown"))
        except ValueError:
            state.vehicle_type = VehicleType.UNKNOWN

        state.classification_confidence = 0.9
        state.classification_reasoning = data.get("reason", "LLM classification")

        # Build requirements
        if not state.requirements:
            state.requirements = UserRequirements(raw_input=user_text)

        if data.get("payload_kg") is not None:
            state.requirements.payload_kg = float(data["payload_kg"])
        if data.get("endurance_hours") is not None:
            state.requirements.endurance_hours = float(data["endurance_hours"])
        if data.get("range_km") is not None:
            state.requirements.range_km = float(data["range_km"])
        if data.get("speed_kmh") is not None:
            state.requirements.speed_kmh = float(data["speed_kmh"])
        if data.get("target_altitude_m") is not None:
            state.requirements.altitude_m = float(data["target_altitude_m"])
        if data.get("mission_type") is not None:
            state.requirements.mission_type = data["mission_type"]

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
        print(f"⚠️  JSON parsing failed: {e}")
        if "response_text" in locals():
            print(f"   Original response: {response_text[:200]}...")
        if "json_text" in locals():
            print(f"   After stripping markdown: {json_text[:200]}...")
        return _fallback_classifier(state)
    except Exception as e:
        print(f"⚠️  LLM supervisor failed: {e}, using fallback")
        return _fallback_classifier(state)


def _fallback_classifier(state: DesignState) -> DesignState:
    """Fallback to keyword-based classification."""
    try:
        from graph.nodes import classify_vehicle

        return classify_vehicle(state)
    except Exception as e:
        state.phase = DesignPhase.ERROR
        state.errors.append(f"Both LLM and fallback failed: {e}")
        return state
