"""
Parameter Agent — fills in missing design parameters using LLM reasoning.
Replaces: llm_parameter_completer.py
"""
import json

from graph.state import DesignPhase, DesignState
from llm.client import OllamaClient

SYSTEM_PROMPT = """You are an aerospace engineer completing a vehicle design specification.
Given a vehicle type and partial requirements, fill in ALL missing parameters.

Base decisions on:
1. Vehicle type and typical performance for that category
2. Already-specified parameters (keep these EXACTLY as given)
3. Engineering relationships (range = endurance x speed)
4. Real-world examples of similar vehicles

Guidelines by vehicle type:
- Small drone (<5kg): 40-80 km/h cruise, 10-60 min endurance
- Fixed-wing UAV (<25kg): 50-90 km/h, 1-6 hour endurance
- Tactical UAV: 80-150 km/h, 4-12 hour endurance
- Manned aircraft: 150-300 km/h, 2-6 hour endurance
- Helicopter: 60-250 km/h depending on size
- Model rocket: altitude in meters, solid motor typical
- Satellite: orbit in km, mission in years

Respond with ONLY a JSON object:
{
  "payload_kg": float,
  "endurance_hours": float,
  "range_km": float,
  "speed_kmh": float,
  "altitude_m": float,
  "mission_type": "string",
  "vehicle_specific": {},
  "reasoning": "Brief explanation"
}"""


def parameter_agent(state: DesignState) -> DesignState:
    """LangGraph node: complete missing parameters."""
    client = OllamaClient()
    req = state.requirements

    # Build context of what's already specified
    specified = {}
    for field in ["payload_kg", "endurance_hours", "range_km", "speed_kmh", "altitude_m", "mission_type"]:
        val = getattr(req, field, None)
        if val is not None:
            specified[field] = val
    if req.vehicle_specific:
        specified["vehicle_specific"] = req.vehicle_specific

    prompt = f"""Vehicle type: {state.vehicle_type}
User request: {req.raw_input}
Currently specified: {json.dumps(specified)}

Complete ALL missing parameters."""

    data = client.chat_json(prompt, system_prompt=SYSTEM_PROMPT)

    if data is None:
        state.warnings.append("Parameter completion failed, proceeding with partial params")
        state.phase = DesignPhase.DESIGNING
        return state

    # Fill missing params — do NOT overwrite user-specified values
    for field in ["payload_kg", "endurance_hours", "range_km", "speed_kmh", "altitude_m", "mission_type"]:
        current = getattr(req, field, None)
        if current is None and data.get(field) is not None:
            setattr(req, field, data[field])

    if data.get("vehicle_specific"):
        for key, val in data["vehicle_specific"].items():
            if key not in req.vehicle_specific:
                req.vehicle_specific[key] = val

    state.requirements = req
    state.metadata["parameter_completion"] = {
        "reasoning": data.get("reasoning", ""),
        "original": specified,
    }
    state.agent_messages.append({"agent": "parameter", "data": data})
    state.phase = DesignPhase.DESIGNING
    return state
