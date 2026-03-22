"""
Understand Agent — classifies vehicle type and extracts requirements.
Replaces: llm_supervisor + classify_vehicle + parse_requirements.
"""
from graph.state import DesignPhase, DesignState, UserRequirements
from llm.client import OllamaClient

SYSTEM_PROMPT = """You are an aerospace engineering assistant. Given a vehicle design request:

1. Identify the vehicle type: drone, fixed_wing, helicopter, rocket, satellite, glider, or unknown
2. Extract any explicitly stated requirements (numbers, specs)

Respond with ONLY a JSON object:
{
  "vehicle_type": "drone",
  "payload_kg": 2.0,
  "endurance_hours": 0.5,
  "range_km": null,
  "speed_kmh": null,
  "altitude_m": null,
  "mission_type": "surveillance",
  "vehicle_specific": {},
  "reasoning": "Brief explanation"
}

Rules:
- Use null for values not mentioned
- Convert minutes to hours (30 min = 0.5 hours)
- "drone" includes quadcopter, multirotor, UAV
- "fixed_wing" includes airplane, aircraft, plane
- For rockets, put target_altitude_m in vehicle_specific
- For satellites, put orbit_altitude_km and mission_years in vehicle_specific"""


def understand_agent(state: DesignState) -> DesignState:
    """LangGraph node: classify vehicle and extract requirements."""
    client = OllamaClient()
    data = client.chat_json(state.raw_input, system_prompt=SYSTEM_PROMPT)

    if data is None:
        state.phase = DesignPhase.ERROR
        state.errors.append("Failed to parse vehicle classification from LLM")
        return state

    # Set vehicle type
    vtype = data.get("vehicle_type", "unknown").lower()
    state.vehicle_type = vtype if vtype in [
        "drone", "fixed_wing", "helicopter", "rocket", "satellite", "glider"
    ] else "unknown"

    state.classification_confidence = 0.9 if state.vehicle_type != "unknown" else 0.3
    state.classification_reasoning = data.get("reasoning", "")

    if state.vehicle_type == "unknown":
        state.warnings.append("Could not confidently classify vehicle type")

    # Populate requirements
    req = state.requirements or UserRequirements(raw_input=state.raw_input)
    for field in ["payload_kg", "endurance_hours", "range_km", "speed_kmh", "altitude_m", "mission_type"]:
        val = data.get(field)
        if val is not None:
            setattr(req, field, val)

    if data.get("vehicle_specific"):
        req.vehicle_specific.update(data["vehicle_specific"])

    state.requirements = req
    state.agent_messages.append({"agent": "understand", "data": data})
    state.phase = DesignPhase.PARAMETERIZING
    return state
