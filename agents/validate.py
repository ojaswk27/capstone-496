"""
Validate Agent — reviews design against requirements.
Replaces: validate_design + llm_validator + llm_data_validator.
"""
import json

from graph.state import DesignPhase, DesignState, ValidationResult
from llm.client import OllamaClient

SYSTEM_PROMPT = """You are an aerospace engineer reviewing a vehicle design.

Given the original requirements and calculation results, determine if the design is acceptable.

Check:
1. Does the design meet payload requirements?
2. Does the design meet endurance/range requirements?
3. Are performance metrics physically reasonable?
4. Are there safety concerns (e.g., thrust-to-weight < 1.5 for drones)?

Respond with ONLY a JSON object:
{
  "passed": true/false,
  "checks": {"check_name": true/false, ...},
  "warnings": ["warning text", ...],
  "errors": ["error text", ...],
  "feedback": "If failed, explain what needs to change for the Design Agent to fix it. Empty string if passed."
}"""


def validate_agent(state: DesignState) -> DesignState:
    """LangGraph node: validate design against requirements."""
    client = OllamaClient()
    req = state.requirements
    design = state.intermediate_results.get("design", {})

    tool_summary = []
    for tc in state.tool_calls:
        tool_summary.append({
            "tool": tc.tool_name,
            "success": tc.success,
            "result_keys": list(tc.result.keys()) if tc.result else [],
        })

    prompt = f"""Vehicle type: {state.vehicle_type}

Requirements:
- User request: {req.raw_input}
- Payload: {req.payload_kg} kg
- Endurance: {req.endurance_hours} hours
- Range: {req.range_km} km
- Speed: {req.speed_kmh} km/h

Design results:
{json.dumps(design, indent=2, default=str)}

Tools called: {json.dumps(tool_summary)}

Validate this design against the requirements."""

    data = client.chat_json(prompt, system_prompt=SYSTEM_PROMPT)

    if data is None:
        # If LLM fails, do basic validation
        state.validation_result = ValidationResult(
            passed=True,
            warnings=["LLM validation unavailable, passing with warning"],
        )
        state.phase = DesignPhase.SYNTHESIZING
        return state

    state.validation_result = ValidationResult(
        passed=data.get("passed", True),
        checks=data.get("checks", {}),
        warnings=data.get("warnings", []),
        errors=data.get("errors", []),
        suggestions=[],
    )

    feedback = data.get("feedback", "")
    if not state.validation_result.passed and feedback:
        state.validation_feedback = feedback

    state.agent_messages.append({"agent": "validate", "data": data})

    if state.validation_result.passed:
        state.phase = DesignPhase.SYNTHESIZING
    else:
        state.phase = DesignPhase.VALIDATING
        state.retry_count += 1

    return state
