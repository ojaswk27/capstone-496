"""
Design Agent — calls calculation tools via Ollama function-calling.
This is the core agentic node. The LLM decides which tools to call,
inspects results, and can call additional tools.
Replaces: the hardcoded perform_calculations() switch statement.
"""
import json
from dataclasses import asdict
from typing import Any, Dict

from config import get_config
from graph.state import DesignPhase, DesignState, ToolCallRecord
from llm.client import OllamaClient
from llm.tools import get_tool_function, get_tools_for_vehicle_type, validate_tool_args

SYSTEM_PROMPT = """You are an aerospace design engineer with access to calculation tools.

Given vehicle requirements, call the appropriate tool(s) to produce a complete design.
- Review the tool results and call additional tools if needed.
- When you have enough data for a complete design, respond with a text summary.
- If a tool call fails, try adjusting the parameters or use a different tool.

Important:
- All numeric parameters must be numbers (not strings).
- Call the main design/sizing tool first for a complete design.
- You can call utility tools afterward for additional analysis."""


def _serialize_result(result: Any) -> Dict[str, Any]:
    """Convert a tool result (often a dataclass) to a JSON-serializable dict."""
    if hasattr(result, "__dataclass_fields__"):
        d = {}
        for field_name in result.__dataclass_fields__:
            val = getattr(result, field_name)
            if hasattr(val, "__dataclass_fields__"):
                d[field_name] = _serialize_result(val)
            elif isinstance(val, list):
                d[field_name] = [
                    _serialize_result(item) if hasattr(item, "__dataclass_fields__") else item
                    for item in val
                ]
            else:
                d[field_name] = val
        return d
    if isinstance(result, (int, float, str, bool)):
        return {"value": result}
    if isinstance(result, tuple):
        return {"values": list(result)}
    return {"result": str(result)}


def design_agent(state: DesignState) -> DesignState:
    """LangGraph node: call tools to produce a design."""
    client = OllamaClient()
    cfg = get_config().llm
    tools = get_tools_for_vehicle_type(state.vehicle_type)

    req = state.requirements
    prompt = f"""Design a {state.vehicle_type} vehicle with these requirements:
- Payload: {req.payload_kg} kg
- Endurance: {req.endurance_hours} hours
- Range: {req.range_km} km
- Speed: {req.speed_kmh} km/h
- Altitude: {req.altitude_m} m
- Mission: {req.mission_type}
- Vehicle-specific: {json.dumps(req.vehicle_specific)}

Call the appropriate calculation tool(s) to size this vehicle."""

    if state.validation_feedback:
        prompt += f"\n\nPrevious design had issues:\n{state.validation_feedback}\nAdjust parameters to fix these issues."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    last_result = None

    for _ in range(cfg.max_tool_calls):
        response = client.chat_with_tools(
            prompt="",
            tools=tools,
            messages=messages,
        )

        if not response.tool_calls:
            break

        for tc in response.tool_calls:
            try:
                cleaned_args = validate_tool_args(tc.name, tc.arguments)
                func = get_tool_function(tc.name)
                raw_result = func(**cleaned_args)
                result_dict = _serialize_result(raw_result)
                last_result = result_dict

                state.tool_calls.append(ToolCallRecord(
                    tool_name=tc.name,
                    arguments=cleaned_args,
                    result=result_dict,
                    success=True,
                ))

                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": tc.name, "arguments": tc.arguments}}
                ]})
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result_dict, default=str),
                })

            except Exception as e:
                error_msg = str(e)
                state.tool_calls.append(ToolCallRecord(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    result={},
                    success=False,
                    error=error_msg,
                ))
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": tc.name, "arguments": tc.arguments}}
                ]})
                messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": error_msg}),
                })

    # Store the last successful tool result as the design
    if last_result:
        state.intermediate_results["design"] = last_result
    elif not any(tc.success for tc in state.tool_calls):
        state.errors.append("Design agent failed: no successful tool calls")

    state.agent_messages.append({
        "agent": "design",
        "tool_calls_count": len(state.tool_calls),
    })
    state.phase = DesignPhase.VALIDATING
    return state
