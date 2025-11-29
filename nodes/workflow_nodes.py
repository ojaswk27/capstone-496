import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from anthropic import Anthropic

from state import AgentState, DesignRequirement, CalculationResult
from tools.aerospace_tools import AerospaceAgentTools

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=ANTHROPIC_API_KEY)

tools = AerospaceAgentTools()  # Reuse your unified tool wrapper


def _clean_json_text(text: str) -> str:
    """Helper to strip markdown formatting from LLM response."""
    # Remove ```json (or other language) start tags
    text = re.sub(r'^```[a-zA-Z]*\s*', '', text, flags=re.MULTILINE)
    # Remove ``` end tags
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _call_claude(system_prompt: str, user_prompt: str) -> str:
    """Small helper to call Claude and return plain text."""

    resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        temperature=0.1,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


# ============================================================================
# FIXED NODES - Each returns ONLY the delta, not full message history
# ============================================================================

def classify_vehicle_node(state: AgentState) -> Dict[str, Any]:
    """Classify vehicle type and goal."""
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else ""

    system_prompt = (
        "You are an aerospace design assistant. "
        "Classify the user's target vehicle type and summarize their goal."
    )
    user_prompt = f"""
User message:
{last_msg}

YOU MUST RETURN ONLY A VALID JSON OBJECT WITH NO ADDITIONAL TEXT OR EXPLANATION.
DO NOT include markdown formatting, backticks, or any text before or after the JSON.

Required JSON format:
{{
  "vehicle_type": "drone | fixed_wing | rocket | satellite | helicopter | glider",
  "design_goal": "short summary"
}}

Example valid response:
{{"vehicle_type": "drone", "design_goal": "long-endurance photography platform"}}
"""

    raw = _call_claude(system_prompt, user_prompt)
    try:
        clean_raw = _clean_json_text(raw)
        data = json.loads(clean_raw)
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error in classify_vehicle_node:")
        print(f"Raw response: {raw[:300]}")
        return {"error": f"Failed to parse vehicle classification JSON: {e}"}

    vehicle_type = data.get("vehicle_type")
    design_goal = data.get("design_goal")

    # Return ONLY the new message (add reducer will append it)
    return {
        "vehicle_type": vehicle_type,
        "design_goal": design_goal,
        "current_step": "requirements",
        "messages": [f"Assistant: Classified as {vehicle_type} for goal: {design_goal}"]
    }


def extract_requirements_node(state: AgentState) -> Dict[str, Any]:
    """Extract structured requirements."""
    messages = state.get("messages", [])
    convo_text = "\n".join(messages)

    system_prompt = (
        "You are an aerospace systems engineer. "
        "Extract quantitative design requirements from the conversation."
    )
    user_prompt = f"""
Conversation so far:
{convo_text}

YOU MUST RETURN ONLY A VALID JSON OBJECT WITH NO ADDITIONAL TEXT OR EXPLANATION.
DO NOT include markdown formatting, backticks, or any text before or after the JSON.

Required JSON format:
{{
  "requirements": [
    {{
      "parameter": "flight_time",
      "value": 30,
      "unit": "minutes",
      "description": "Minimum flight time in hover"
    }}
  ]
}}

If no specific requirements can be extracted, return: {{"requirements": []}}
"""

    raw = _call_claude(system_prompt, user_prompt)
    try:
        clean_raw = _clean_json_text(raw)
        data = json.loads(clean_raw)
        reqs = data.get("requirements", [])
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error in extract_requirements_node:")
        print(f"Raw response: {raw[:300]}")
        return {"error": f"Failed to parse requirements JSON: {e}"}

    # requirements does NOT use add reducer, so you can append manually if you want,
    # but since it's a plain List, nodes should return the FULL new list
    existing = state.get("requirements", [])
    new_reqs = existing + reqs

    summary_line = f"Extracted {len(reqs)} additional requirement(s)."

    # Return ONLY the new message (add reducer appends it to messages)
    # Return FULL new requirements list (no reducer, so overwrite)
    return {
        "requirements": new_reqs,
        "messages": [f"Assistant: {summary_line}"],
        "current_step": "analysis",
    }


def plan_calculations_node(state: AgentState) -> Dict[str, Any]:
    """Plan which physics calculations to run."""
    vehicle_type = state.get("vehicle_type")
    reqs = state.get("requirements", [])

    system_prompt = (
        "You are an aerospace performance analyst. "
        "Given the requirements and vehicle type, propose which formulas "
        "and physics calculations should be run (in plain text)."
    )
    user_prompt = f"""
Vehicle type: {vehicle_type}
Requirements: {json.dumps(reqs, indent=2)}

Describe which key equations (by name) should be used next.
Keep it under 5 bullet points. Do NOT return JSON.
"""

    plan_text = _call_claude(system_prompt, user_prompt)

    # Return ONLY the new message
    return {
        "messages": [f"Assistant (plan): {plan_text}"],
        "current_step": "calculation",
    }


def run_calculations_node(state: AgentState) -> Dict[str, Any]:
    """Run one physics calculation."""
    vehicle_type = state.get("vehicle_type") or "drone"

    if vehicle_type == "drone":
        query = "hover thrust using momentum theory"
        # For thrust calculation: T = mass * g (hover condition)
        # Assume total mass = payload (2kg) + airframe (~3kg) = 5kg estimate
        inputs = {
            "mass": 5.0,  # Total drone mass in kg
            "g": 9.81,  # Gravitational acceleration
            "rho": 1.225,  # Air density (may be needed for some formulas)
            "A": 0.2,  # Rotor disk area (may be needed)
        }
        unit = "N"
        variable = "thrust_hover"
    elif vehicle_type == "rocket":
        query = "delta v using Tsiolkovsky rocket equation"
        inputs = {"I_sp": 250.0, "g_0": 9.81, "m_0": 100.0, "m_f": 20.0}
        unit = "m/s"
        variable = "delta_v"
    else:
        return {
            "messages": [f"Assistant: No demo calculation configured for {vehicle_type} yet."],
            "current_step": "design",
        }

    print(f"DEBUG: About to calculate with query='{query}', type='{vehicle_type}'")
    result = tools.solve_physics_problem(query=query, vehicle_type=vehicle_type, inputs=inputs)

    print(f"DEBUG: Calculation result: {result}")
    print(f"DEBUG: Formula name: {result.get('formula_name')}")
    print(f"DEBUG: Formula code: {result.get('formula_code')}")
    print(f"DEBUG: Inputs used: {result.get('inputs_used')}")
    print(f"DEBUG: Result value: {result.get('result')}")

    calc_entry: CalculationResult = {
        "variable": variable,
        "value": float(result.get("result")) if isinstance(result.get("result"), (int, float)) else 0.0,
        "unit": unit,
        "formula_used": result.get("formula_name", "unknown"),
    }

    existing_calcs = state.get("calculations", [])
    new_calcs = existing_calcs + [calc_entry]

    msg = (
        f"Computed {variable} = {calc_entry['value']} {unit} "
        f"using {calc_entry['formula_used']}."
    )

    # Return ONLY the new message, and full new calculations list
    return {
        "calculations": new_calcs,
        "messages": [f"Assistant (calc): {msg}"],
        "current_step": "design",
    }


def generate_design_node(state: AgentState) -> Dict[str, Any]:
    """Generate preliminary design summary."""
    system_prompt = (
        "You are an aerospace design engineer. "
        "Write a concise preliminary design summary based on the data."
    )
    user_prompt = f"""
Vehicle type: {state.get('vehicle_type')}
Goal: {state.get('design_goal')}

Requirements:
{json.dumps(state.get('requirements', []), indent=2)}

Key calculations:
{json.dumps(state.get('calculations', []), indent=2)}

Write a 2–3 paragraph technical summary of the current design state.
"""

    summary = _call_claude(system_prompt, user_prompt)

    # Return ONLY the new message
    return {
        "messages": [f"Assistant (design): {summary}"],
        "current_step": "review",
    }


def review_design_node(state: AgentState) -> Dict[str, Any]:
    """Review design against requirements."""
    system_prompt = (
        "You are reviewing an aerospace vehicle design. "
        "Compare requirements and current calculations and state if they seem consistent."
    )
    user_prompt = f"""
Requirements:
{json.dumps(state.get('requirements', []), indent=2)}

Calculations:
{json.dumps(state.get('calculations', []), indent=2)}

State briefly whether the design appears feasible and call out any obvious issues.
"""

    review = _call_claude(system_prompt, user_prompt)

    # Return ONLY the new message
    return {
        "messages": [f"Assistant (review): {review}"],
        "current_step": "refine",
    }


def refine_design_node(state: AgentState) -> Dict[str, Any]:
    """Propose design improvements."""
    system_prompt = (
        "You are improving an aerospace design. "
        "Suggest specific parameter changes to better meet the requirements."
    )
    user_prompt = f"""
Current design:
Vehicle: {state.get('vehicle_type')}
Goal: {state.get('design_goal')}

Full conversation so far:
{chr(10).join(state.get('messages', [])[-10:])}  # Last 10 messages for context

Suggest 1–3 specific design tweaks.
"""

    suggestions = _call_claude(system_prompt, user_prompt)

    # Return ONLY the new message
    return {
        "messages": [f"Assistant (refine): {suggestions}"],
        "current_step": "finalize",
    }


def finalize_node(state: AgentState) -> Dict[str, Any]:
    """Mark design as complete."""
    return {
        "is_complete": True,
        "current_step": "done",
        "messages": ["Assistant: Design session marked as complete."],
    }