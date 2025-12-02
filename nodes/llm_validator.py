"""
LLM Validator - validates design outputs against requirements using Claude.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from anthropic import Anthropic

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import UserRequirements, ValidationResult

# Initialize client
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key) if api_key else None


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


def validate_with_llm(
    vehicle: str, calc_outputs: Dict[str, Any], requirements: UserRequirements
) -> ValidationResult:
    """Validate calculated design against requirements using LLM."""

    if not client or not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set, using basic validation")
        return _fallback_validation(calc_outputs, requirements)

    prompt = f"""You are an aerospace engineer. Validate this design against requirements.

Vehicle Type: {vehicle}
Requirements: {requirements.raw_input}
Calculated Outputs: {json.dumps(calc_outputs, indent=2)}

Analyze if the design meets requirements. Return ONLY a JSON object:
{{
  "passed": bool,
  "checks": {{"check_name": true/false}},
  "warnings": ["warning1", ...],
  "errors": ["error1", ...],
  "suggestions": ["suggestion1", ...]
}}

Check: payload capacity, flight time/endurance, performance metrics, safety/stability."""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=800,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = resp.content[0].text.strip()
        json_text = _strip_markdown_json(response_text)
        data = json.loads(json_text)

        return ValidationResult(
            passed=data.get("passed", True),
            checks=data.get("checks", {}),
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
            suggestions=data.get("suggestions", []),
        )

    except Exception as e:
        print(f"⚠️  LLM validation failed: {e}, using fallback")
        return _fallback_validation(calc_outputs, requirements)


def _fallback_validation(
    calc_outputs: Dict[str, Any], requirements: UserRequirements
) -> ValidationResult:
    """Simple rule-based validation fallback."""
    warnings = []
    errors = []
    checks = {}

    # Basic checks
    if requirements.payload_kg:
        calc_payload = calc_outputs.get(
            "payload_kg", calc_outputs.get("payload_capacity", 0)
        )
        if calc_payload < requirements.payload_kg:
            errors.append(
                f"Payload capacity ({calc_payload} kg) < required ({requirements.payload_kg} kg)"
            )
            checks["payload"] = False
        else:
            checks["payload"] = True

    if requirements.endurance_hours:
        calc_endurance = calc_outputs.get(
            "endurance_hours", calc_outputs.get("flight_time_hours", 0)
        )
        if calc_endurance < requirements.endurance_hours:
            warnings.append(
                f"Flight time ({calc_endurance:.1f}h) < desired ({requirements.endurance_hours}h)"
            )
            checks["endurance"] = False
        else:
            checks["endurance"] = True

    passed = len(errors) == 0

    return ValidationResult(
        passed=passed,
        checks=checks,
        warnings=warnings,
        errors=errors,
        suggestions=["Consider adding safety margins", "Verify weight estimates"],
    )
