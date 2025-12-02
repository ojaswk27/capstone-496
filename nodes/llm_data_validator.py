"""
LLM Data Validator - Reviews and corrects RAG-retrieved data for scale/context

This module uses LLM to validate that retrieved formulas and data from research
papers are appropriate for the actual vehicle being designed. It catches common
issues like applying manned aircraft parameters to small drones.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from anthropic import Anthropic

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import DesignState, ExtractedFormula, SearchResult

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


def validate_and_correct_data_llm(state: DesignState) -> DesignState:
    """
    Use LLM to review RAG-retrieved data and correct for scale/context mismatches.

    This catches issues like:
    - Manned aircraft formulas applied to drones
    - Wrong default values for the vehicle scale
    - Inappropriate assumptions from source material

    Args:
        state: Current DesignState with retrieved formulas and search results

    Returns:
        Updated DesignState with corrections applied
    """
    if not client or not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set, skipping data validation")
        return state

    # Prepare context
    # Handle both enum and string vehicle_type
    vehicle_type = (
        state.vehicle_type.value
        if hasattr(state.vehicle_type, "value")
        else str(state.vehicle_type)
    )
    req = state.requirements

    if not req:
        return state

    # Summarize retrieved formulas
    formulas_summary = []
    for formula in state.extracted_formulas[:5]:  # Top 5 formulas
        formulas_summary.append(
            {
                "name": formula.name,
                "formula": formula.formula,
                "source": formula.source,
                "variables": list(formula.variables.keys())[:5],  # First 5 variables
            }
        )

    # Summarize search results
    search_context = (
        "\n".join(
            [
                f"- {result.source}: {result.content[:150]}..."
                for result in state.search_results[:3]
            ]
        )
        if state.search_results
        else "No search results available"
    )

    prompt = f"""You are an aerospace engineer reviewing design data retrieved from research papers.

    DESIGN REQUIREMENTS:
    - Vehicle Type: {vehicle_type}
    - User Request: {req.raw_input}
    - Payload: {req.payload_kg or 'not specified'} kg
    - Endurance: {req.endurance_hours or 'not specified'} hours
    - Range: {req.range_km or 'not specified'} km
    - Speed: {req.speed_kmh or 'not specified'} km/h

    RETRIEVED DATA SUMMARY:
    Formulas: {json.dumps(formulas_summary, indent=2) if formulas_summary else 'None'}

    Search Results Context:
    {search_context}

    TASK: Review this data and identify scale/context mismatches. 

    IMPORTANT: Return ONLY valid JSON. No trailing commas, all strings quoted, proper formatting.

    Return this exact structure:
    {{
      "issues_found": [
        {{
          "issue": "description of problem",
          "severity": "high/medium/low",
          "affected_parameter": "parameter name"
        }}
      ],
      "corrections": {{
        "cruise_speed_kmh": 0.0,
        "default_range_km": 0.0,
        "weight_estimation_notes": "Adjusted for UAV scale",
        "other_adjustments": {{}}
      }},
      "reasoning": "Brief explanation of corrections made",
      "confidence": 0.85
    }}

    Rules:
    1. All numeric values must be valid JSON numbers (not null, not strings)
    2. Use null (not "null") if no value
    3. No trailing commas before closing braces or brackets
    4. All keys must be double-quoted strings
    5. confidence must be a number between 0.0 and 1.0

    Common issues to check:
    1. Are cruise speeds appropriate for vehicle scale? (4kg drone → 60-80 km/h, not 200 km/h)
    2. Are weight formulas appropriate for the vehicle size?
    3. Are default values reasonable for the payload and vehicle type?"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()
        json_text = _strip_markdown_json(response_text)
        data = json.loads(json_text)

        print("\n🔍 LLM Data Validation Results:")
        print(f"   Confidence: {data.get('confidence', 0):.0%}")
        print(f"   Reasoning: {data.get('reasoning', 'N/A')}")

        if data.get("issues_found"):
            print(f"   Issues Found: {len(data['issues_found'])}")
            for issue in data["issues_found"]:
                severity_icon = (
                    "🔴"
                    if issue["severity"] == "high"
                    else "🟡"
                    if issue["severity"] == "medium"
                    else "🟢"
                )
                print(f"   {severity_icon} {issue['issue']}")
        else:
            print("   ✅ No critical issues found")

        corrections = data.get("corrections", {})

        # Apply corrections to state
        if corrections.get("cruise_speed_kmh") and not req.speed_kmh:
            req.speed_kmh = float(corrections["cruise_speed_kmh"])
            print(f"   ✅ Corrected cruise speed to {req.speed_kmh} km/h")

        if corrections.get("default_range_km") and not req.range_km:
            if req.endurance_hours and req.speed_kmh:
                # Recalculate range with corrected speed
                req.range_km = req.endurance_hours * req.speed_kmh
                print(
                    f"   ✅ Recalculated range: {req.endurance_hours}h × {req.speed_kmh} km/h = {req.range_km} km"
                )
            else:
                req.range_km = float(corrections["default_range_km"])
                print(f"   ✅ Set default range to {req.range_km} km")

        # Store validation results in metadata
        state.metadata["llm_data_validation"] = {
            "issues": data.get("issues_found", []),
            "corrections": corrections,
            "reasoning": data.get("reasoning", ""),
            "confidence": data.get("confidence", 0),
        }

        # Add warnings for high-severity issues
        for issue in data.get("issues_found", []):
            if issue.get("severity") == "high":
                state.warnings.append(f"Data validation: {issue['issue']}")

        return state

    except json.JSONDecodeError as e:
        print(f"⚠️  LLM data validation JSON parse failed: {e}")
        return state
    except Exception as e:
        print(f"⚠️  LLM data validation failed: {e}")
        import traceback

        traceback.print_exc()
        return state
