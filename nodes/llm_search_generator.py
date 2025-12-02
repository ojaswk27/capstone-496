"""
LLM Search Query Generator - creates technical search queries using Claude.
"""

import json
import os
import sys
from pathlib import Path
from typing import List

from anthropic import Anthropic

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import UserRequirements, VehicleType

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


def generate_queries_llm(vehicle: VehicleType, req: UserRequirements) -> List[str]:
    """Generate technical search queries using LLM."""

    if not client or not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set, using default queries")
        return _fallback_queries(vehicle, req)

    prompt = f"""Generate 5 technical search queries for aerospace research papers about designing a {vehicle.value}.

Requirements: {req.raw_input}

Use proper aerospace terminology targeting academic/technical sources.

Return ONLY a JSON array of query strings.
Example: ["UAV propulsion systems", "quadcopter flight dynamics"]"""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=300,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = resp.content[0].text.strip()
        json_text = _strip_markdown_json(response_text)
        queries = json.loads(json_text)

        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:5]
        else:
            return _fallback_queries(vehicle, req)

    except Exception as e:
        print(f"⚠️  Query generation failed: {e}, using fallback")
        return _fallback_queries(vehicle, req)


def _fallback_queries(vehicle: VehicleType, req: UserRequirements) -> List[str]:
    """Fallback query generator using templates."""
    base_queries = {
        VehicleType.DRONE: [
            f"{vehicle.value} design",
            "UAV propulsion sizing",
            "multicopter flight time optimization",
            "quadcopter battery selection",
            "drone motor thrust calculation",
        ],
        VehicleType.FIXED_WING: [
            f"{vehicle.value} aircraft design",
            "wing sizing methodology",
            "aircraft performance estimation",
            "propeller aircraft design",
            "general aviation aircraft sizing",
        ],
        VehicleType.HELICOPTER: [
            "helicopter rotor design",
            "rotorcraft performance",
            "helicopter sizing methodology",
            "rotor disk loading calculation",
            "helicopter power requirements",
        ],
        VehicleType.ROCKET: [
            "rocket motor sizing",
            "rocket trajectory optimization",
            "model rocket design calculations",
            "rocket staging analysis",
            "solid rocket motor performance",
        ],
        VehicleType.SATELLITE: [
            "satellite design handbook",
            "spacecraft power budget",
            "orbital mechanics calculations",
            "CubeSat design guide",
            "satellite subsystem sizing",
        ],
        VehicleType.GLIDER: [
            "sailplane design",
            "glider aerodynamics",
            "glide ratio optimization",
            "unpowered aircraft design",
            "thermal soaring performance",
        ],
    }

    return base_queries.get(vehicle, [f"{vehicle.value} design methodology"])
