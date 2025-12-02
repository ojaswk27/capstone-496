"""
LLM Formula Extractor - extracts formulas from research papers using Claude.
"""

import json
import os
import sys
from pathlib import Path
from typing import List

from anthropic import Anthropic

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.state import ExtractedFormula

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


def extract_formulas_llm(
    search_text: str, source: str = "RAG Document"
) -> List[ExtractedFormula]:
    """Extract engineering formulas from technical text using LLM."""

    if not client or not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set, skipping formula extraction")
        return []

    prompt = f"""From the following aerospace engineering text, extract all formulas.

For each formula provide:
- name: Brief descriptive name
- formula: The equation as a string
- variables: Dict of variable names and descriptions
- applicable_to: List of vehicle types (e.g., ["drone", "fixed_wing"])

Return ONLY a JSON array of formula objects.

Text:
{search_text[:3000]}"""

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = resp.content[0].text.strip()
        json_text = _strip_markdown_json(response_text)
        data = json.loads(json_text)

        formulas = []
        for item in data:
            formulas.append(
                ExtractedFormula(
                    name=item.get("name", "Unknown Formula"),
                    formula=item.get("formula", ""),
                    variables=item.get("variables", {}),
                    source=source,
                    applicable_to=item.get("applicable_to", ["unknown"]),
                    confidence=0.8,
                )
            )

        return formulas

    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse LLM response: {e}")
        return []
    except Exception as e:
        print(f"⚠️  Formula extraction failed: {e}")
        return []
