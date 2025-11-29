import os
import json
import re
from typing import List, Dict, Any
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AerospaceExtractor:
    def __init__(self, retriever_instance):
        self.retriever = retriever_instance

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("⚠️  WARNING: ANTHROPIC_API_KEY not found in .env file.")

        self.client = Anthropic(api_key=api_key)

    def _clean_json_text(self, text: str) -> str:
        """Helper to strip markdown formatting from LLM response."""
        # Remove ```json (or other language) start tags
        text = re.sub(r'^```[a-zA-Z]*\s*', '', text, flags=re.MULTILINE)
        # Remove ``` end tags
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        return text.strip()

    def extract_formulas(self, query: str, vehicle_type: str, desired_output: str = None) -> List[Dict]:
        """
        1. Retrieves context from the Vector DB.
        2. Asks Claude to extract formulas from that context.
        3. Returns structured JSON.

        Args:
            query: What physics to simulate (e.g., "hover thrust")
            vehicle_type: Type of vehicle (e.g., "drone")
            desired_output: What variable to solve for (e.g., "thrust", "velocity")
        """

        # 1. Get Context (using the new API)
        print(f"🔍 Retrieving context for: '{query}'...")
        results = self.retriever.query(user_query=query, vehicle_type=vehicle_type, k=3)
        context_text = self.retriever.format_results(results)

        if not context_text:
            print("⚠️  No context found in documents.")
            return []

        # 2. Construct Prompt
        system_prompt = """You are an expert Aerospace Engineer specializing in Python simulation.
        Your task is to extract mathematical formulas from the provided technical papers.
        """

        output_instruction = ""
        if desired_output:
            output_instruction = f"""
        ⚠️ CRITICAL REQUIREMENTS ⚠️:
        1. The formula MUST calculate "{desired_output}" as the OUTPUT
        2. The formula must NOT require "{desired_output}" as an INPUT
        3. PREFER THE SIMPLEST, MOST DIRECT formula available
        4. If you see "T = mass * g" for thrust, USE THAT instead of complex momentum theory
        5. Avoid formulas that require solving for intermediate variables first

        For hover thrust specifically: The SIMPLEST formula is T = mass * g (hover equilibrium)
        """

        user_message = f"""
        I need to simulate the following physics: "{query}"
        {output_instruction}

        Here is the reference text from technical documents:
        {context_text}

        Please extract the most relevant formulas and convert them into a JSON format.

        Rules:
        1. Return ONLY a JSON object containing a "formulas" list.
        2. The "expression" field must be valid, executable Python code (use math.sin, math.sqrt, etc.).
        3. Map variables clearly in the "variables" dictionary.
        4. Do not include any conversational text, only the JSON.
        5. The formula should CALCULATE the desired output, not require it as input.

        Required JSON Structure:
        {{
            "formulas": [
                {{
                    "name": "Formula Name",
                    "expression": "mass * g",
                    "variables": {{"mass": "Total aircraft mass", "g": "Gravitational acceleration"}},
                    "units": {{"mass": "kg", "g": "m/s^2"}},
                    "description": "Calculates weight/thrust for hover",
                    "output_variable": "thrust"
                }}
            ]
        }}

        EXAMPLE FOR HOVER THRUST:
        GOOD: "expression": "mass * g"  ← Simple, direct, calculates thrust
        BAD:  "expression": "math.sqrt(T**3 / (2 * rho * A))"  ← Requires T as input!
        BAD:  "expression": "math.sqrt((mass * g)**2 / (2 * rho * A))"  ← Unnecessarily complex!
        """

        # 3. Call Anthropic API
        try:
            print("🤖 Calling Claude for extraction...")
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0
            )

            raw_content = response.content[0].text
            clean_content = self._clean_json_text(raw_content)

            data = json.loads(clean_content)

            # Robustly handle structure
            if "formulas" in data:
                return data["formulas"]
            elif isinstance(data, list):
                return data
            else:
                return [data]

        except json.JSONDecodeError:
            print(f"❌ Failed to parse JSON from Claude. Raw output:\n{raw_content[:200]}...")
            return []
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return []