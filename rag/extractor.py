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
            print("⚠️  WARNING: ANTHROPIC_API_KEY not found.")
        self.client = Anthropic(api_key=api_key)

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to fix common truncation errors in JSON.
        """
        json_str = json_str.strip()

        # 1. Check if it ends with a valid closer
        if json_str.endswith("}"):
            return json_str

        print("🔧 Attempting to repair truncated JSON...")

        # 2. Remove trailing commas (common cause of errors if we append braces)
        # Remove comma if it's the last non-whitespace char
        if json_str.rstrip()[-1] == ',':
            json_str = json_str.rstrip()[:-1]

        # 3. Balance braces/brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')

        # Append missing closing characters in reverse order of likelihood
        # This is a heuristic: usually lists need closing first, then objects
        while open_brackets > close_brackets:
            json_str += "]"
            close_brackets += 1

        while open_braces > close_braces:
            json_str += "}"
            close_braces += 1

        return json_str

    def _clean_json_text(self, text: str) -> str:
        """
        Extract JSON from Markdown and handle pre-amble/post-amble.
        """
        # Remove Markdown fences
        text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*', '', text)

        # Find first '{'
        start = text.find('{')
        if start == -1:
            return text  # Let it fail in the parser to show the raw text

        # If we can't find the last '}', take everything from start to end of string
        # and let _repair_json handle it.
        end = text.rfind('}')

        if end != -1 and end > start:
            # We found a closing brace, but is it the *actual* last one?
            # If the response was truncated, the last '}' might be inside a string or inner object.
            # We'll take the substring, but pass it to repair just in case.
            candidate = text[start:end + 1]
            return self._repair_json(candidate)

        return self._repair_json(text[start:])

    def extract_formulas(self, query: str, vehicle_type: str) -> List[Dict]:
        print(f"🔍 Retrieving context for: '{query}'...")
        context_text = self.retriever.get_context_string(query, limit=3, vehicle_type=vehicle_type)

        if not context_text:
            print("⚠️  No context found.")
            return []

        # STRICT Prompting to reduce verbosity (helps prevent truncation)
        system_prompt = """You are a Python backend system. 
        Output ONLY raw JSON. No markdown, no explanation, no 'Here is the JSON'.
        Minify your JSON (remove unnecessary whitespace) to save tokens."""

        user_message = f"""
        Extract formulas for "{query}" from this text:
        {context_text[:15000]} 

        Return a JSON object with this EXACT schema:
        {{
            "formulas": [
                {{
                    "name": "string",
                    "expression": "python_math_code",
                    "variables": {{ "var_name": "description" }},
                    "desc": "short_string"
                }}
            ]
        }}
        """
        # Note: I limited context_text to 15000 chars to prevent prompt overload

        try:
            print("🤖 Calling Claude...")
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                temperature=0.0
            )

            raw_content = response.content[0].text

            # --- DEBUG LOGGING ---
            # Save raw output to a file so you can inspect it if it fails
            with open("debug_last_response.txt", "w") as f:
                f.write(raw_content)
            # ---------------------

            clean_content = self._clean_json_text(raw_content)
            data = json.loads(clean_content)

            if "formulas" in data:
                return data["formulas"]
            return [data] if isinstance(data, dict) else data

        except json.JSONDecodeError as e:
            print(f"\n❌ JSON PARSE ERROR: {e}")
            print(f"Check 'debug_last_response.txt' for the raw output.")
            return []
        except Exception as e:
            print(f"❌ Error: {e}")
            return []