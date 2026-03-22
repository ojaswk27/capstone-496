"""
Shared Ollama client for all agents.
Single point of LLM access — no agent touches Ollama directly.
"""
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import ollama

from config import get_config


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResponse:
    message: str
    tool_calls: List[ToolCall]
    raw_response: Dict[str, Any]


class OllamaClient:
    """Shared client wrapping the Ollama Python API."""

    def __init__(self):
        cfg = get_config().llm
        self.model = cfg.ollama_model
        self.base_url = cfg.ollama_base_url
        self.temperature = cfg.temperature
        self.max_retries = cfg.max_retries

    def chat(self, prompt: str, system_prompt: str = "") -> str:
        """Simple text chat. Returns the assistant message content."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]

    def chat_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolResponse:
        """Chat with tool-calling support. Returns ToolResponse."""
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            options={"temperature": self.temperature},
        )

        msg = response["message"]
        text = msg.get("content", "") or ""
        raw_tool_calls = msg.get("tool_calls", [])

        parsed_calls = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            parsed_calls.append(
                ToolCall(
                    id=str(uuid.uuid4())[:8],
                    name=func.get("name", ""),
                    arguments=func.get("arguments", {}),
                )
            )

        return ToolResponse(message=text, tool_calls=parsed_calls, raw_response=response)

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response text. Handles markdown wrapping and surrounding text."""
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip markdown code blocks
        if "```" in text:
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    pass

        # Find first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def chat_json(self, prompt: str, system_prompt: str = "") -> Optional[Dict[str, Any]]:
        """Chat and parse the response as JSON with retries."""
        for attempt in range(self.max_retries + 1):
            try:
                text = self.chat(prompt, system_prompt)
                result = self.extract_json(text)
                if result is not None:
                    return result
                if attempt < self.max_retries:
                    prompt_retry = f"Your previous response was not valid JSON. Please respond with ONLY a JSON object.\n\nOriginal request: {prompt}"
                    text = self.chat(prompt_retry, system_prompt)
                    result = self.extract_json(text)
                    if result is not None:
                        return result
            except Exception:
                if attempt == self.max_retries:
                    raise
        return None
