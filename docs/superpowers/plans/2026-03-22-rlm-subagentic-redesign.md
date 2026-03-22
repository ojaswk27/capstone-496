# RLM Sub-Agentic Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken RAG pipeline with a deterministic LangGraph workflow powered by four focused Ollama sub-agents with native tool-calling.

**Architecture:** Deterministic LangGraph graph routes through Understand → Parameter → Design → Validate agents. Each agent uses a shared Ollama client. The Design Agent calls existing calculation tools via Ollama function-calling. Validation failures loop back to Design with feedback (max 2 retries).

**Tech Stack:** Python 3.11+, LangGraph, Ollama (Qwen 3.5 9B), Pydantic, existing aerospace calculation tools.

**Spec:** `docs/superpowers/specs/2026-03-22-rlm-subagentic-redesign.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `llm/__init__.py` | Create | Package init |
| `llm/client.py` | Create | Shared Ollama client with retry + JSON extraction |
| `llm/tools.py` | Create | Tool schema generation for Ollama function-calling |
| `config.py` | Rewrite | Simplified config for Ollama |
| `graph/state.py` | Rewrite | New DesignState without RAG fields |
| `agents/__init__.py` | Create | Package init |
| `agents/understand.py` | Create | Vehicle classification + requirement extraction |
| `agents/parameter.py` | Create | Missing parameter completion |
| `agents/design.py` | Create | Tool-calling agent with loop |
| `agents/validate.py` | Create | Design validation + feedback |
| `graph/workflow.py` | Rewrite | LangGraph graph with new agents |
| `graph/__init__.py` | Rewrite | Update exports for new state/workflow |
| `tools/fixed_wing_tools.py` | Modify | Remove embedded Anthropic API call |
| `main.py` | Rewrite | Simplified CLI |
| `tests/test_all.py` | Rewrite | Tests that actually work |
| `requirements.txt` | Rewrite | Updated dependencies |

---

### Task 1: Config Rewrite

**Files:**
- Rewrite: `config.py`
- Test: `tests/test_all.py` (config section)

- [ ] **Step 1: Write config test**

```python
# tests/test_all.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

class TestConfig(unittest.TestCase):
    def test_default_config(self):
        from config import get_config
        cfg = get_config()
        assert cfg.llm.ollama_base_url == "http://localhost:11434"
        assert cfg.llm.temperature == 0.1
        assert cfg.llm.max_retries == 2
        assert cfg.llm.max_tool_calls == 5
        assert cfg.llm.max_validation_retries == 2

    def test_paths_exist(self):
        from config import get_config
        cfg = get_config()
        assert cfg.paths.base_dir.exists()

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestConfig -v`
Expected: FAIL (old config has no `ollama_base_url`)

- [ ] **Step 3: Rewrite config.py**

```python
"""
Configuration for Aerospace Design Assistant.
Simplified for Ollama-based LLM inference.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    """Ollama LLM configuration."""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:latest"
    temperature: float = 0.1
    max_retries: int = 2
    max_tool_calls: int = 5
    max_validation_retries: int = 2

    def __post_init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.ollama_base_url)
        self.ollama_model = os.getenv("OLLAMA_MODEL", self.ollama_model)


@dataclass
class WorkflowConfig:
    """LangGraph workflow configuration."""
    verbose: bool = True
    timeout_seconds: int = 300


@dataclass
class PathConfig:
    """File path configuration."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output")
    examples_dir: Path = field(default_factory=lambda: Path(__file__).parent / "examples")

    def __post_init__(self):
        for dir_path in [self.output_dir, self.examples_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def print_status(self):
        print("\n" + "=" * 50)
        print("Aerospace Design Assistant - Configuration")
        print("=" * 50)
        print(f"  Ollama URL: {self.llm.ollama_base_url}")
        print(f"  Model: {self.llm.ollama_model}")
        print(f"  Temperature: {self.llm.temperature}")
        print("=" * 50 + "\n")


# Global instance
_config = Config()

def get_config() -> Config:
    return _config

def reload_config() -> Config:
    global _config
    _config = Config()
    return _config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestConfig -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_all.py
git commit -m "feat: rewrite config for Ollama"
```

---

### Task 2: Ollama Client

**Files:**
- Create: `llm/__init__.py`
- Create: `llm/client.py`
- Test: `tests/test_all.py` (client section)

- [ ] **Step 1: Create package and write client test**

Create `llm/__init__.py` as empty file.

```python
# Add to tests/test_all.py
from unittest.mock import patch, MagicMock

class TestOllamaClient(unittest.TestCase):
    def test_extract_json_clean(self):
        from llm.client import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        result = client.extract_json('{"vehicle": "drone", "payload_kg": 2.0}')
        assert result == {"vehicle": "drone", "payload_kg": 2.0}

    def test_extract_json_markdown_wrapped(self):
        from llm.client import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        text = '```json\n{"vehicle": "drone"}\n```'
        result = client.extract_json(text)
        assert result == {"vehicle": "drone"}

    def test_extract_json_with_surrounding_text(self):
        from llm.client import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        text = 'Here is the result:\n{"vehicle": "drone"}\nDone.'
        result = client.extract_json(text)
        assert result == {"vehicle": "drone"}

    def test_extract_json_invalid_returns_none(self):
        from llm.client import OllamaClient
        client = OllamaClient.__new__(OllamaClient)
        result = client.extract_json("not json at all")
        assert result is None

    @patch("llm.client.ollama")
    def test_chat_calls_ollama(self, mock_ollama):
        mock_ollama.chat.return_value = {
            "message": {"role": "assistant", "content": "Hello"}
        }
        from llm.client import OllamaClient
        client = OllamaClient()
        result = client.chat("Say hello", system_prompt="Be friendly")
        assert result == "Hello"
        mock_ollama.chat.assert_called_once()

    @patch("llm.client.ollama")
    def test_chat_json_retries_on_bad_output(self, mock_ollama):
        mock_ollama.chat.side_effect = [
            {"message": {"role": "assistant", "content": "not json at all"}},
            {"message": {"role": "assistant", "content": '{"vehicle": "drone"}'}},
        ]
        from llm.client import OllamaClient
        client = OllamaClient()
        result = client.chat_json("classify this")
        assert result == {"vehicle": "drone"}
        assert mock_ollama.chat.call_count == 2

    @patch("llm.client.ollama")
    def test_chat_with_tools_parses_tool_calls(self, mock_ollama):
        mock_ollama.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "size_drone",
                            "arguments": {"payload_kg": 2.0, "flight_time_minutes": 30},
                        }
                    }
                ],
            }
        }
        from llm.client import OllamaClient, ToolResponse
        client = OllamaClient()
        resp = client.chat_with_tools("Design a drone", tools=[])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "size_drone"
        assert resp.tool_calls[0].arguments == {"payload_kg": 2.0, "flight_time_minutes": 30}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestOllamaClient -v`
Expected: FAIL (module doesn't exist)

- [ ] **Step 3: Implement llm/client.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestOllamaClient -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llm/__init__.py llm/client.py tests/test_all.py
git commit -m "feat: add shared Ollama client with JSON extraction and tool-calling"
```

---

### Task 3: Tool Schema Generator

**Files:**
- Create: `llm/tools.py`
- Test: `tests/test_all.py` (tools section)

- [ ] **Step 1: Write tool schema test**

```python
# Add to tests/test_all.py
class TestToolSchemas(unittest.TestCase):
    def test_generate_schema_for_size_drone(self):
        from llm.tools import generate_tool_schema
        from tools.drone_tools import size_drone
        schema = generate_tool_schema(
            size_drone,
            description="Complete drone sizing from requirements"
        )
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "size_drone"
        props = schema["function"]["parameters"]["properties"]
        assert "payload_kg" in props
        assert props["payload_kg"]["type"] == "number"
        assert "flight_time_minutes" in props
        required = schema["function"]["parameters"]["required"]
        assert "payload_kg" in required
        assert "flight_time_minutes" in required
        # Optional params should NOT be in required
        assert "num_motors" not in required

    def test_get_tools_for_vehicle(self):
        from llm.tools import get_tools_for_vehicle_type
        tools = get_tools_for_vehicle_type("drone")
        names = [t["function"]["name"] for t in tools]
        assert "size_drone" in names

    def test_all_vehicle_types_have_tools(self):
        from llm.tools import get_tools_for_vehicle_type
        for vtype in ["drone", "fixed_wing", "helicopter", "rocket", "satellite", "glider"]:
            tools = get_tools_for_vehicle_type(vtype)
            assert len(tools) > 0, f"No tools for {vtype}"

    def test_validate_tool_args_coerces_types(self):
        from llm.tools import validate_tool_args
        # 9B models often return strings instead of floats
        result = validate_tool_args("size_drone", {
            "payload_kg": "2.0",
            "flight_time_minutes": "30"
        })
        assert isinstance(result["payload_kg"], float)
        assert isinstance(result["flight_time_minutes"], float)
        assert result["payload_kg"] == 2.0

    def test_validate_tool_args_rejects_unknown_tool(self):
        from llm.tools import validate_tool_args
        with self.assertRaises(ValueError):
            validate_tool_args("nonexistent_tool", {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestToolSchemas -v`
Expected: FAIL

- [ ] **Step 3: Implement llm/tools.py**

```python
"""
Tool schema generation for Ollama function-calling.
Converts existing calculation tool functions into Ollama-compatible schemas
using function signature introspection.
"""
import inspect
from typing import Any, Callable, Dict, List, Optional, get_type_hints

from tools import (
    size_drone, calculate_hover_thrust, calculate_flight_time,
    size_aircraft, calculate_lift, calculate_stall_speed,
    design_helicopter,
    design_rocket, tsiolkovsky_delta_v,
    design_satellite, calculate_orbital_velocity, calculate_orbital_period,
    design_glider, calculate_glide_performance, calculate_best_glide_speed,
)

# Python type -> JSON Schema type
TYPE_MAP = {
    float: "number",
    int: "integer",
    str: "string",
    bool: "boolean",
}

# Tool registry: vehicle_type -> list of (function, description)
VEHICLE_TOOL_REGISTRY: Dict[str, List[tuple]] = {
    "drone": [
        (size_drone, "Complete drone sizing from payload and flight time requirements"),
        (calculate_hover_thrust, "Calculate thrust required per motor for hover"),
        (calculate_flight_time, "Calculate estimated flight time from battery specs"),
    ],
    "fixed_wing": [
        (size_aircraft, "Complete aircraft sizing from payload, range, and speed"),
        (calculate_lift, "Calculate lift force at given speed, wing area, and lift coefficient"),
        (calculate_stall_speed, "Calculate stall speed for given weight and wing"),
    ],
    "helicopter": [
        (design_helicopter, "Complete helicopter design from payload, range, and speed"),
    ],
    "rocket": [
        (design_rocket, "Complete rocket design for target altitude"),
        (tsiolkovsky_delta_v, "Calculate delta-v using the Tsiolkovsky rocket equation"),
    ],
    "satellite": [
        (design_satellite, "Complete satellite design. Takes payload_power (electrical power in Watts), payload_mass (kg), altitude (meters), mission_years"),
        (calculate_orbital_velocity, "Calculate circular orbital velocity at altitude (meters)"),
        (calculate_orbital_period, "Calculate orbital period at altitude (meters)"),
    ],
    "glider": [
        (design_glider, "Design glider for target glide ratio and class"),
        (calculate_glide_performance, "Calculate glide performance at given conditions"),
        (calculate_best_glide_speed, "Calculate speed for best lift-to-drag ratio"),
    ],
}


def generate_tool_schema(func: Callable, description: str) -> Dict[str, Any]:
    """Generate an Ollama-compatible tool schema from a Python function."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties = {}
    required = []

    for name, param in sig.parameters.items():
        # Get the type annotation
        ann = hints.get(name, str)
        # Handle Optional types
        origin = getattr(ann, "__origin__", None)
        if origin is not None:
            args = getattr(ann, "__args__", ())
            ann = args[0] if args else str

        json_type = TYPE_MAP.get(ann, "string")
        properties[name] = {"type": json_type, "description": name.replace("_", " ")}

        # Parameters without defaults are required
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def get_tools_for_vehicle_type(vehicle_type: str) -> List[Dict[str, Any]]:
    """Get Ollama tool schemas for a specific vehicle type."""
    entries = VEHICLE_TOOL_REGISTRY.get(vehicle_type, [])
    return [generate_tool_schema(func, desc) for func, desc in entries]


def get_tool_function(name: str) -> Optional[Callable]:
    """Look up a tool function by name across all vehicle types."""
    for entries in VEHICLE_TOOL_REGISTRY.values():
        for func, _ in entries:
            if func.__name__ == name:
                return func
    return None


def validate_tool_args(
    func_name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and coerce tool call arguments against the function signature.
    Returns cleaned arguments dict. Raises ValueError for unrecoverable mismatches.
    """
    func = get_tool_function(func_name)
    if func is None:
        raise ValueError(f"Unknown tool: {func_name}")

    sig = inspect.signature(func)
    hints = get_type_hints(func)
    cleaned = {}

    for name, param in sig.parameters.items():
        if name in arguments:
            value = arguments[name]
            expected_type = hints.get(name, str)

            # Handle Optional
            origin = getattr(expected_type, "__origin__", None)
            if origin is not None:
                args = getattr(expected_type, "__args__", ())
                expected_type = args[0] if args else str

            # Coerce types
            try:
                if expected_type == float and not isinstance(value, float):
                    value = float(value)
                elif expected_type == int and not isinstance(value, int):
                    value = int(float(value))
                elif expected_type == bool and not isinstance(value, bool):
                    value = str(value).lower() in ("true", "1", "yes")
                elif expected_type == str and not isinstance(value, str):
                    value = str(value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Parameter '{name}' expected {expected_type.__name__}, "
                    f"got {type(value).__name__}: {value}"
                ) from e

            cleaned[name] = value
        elif param.default is not inspect.Parameter.empty:
            pass  # Will use function's default
        else:
            raise ValueError(f"Missing required parameter: {name}")

    return cleaned
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestToolSchemas -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add llm/tools.py tests/test_all.py
git commit -m "feat: add tool schema generator for Ollama function-calling"
```

---

### Task 4: Strip Anthropic from fixed_wing_tools.py

**Files:**
- Modify: `tools/fixed_wing_tools.py:27-99` (the `_llm_classify_aircraft` function)

- [ ] **Step 1: Run existing fixed-wing test to establish baseline**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestFixedWingTools -v`
Expected: PASS (tests use pure calculation functions, not the LLM classifier)

- [ ] **Step 2: Replace `_llm_classify_aircraft` with rule-based-only version**

Remove the entire Anthropic import and API call from `_llm_classify_aircraft`. Keep only the `_fallback_classification` logic. Rename it to `_classify_aircraft`. Remove the `user_requirements` parameter from `size_aircraft` since it was only used for the LLM call.

In `_llm_classify_aircraft` (around line 27), replace the function body so it just calls `_fallback_classification` directly. The Parameter Agent will set `aircraft_type` correctly in the new architecture, so this internal classifier is just a safety net.

Remove: `import os`, `from anthropic import Anthropic`, and the entire try/except block that calls Claude.

- [ ] **Step 3: Run fixed-wing tests to verify nothing broke**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestFixedWingTools -v`
Expected: PASS

- [ ] **Step 4: Verify no more Anthropic references in tools/**

Run: `grep -r "anthropic\|Anthropic\|ANTHROPIC" tools/`
Expected: No output

- [ ] **Step 5: Commit**

```bash
git add tools/fixed_wing_tools.py
git commit -m "fix: remove embedded Anthropic API call from fixed_wing_tools"
```

---

### Task 5: State Redesign

**Files:**
- Rewrite: `graph/state.py`
- Test: `tests/test_all.py` (state section)

- [ ] **Step 1: Write state tests**

```python
# Add to tests/test_all.py
class TestDesignState(unittest.TestCase):
    def test_create_initial_state(self):
        from graph.state import create_initial_state
        state = create_initial_state("design a drone with 2kg payload")
        assert state.raw_input == "design a drone with 2kg payload"
        assert state.phase == "understanding"
        assert state.vehicle_type == "unknown"
        assert state.retry_count == 0

    def test_state_has_no_rag_fields(self):
        from graph.state import DesignState
        fields = set(DesignState.model_fields.keys())
        assert "search_queries" not in fields
        assert "search_results" not in fields
        assert "extracted_formulas" not in fields
        assert "extracted_data" not in fields

    def test_state_has_new_fields(self):
        from graph.state import DesignState
        fields = set(DesignState.model_fields.keys())
        assert "agent_messages" in fields
        assert "tool_calls" in fields
        assert "retry_count" in fields
        assert "validation_feedback" in fields

    def test_vehicle_type_is_string(self):
        from graph.state import create_initial_state
        state = create_initial_state("drone")
        # With use_enum_values=True, vehicle_type should be a string
        assert isinstance(state.vehicle_type, str)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestDesignState -v`
Expected: FAIL

- [ ] **Step 3: Rewrite graph/state.py**

```python
"""
State schema for Aerospace Design Assistant.
Redesigned for RLM sub-agentic workflow — no RAG fields.
"""
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VehicleType(str, Enum):
    DRONE = "drone"
    FIXED_WING = "fixed_wing"
    HELICOPTER = "helicopter"
    ROCKET = "rocket"
    SATELLITE = "satellite"
    GLIDER = "glider"
    UNKNOWN = "unknown"


class DesignPhase(str, Enum):
    UNDERSTANDING = "understanding"
    PARAMETERIZING = "parameterizing"
    DESIGNING = "designing"
    VALIDATING = "validating"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    ERROR = "error"


class UserRequirements(BaseModel):
    raw_input: str = ""
    payload_kg: Optional[float] = None
    range_km: Optional[float] = None
    endurance_hours: Optional[float] = None
    speed_kmh: Optional[float] = None
    altitude_m: Optional[float] = None
    mission_type: Optional[str] = None
    max_weight_kg: Optional[float] = None
    vehicle_specific: Dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class ValidationResult(BaseModel):
    passed: bool
    checks: Dict[str, bool] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class DesignOutput(BaseModel):
    vehicle_type: str
    summary: str
    specifications: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    weight_breakdown: Dict[str, float] = Field(default_factory=dict)
    validation: Optional[ValidationResult] = None
    confidence_score: float = 0.0


class DesignState(BaseModel):
    """Main state passed between LangGraph nodes."""
    session_id: str = ""
    phase: DesignPhase = DesignPhase.UNDERSTANDING
    raw_input: str = ""
    requirements: Optional[UserRequirements] = None

    # Classification
    vehicle_type: VehicleType = VehicleType.UNKNOWN
    classification_confidence: float = 0.0
    classification_reasoning: str = ""

    # Agent communication
    agent_messages: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)

    # Design results
    intermediate_results: Dict[str, Any] = Field(default_factory=dict)
    validation_result: Optional[ValidationResult] = None
    validation_feedback: Optional[str] = None
    retry_count: int = 0

    # Output
    design_output: Optional[DesignOutput] = None

    # Diagnostics
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


def create_initial_state(user_input: str, session_id: str = "") -> DesignState:
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    return DesignState(
        session_id=session_id,
        raw_input=user_input,
        requirements=UserRequirements(raw_input=user_input),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestDesignState -v`
Expected: PASS

- [ ] **Step 5: Rewrite graph/__init__.py to prevent import breakage**

The old `graph/__init__.py` imports deleted types (`DesignComponent`, `ExtractedFormula`, `SearchResult`) and old node functions from `graph/nodes.py`. Rewrite it now to only export what exists after the state rewrite. The workflow exports will be added in Task 10.

```python
# graph/__init__.py
from .state import DesignState, VehicleType, DesignPhase, create_initial_state
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestDesignState -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add graph/state.py graph/__init__.py tests/test_all.py
git commit -m "feat: rewrite state schema for RLM workflow"
```

---

### Task 6: Understand Agent

**Files:**
- Create: `agents/__init__.py`
- Create: `agents/understand.py`
- Test: `tests/test_all.py` (understand section)

- [ ] **Step 1: Write understand agent test**

```python
# Add to tests/test_all.py
class TestUnderstandAgent(unittest.TestCase):
    @patch("agents.understand.OllamaClient")
    def test_classifies_drone(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = {
            "vehicle_type": "drone",
            "payload_kg": 2.0,
            "endurance_hours": 0.5,
            "range_km": None,
            "speed_kmh": None,
            "altitude_m": None,
            "mission_type": "surveillance",
            "reasoning": "User wants a drone with 2kg payload"
        }
        from agents.understand import understand_agent
        from graph.state import create_initial_state
        state = create_initial_state("surveillance drone, 2kg payload, 30min flight")
        result = understand_agent(state)
        assert result.vehicle_type == "drone"
        assert result.requirements.payload_kg == 2.0
        assert result.requirements.endurance_hours == 0.5
        assert result.phase == "parameterizing"

    @patch("agents.understand.OllamaClient")
    def test_handles_llm_failure(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = None
        from agents.understand import understand_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone 2kg")
        result = understand_agent(state)
        assert result.phase == "error"
        assert len(result.errors) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestUnderstandAgent -v`
Expected: FAIL

- [ ] **Step 3: Implement agents/understand.py**

Create `agents/__init__.py` as empty file.

```python
"""
Understand Agent — classifies vehicle type and extracts requirements.
Replaces: llm_supervisor + classify_vehicle + parse_requirements.
"""
from graph.state import DesignPhase, DesignState, UserRequirements
from llm.client import OllamaClient

SYSTEM_PROMPT = """You are an aerospace engineering assistant. Given a vehicle design request:

1. Identify the vehicle type: drone, fixed_wing, helicopter, rocket, satellite, glider, or unknown
2. Extract any explicitly stated requirements (numbers, specs)

Respond with ONLY a JSON object:
{
  "vehicle_type": "drone",
  "payload_kg": 2.0,
  "endurance_hours": 0.5,
  "range_km": null,
  "speed_kmh": null,
  "altitude_m": null,
  "mission_type": "surveillance",
  "vehicle_specific": {},
  "reasoning": "Brief explanation"
}

Rules:
- Use null for values not mentioned
- Convert minutes to hours (30 min = 0.5 hours)
- "drone" includes quadcopter, multirotor, UAV
- "fixed_wing" includes airplane, aircraft, plane
- For rockets, put target_altitude_m in vehicle_specific
- For satellites, put orbit_altitude_km and mission_years in vehicle_specific"""


def understand_agent(state: DesignState) -> DesignState:
    """LangGraph node: classify vehicle and extract requirements."""
    client = OllamaClient()
    data = client.chat_json(state.raw_input, system_prompt=SYSTEM_PROMPT)

    if data is None:
        state.phase = DesignPhase.ERROR
        state.errors.append("Failed to parse vehicle classification from LLM")
        return state

    # Set vehicle type
    vtype = data.get("vehicle_type", "unknown").lower()
    state.vehicle_type = vtype if vtype in [
        "drone", "fixed_wing", "helicopter", "rocket", "satellite", "glider"
    ] else "unknown"

    state.classification_confidence = 0.9 if state.vehicle_type != "unknown" else 0.3
    state.classification_reasoning = data.get("reasoning", "")

    if state.vehicle_type == "unknown":
        state.warnings.append("Could not confidently classify vehicle type")

    # Populate requirements
    req = state.requirements or UserRequirements(raw_input=state.raw_input)
    for field in ["payload_kg", "endurance_hours", "range_km", "speed_kmh", "altitude_m", "mission_type"]:
        val = data.get(field)
        if val is not None:
            setattr(req, field, val)

    if data.get("vehicle_specific"):
        req.vehicle_specific.update(data["vehicle_specific"])

    state.requirements = req
    state.agent_messages.append({"agent": "understand", "data": data})
    state.phase = DesignPhase.PARAMETERIZING
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestUnderstandAgent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/__init__.py agents/understand.py tests/test_all.py
git commit -m "feat: add Understand Agent for vehicle classification"
```

---

### Task 7: Parameter Agent

**Files:**
- Create: `agents/parameter.py`
- Test: `tests/test_all.py` (parameter section)

- [ ] **Step 1: Write parameter agent test**

```python
# Add to tests/test_all.py
class TestParameterAgent(unittest.TestCase):
    @patch("agents.parameter.OllamaClient")
    def test_fills_missing_params(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = {
            "payload_kg": 2.0,
            "endurance_hours": 0.5,
            "range_km": 30.0,
            "speed_kmh": 60.0,
            "altitude_m": 500.0,
            "mission_type": "surveillance",
            "vehicle_specific": {"num_motors": 4, "application": "photography"},
            "reasoning": "Small surveillance drone"
        }
        from agents.parameter import parameter_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone 2kg payload 30min")
        state.vehicle_type = "drone"
        state.requirements.payload_kg = 2.0
        state.requirements.endurance_hours = 0.5
        result = parameter_agent(state)
        assert result.requirements.range_km == 30.0
        assert result.requirements.speed_kmh == 60.0
        assert result.phase == "designing"

    @patch("agents.parameter.OllamaClient")
    def test_preserves_existing_params(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = {
            "payload_kg": 2.0,
            "endurance_hours": 0.5,
            "range_km": 30.0,
            "speed_kmh": 60.0,
            "altitude_m": 500.0,
            "mission_type": "surveillance",
            "vehicle_specific": {},
            "reasoning": "test"
        }
        from agents.parameter import parameter_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone")
        state.vehicle_type = "drone"
        state.requirements.payload_kg = 5.0  # User specified
        result = parameter_agent(state)
        # LLM returned 2.0 but user specified 5.0 — user value should win
        assert result.requirements.payload_kg == 5.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestParameterAgent -v`
Expected: FAIL

- [ ] **Step 3: Implement agents/parameter.py**

```python
"""
Parameter Agent — fills in missing design parameters using LLM reasoning.
Replaces: llm_parameter_completer.py
"""
import json

from graph.state import DesignPhase, DesignState
from llm.client import OllamaClient

SYSTEM_PROMPT = """You are an aerospace engineer completing a vehicle design specification.
Given a vehicle type and partial requirements, fill in ALL missing parameters.

Base decisions on:
1. Vehicle type and typical performance for that category
2. Already-specified parameters (keep these EXACTLY as given)
3. Engineering relationships (range = endurance x speed)
4. Real-world examples of similar vehicles

Guidelines by vehicle type:
- Small drone (<5kg): 40-80 km/h cruise, 10-60 min endurance
- Fixed-wing UAV (<25kg): 50-90 km/h, 1-6 hour endurance
- Tactical UAV: 80-150 km/h, 4-12 hour endurance
- Manned aircraft: 150-300 km/h, 2-6 hour endurance
- Helicopter: 60-250 km/h depending on size
- Model rocket: altitude in meters, solid motor typical
- Satellite: orbit in km, mission in years

Respond with ONLY a JSON object:
{
  "payload_kg": float,
  "endurance_hours": float,
  "range_km": float,
  "speed_kmh": float,
  "altitude_m": float,
  "mission_type": "string",
  "vehicle_specific": {},
  "reasoning": "Brief explanation"
}"""


def parameter_agent(state: DesignState) -> DesignState:
    """LangGraph node: complete missing parameters."""
    client = OllamaClient()
    req = state.requirements

    # Build context of what's already specified
    specified = {}
    for field in ["payload_kg", "endurance_hours", "range_km", "speed_kmh", "altitude_m", "mission_type"]:
        val = getattr(req, field, None)
        if val is not None:
            specified[field] = val
    if req.vehicle_specific:
        specified["vehicle_specific"] = req.vehicle_specific

    prompt = f"""Vehicle type: {state.vehicle_type}
User request: {req.raw_input}
Currently specified: {json.dumps(specified)}

Complete ALL missing parameters."""

    data = client.chat_json(prompt, system_prompt=SYSTEM_PROMPT)

    if data is None:
        state.warnings.append("Parameter completion failed, proceeding with partial params")
        state.phase = DesignPhase.DESIGNING
        return state

    # Fill missing params — do NOT overwrite user-specified values
    for field in ["payload_kg", "endurance_hours", "range_km", "speed_kmh", "altitude_m", "mission_type"]:
        current = getattr(req, field, None)
        if current is None and data.get(field) is not None:
            setattr(req, field, data[field])

    if data.get("vehicle_specific"):
        for key, val in data["vehicle_specific"].items():
            if key not in req.vehicle_specific:
                req.vehicle_specific[key] = val

    state.requirements = req
    state.metadata["parameter_completion"] = {
        "reasoning": data.get("reasoning", ""),
        "original": specified,
    }
    state.agent_messages.append({"agent": "parameter", "data": data})
    state.phase = DesignPhase.DESIGNING
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestParameterAgent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/parameter.py tests/test_all.py
git commit -m "feat: add Parameter Agent for missing param completion"
```

---

### Task 8: Design Agent

**Files:**
- Create: `agents/design.py`
- Test: `tests/test_all.py` (design section)

- [ ] **Step 1: Write design agent test**

```python
# Add to tests/test_all.py
class TestDesignAgent(unittest.TestCase):
    @patch("agents.design.OllamaClient")
    def test_calls_tool_and_stores_result(self, MockClient):
        from llm.client import ToolCall, ToolResponse

        instance = MockClient.return_value
        # First call: LLM requests a tool call
        instance.chat_with_tools.side_effect = [
            ToolResponse(
                message="",
                tool_calls=[ToolCall(
                    id="abc",
                    name="size_drone",
                    arguments={"payload_kg": 0.5, "flight_time_minutes": 20.0}
                )],
                raw_response={},
            ),
            # Second call: LLM produces final text (no more tool calls)
            ToolResponse(
                message="Design complete.",
                tool_calls=[],
                raw_response={},
            ),
        ]

        from agents.design import design_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone 0.5kg 20min")
        state.vehicle_type = "drone"
        state.requirements.payload_kg = 0.5
        state.requirements.endurance_hours = 20 / 60
        state.phase = "designing"
        result = design_agent(state)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "size_drone"
        assert result.tool_calls[0].success is True
        assert "design" in result.intermediate_results

    @patch("agents.design.OllamaClient")
    def test_handles_bad_tool_name(self, MockClient):
        from llm.client import ToolCall, ToolResponse

        instance = MockClient.return_value
        instance.chat_with_tools.side_effect = [
            ToolResponse(
                message="",
                tool_calls=[ToolCall(id="x", name="nonexistent_tool", arguments={})],
                raw_response={},
            ),
            ToolResponse(message="Could not complete.", tool_calls=[], raw_response={}),
        ]

        from agents.design import design_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone")
        state.vehicle_type = "drone"
        state.phase = "designing"
        result = design_agent(state)
        # Should have logged a failed tool call
        assert any(not tc.success for tc in result.tool_calls)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestDesignAgent -v`
Expected: FAIL

- [ ] **Step 3: Implement agents/design.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestDesignAgent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/design.py tests/test_all.py
git commit -m "feat: add Design Agent with Ollama tool-calling loop"
```

---

### Task 9: Validate Agent

**Files:**
- Create: `agents/validate.py`
- Test: `tests/test_all.py` (validate section)

- [ ] **Step 1: Write validate agent test**

```python
# Add to tests/test_all.py
class TestValidateAgent(unittest.TestCase):
    @patch("agents.validate.OllamaClient")
    def test_passes_good_design(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = {
            "passed": True,
            "checks": {"payload": True, "endurance": True},
            "warnings": [],
            "errors": [],
            "feedback": ""
        }
        from agents.validate import validate_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone 0.5kg 20min")
        state.vehicle_type = "drone"
        state.requirements.payload_kg = 0.5
        state.intermediate_results["design"] = {"total_weight": 1.5, "hover_time": 22}
        result = validate_agent(state)
        assert result.validation_result.passed is True
        assert result.phase == "synthesizing"

    @patch("agents.validate.OllamaClient")
    def test_fails_bad_design(self, MockClient):
        instance = MockClient.return_value
        instance.chat_json.return_value = {
            "passed": False,
            "checks": {"payload": True, "endurance": False},
            "warnings": ["Low flight time"],
            "errors": ["Flight time 10min below target 20min"],
            "feedback": "Increase battery capacity or reduce weight"
        }
        from agents.validate import validate_agent
        from graph.state import create_initial_state
        state = create_initial_state("drone 0.5kg 20min")
        state.vehicle_type = "drone"
        state.intermediate_results["design"] = {"total_weight": 2.0, "hover_time": 10}
        result = validate_agent(state)
        assert result.validation_result.passed is False
        assert result.validation_feedback is not None
        assert result.phase == "validating"  # stays in validating for retry routing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestValidateAgent -v`
Expected: FAIL

- [ ] **Step 3: Implement agents/validate.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestValidateAgent -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/validate.py tests/test_all.py
git commit -m "feat: add Validate Agent for design review"
```

---

### Task 10: LangGraph Workflow

**Files:**
- Rewrite: `graph/workflow.py`
- Rewrite: `graph/__init__.py`
- Test: `tests/test_all.py` (workflow section)

- [ ] **Step 1: Write workflow wiring test**

```python
# Add to tests/test_all.py
class TestWorkflow(unittest.TestCase):
    def test_graph_builds(self):
        from graph.workflow import build_design_graph
        graph = build_design_graph()
        assert graph is not None

    def test_synthesize_formats_output(self):
        from graph.workflow import synthesize_output
        from graph.state import create_initial_state, DesignPhase, ValidationResult
        state = create_initial_state("drone 0.5kg")
        state.vehicle_type = "drone"
        state.requirements.payload_kg = 0.5
        state.classification_confidence = 0.9
        state.intermediate_results["design"] = {
            "total_weight": 1.5,
            "hover_time": 22,
            "frame_size": 350,
        }
        state.validation_result = ValidationResult(passed=True)
        result = synthesize_output(state)
        assert result.design_output is not None
        assert result.design_output.vehicle_type == "drone"
        assert result.phase == "complete"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestWorkflow -v`
Expected: FAIL

- [ ] **Step 3: Rewrite graph/workflow.py**

```python
"""
LangGraph workflow for Aerospace Design Assistant.
Deterministic graph with LLM sub-agents.
"""
from langgraph.graph import END, StateGraph

from agents.design import design_agent
from agents.parameter import parameter_agent
from agents.understand import understand_agent
from agents.validate import validate_agent
from config import get_config
from graph.state import (
    DesignOutput,
    DesignPhase,
    DesignState,
    ValidationResult,
    create_initial_state,
)


def synthesize_output(state: DesignState) -> DesignState:
    """Format intermediate results into final DesignOutput. Pure code, no LLM."""
    state.phase = DesignPhase.SYNTHESIZING
    design = state.intermediate_results.get("design", {})
    req = state.requirements

    if not design:
        state.errors.append("No design data to synthesize")
        state.phase = DesignPhase.ERROR
        return state

    # Build summary
    vtype = state.vehicle_type
    payload_str = f"{req.payload_kg}kg payload" if req.payload_kg else "unspecified payload"
    summary = f"{vtype.replace('_', ' ').title()} design for {payload_str}."

    # Build weight breakdown from design data
    weight_breakdown = {}
    for key in design:
        if "weight" in key or "mass" in key:
            val = design[key]
            if isinstance(val, (int, float)):
                weight_breakdown[key] = float(val)

    validation = state.validation_result or ValidationResult(
        passed=True, warnings=state.warnings, errors=state.errors
    )

    state.design_output = DesignOutput(
        vehicle_type=vtype,
        summary=summary,
        specifications=design,
        performance={},
        weight_breakdown=weight_breakdown,
        validation=validation,
        confidence_score=state.classification_confidence,
    )
    state.phase = DesignPhase.COMPLETE
    return state


def route_after_validation(state: DesignState) -> str:
    """Route after validation: retry design or synthesize. Pure function — no state mutation."""
    cfg = get_config().llm
    if state.validation_result and not state.validation_result.passed:
        # retry_count is incremented by the validate agent, not here
        if state.retry_count <= cfg.max_validation_retries:
            return "design"
    return "synthesize"


def handle_error(state: DesignState) -> DesignState:
    """Terminal error handler."""
    state.phase = DesignPhase.ERROR
    if not state.errors:
        state.errors.append("Unknown error in workflow")
    return state


def route_after_understand(state: DesignState) -> str:
    """Route after understand: error or continue."""
    if state.phase == DesignPhase.ERROR:
        return "error"
    return "parameter"


def build_design_graph():
    """Build the LangGraph workflow."""
    workflow = StateGraph(DesignState)

    # Nodes
    workflow.add_node("understand", understand_agent)
    workflow.add_node("parameter", parameter_agent)
    workflow.add_node("design", design_agent)
    workflow.add_node("validate", validate_agent)
    workflow.add_node("synthesize", synthesize_output)
    workflow.add_node("error", handle_error)

    # Entry
    workflow.set_entry_point("understand")

    # Edges
    workflow.add_conditional_edges(
        "understand",
        route_after_understand,
        {"parameter": "parameter", "error": "error"},
    )
    workflow.add_edge("parameter", "design")
    workflow.add_edge("design", "validate")
    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {"design": "design", "synthesize": "synthesize"},
    )
    workflow.add_edge("synthesize", END)
    workflow.add_edge("error", END)

    return workflow.compile()


class AerospaceDesignWorkflow:
    """High-level interface."""

    def __init__(self):
        self.graph = build_design_graph()

    def run(self, user_input: str, session_id: str = "") -> DesignState:
        initial = create_initial_state(user_input, session_id)
        result = self.graph.invoke(initial)
        if isinstance(result, dict):
            return DesignState(**result)
        return result
```

Update `graph/__init__.py`:

```python
from .state import DesignState, VehicleType, DesignPhase, create_initial_state
from .workflow import AerospaceDesignWorkflow, build_design_graph
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py::TestWorkflow -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add graph/workflow.py graph/__init__.py tests/test_all.py
git commit -m "feat: rewrite LangGraph workflow with agent nodes"
```

---

### Task 11: Simplify main.py and requirements.txt

**Files:**
- Rewrite: `main.py`
- Rewrite: `requirements.txt`

- [ ] **Step 1: Rewrite requirements.txt**

```
# Core
langgraph>=0.2.0
pydantic>=2.0
ollama>=0.4.0
python-dotenv>=1.0

# CLI
typer>=0.9.0
rich>=13.0

# Calculation tools
numpy>=1.24
scipy>=1.10
```

- [ ] **Step 2: Rewrite main.py**

Strip out: batch processing (references undefined `fallback_design`), `run_direct_calculation` (does nothing), all RAG/Anthropic imports. Keep: `--design`, `--interactive`, `--status` modes.

```python
#!/usr/bin/env python3
"""
Aerospace Design Assistant — Main Entry Point.

Usage:
    python main.py --design "surveillance drone, 2kg payload, 60min flight"
    python main.py --interactive
    python main.py --status
"""
import argparse
import sys

from config import get_config


def print_banner():
    print("""
    ========================================================
    AI-Powered Aerospace Design Assistant
    Vehicles: Drones | Aircraft | Helicopters | Rockets
              Satellites | Gliders
    ========================================================
    """)


def process_design(user_input: str):
    """Run the design workflow and print results."""
    print(f"\nProcessing: \"{user_input}\"\n")

    from graph.workflow import AerospaceDesignWorkflow

    workflow = AerospaceDesignWorkflow()
    result = workflow.run(user_input)

    if result.phase == "error":
        print(f"\nDesign failed:")
        for e in result.errors:
            print(f"  - {e}")
        return

    if result.design_output:
        do = result.design_output
        print(f"\nVehicle Type: {do.vehicle_type.upper()}")
        print(f"Confidence: {do.confidence_score:.0%}")
        print(f"\n{do.summary}")
        print(f"\nSpecifications:")
        print("-" * 50)
        for k, v in do.specifications.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            elif isinstance(v, dict):
                continue  # Skip nested dicts in top-level display
            elif isinstance(v, list):
                continue  # Skip lists in top-level display
            else:
                print(f"  {k}: {v}")

        if do.weight_breakdown:
            print(f"\nWeight Breakdown:")
            for k, v in do.weight_breakdown.items():
                print(f"  {k}: {v:.2f} kg")

        if do.validation:
            if do.validation.warnings:
                print(f"\nWarnings:")
                for w in do.validation.warnings:
                    print(f"  - {w}")
            if do.validation.errors:
                print(f"\nErrors:")
                for e in do.validation.errors:
                    print(f"  - {e}")

    print()


def run_interactive():
    print_banner()
    print("Describe the vehicle you want to design. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("Design > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            process_design(user_input)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Aerospace Design Assistant")
    parser.add_argument("--design", "-d", type=str, help="Single design request")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--status", "-s", action="store_true", help="Show config status")
    args = parser.parse_args()

    if args.status:
        print_banner()
        get_config().print_status()
    elif args.design:
        print_banner()
        process_design(args.design)
    elif args.interactive or len(sys.argv) == 1:
        run_interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify main.py syntax**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -c "import ast; ast.parse(open('main.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add main.py requirements.txt
git commit -m "feat: simplify main.py and update dependencies for Ollama"
```

---

### Task 12: Delete Old Code

**Files to delete:**
- `rag/` — entire directory
- `nodes/` — entire directory
- `utils/` — empty package
- `ingest_data.py`
- `download_real_data.py`
- `verify_updates.py`
- `graph/nodes.py`
- `graph/studio_graph.py`
- `design_report.json`
- `langgraph.json`
- `.env.template` (recreate if needed, without Anthropic/OpenAI keys)
- `setup.py` (not needed)

- [ ] **Step 1: Verify no imports reference deleted files**

Run: `cd /home/monarq/Work/academic/capstone-496 && grep -r "from nodes\.\|from rag\.\|from utils\.\|import nodes\.\|import rag\.\|import utils\." agents/ graph/ llm/ main.py config.py`
Expected: No output (new code doesn't import old modules)

- [ ] **Step 2: Delete old files**

```bash
rm -rf rag/ nodes/ utils/
rm -f ingest_data.py download_real_data.py verify_updates.py
rm -f graph/nodes.py graph/studio_graph.py
rm -f design_report.json langgraph.json setup.py
```

- [ ] **Step 3: Run all tests to verify nothing broke**

Run: `cd /home/monarq/Work/academic/capstone-496 && python -m pytest tests/test_all.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove old RAG pipeline, nodes, and dead code"
```

---

### Task 13: Smoke Test with Ollama

This is a manual integration test requiring a running Ollama instance.

- [ ] **Step 1: Verify Ollama is running and model is available**

Run: `ollama list`
Expected: Shows the target model (e.g., `qwen3.5` or similar)

- [ ] **Step 2: Run a design end-to-end**

Run: `cd /home/monarq/Work/academic/capstone-496 && python main.py --design "quadcopter drone with 1kg payload and 20 minute flight time"`

Expected output:
- Vehicle classified as drone
- Parameters completed
- Tool(s) called (size_drone at minimum)
- Design specifications printed
- No Python tracebacks

- [ ] **Step 3: Test another vehicle type**

Run: `cd /home/monarq/Work/academic/capstone-496 && python main.py --design "model rocket to reach 500m altitude with 200g payload"`

Expected: Classified as rocket, calls design_rocket, produces altitude/staging output.

- [ ] **Step 4: Test interactive mode briefly**

Run: `cd /home/monarq/Work/academic/capstone-496 && python main.py --interactive`
Type: "satellite at 400km orbit with 50kg payload"
Type: "quit"

Expected: Produces satellite design, exits cleanly.

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: integration test fixes for Ollama workflow"
```
