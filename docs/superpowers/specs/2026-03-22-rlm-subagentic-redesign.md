# RLM Sub-Agentic Redesign: RAG-to-Ollama Migration

## Problem

The current Aerospace Design Assistant has fundamental architectural issues:

1. **RAG pipeline is broken**: ChromaDB retrieval frequently fails, falling back to a `MockSearchSystem` with hardcoded results. Retrieved data never feeds into actual calculations.
2. **Inconsistent results**: 6 independent Anthropic API calls with uncoordinated prompts and fragile JSON parsing. Each node duplicates client init, model strings, and markdown-stripping logic.
3. **Dead code paths**: `run_simple_workflow` is truncated. LLM search generator and formula extractor are imported but never called. Config system is unused.
4. **No real agency**: Despite using LangGraph, the workflow is entirely linear. The LLM never actually calls tools — a hardcoded switch statement picks tools based on vehicle type.

## Solution

Replace the RAG pipeline with an RLM (Reasoning Language Model) sub-agentic architecture:

- **Deterministic LangGraph orchestration** handles workflow routing (no LLM routing mistakes)
- **Four focused LLM sub-agents** handle tasks that need reasoning
- **The Design Agent uses Ollama tool-calling** to select and invoke calculation tools
- **All LLM calls go through a shared Ollama client** (local inference, no API costs)
- **Existing calculation tools are preserved** as the Design Agent's callable tools

## Architecture

### Graph Structure

```
User Input
    |
    v
[Understand Agent]  -- LLM: classify vehicle, extract requirements
    |
    v
[Parameter Agent]   -- LLM: fill missing params with domain reasoning
    |
    v
[Design Agent]      -- LLM + Tools: call calculation tools, inspect results
    |
    v
[Validate Agent]    -- LLM: review design against requirements
    |
    |--- fail (max 2 retries) --> [Design Agent] with feedback
    |
    v (pass)
[Synthesize]        -- Code-only: format final output
    |
    v
Final Output
```

All edges between agents are deterministic LangGraph edges. The only conditional is the validation retry loop.

### Ollama Integration Layer

#### `llm/client.py`

Single `OllamaClient` class shared by all agents:

- `chat(messages, system_prompt) -> str` -- text response
- `chat_with_tools(messages, system_prompt, tools) -> ToolResponse` -- function-calling response
- Built-in JSON extraction and validation (replaces 6 copies of `_strip_markdown_json`)
- Configurable retry logic (important for 9B models producing occasional malformed output)
- Model and base URL from `config.py`

#### `llm/tools.py`

Converts existing calculation tools into Ollama function-call schemas:

- Reads from `VEHICLE_TOOLS` registry in `tools/__init__.py`
- Generates tool definitions in Ollama's expected format (name, description, parameters as JSON Schema)
- Each vehicle type maps to its tool set so the Design Agent sees only relevant tools

**Schema generation strategy:** The existing tool registry stores parameters as simple `{"altitude": "Altitude (m)"}` dicts — no types. To generate proper JSON Schema:

1. Use `inspect.signature()` on each tool function to extract parameter names, type annotations, and defaults.
2. Map Python types to JSON Schema types (`float` -> `"number"`, `int` -> `"integer"`, `str` -> `"string"`).
3. Parameters without defaults are marked `required`.
4. The registry's string descriptions become the `description` field in the schema.

**Complex parameter types:** Some tools (e.g., `helicopter_tools.calculate_hover_power`) accept dataclass parameters like `RotorResult`. These tools are NOT exposed directly to the LLM. Instead, only the top-level "design" functions are exposed (`size_drone`, `size_aircraft`, `design_helicopter`, `design_rocket`, `design_satellite`, `design_glider`) plus select utility functions that take only primitives (`calculate_hover_thrust`, `calculate_lift`, `tsiolkovsky_delta_v`, etc.). The design functions internally chain the lower-level calls. This keeps the tool interface flat — only `float`, `int`, `str`, and `bool` parameters.

**`ToolResponse` type:**

```python
@dataclass
class ToolCall:
    id: str              # unique call ID
    name: str            # tool function name
    arguments: Dict[str, Any]  # parsed arguments

@dataclass
class ToolResponse:
    message: str                    # LLM's text response (may be empty if tool call)
    tool_calls: List[ToolCall]      # zero or more tool calls requested
    raw_response: Dict[str, Any]    # full Ollama response for debugging
```

#### `config.py` Changes

Replace Anthropic/OpenAI/RAG/Embedding configs with:

```python
@dataclass
class LLMConfig:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:latest"  # verify with `ollama list` before use
    temperature: float = 0.1
    max_retries: int = 2        # client-level retries for malformed LLM output
    max_tool_calls: int = 5     # Design Agent tool-call loop limit
    max_validation_retries: int = 2  # validation -> redesign loop limit
```

Remove: `EmbeddingConfig`, `VectorDBConfig`, `RAGConfig`, `LangSmithConfig`.

### State Redesign

#### Fields Removed
- `search_queries`, `search_results` -- RAG is gone
- `extracted_formulas`, `extracted_data` -- no formula extraction from papers

#### Fields Added
- `agent_messages: List[Dict]` -- tracks what each agent said/did (debugging, and Validate Agent can see Design Agent's reasoning)
- `tool_calls: List[ToolCallRecord]` -- structured log of every tool call with inputs, outputs, success
- `retry_count: int = 0` -- tracks validation loop retries (replaces old `iteration_count` and `max_iterations`, which are both removed)
- `validation_feedback: Optional[str]` -- carries Validate Agent's feedback to Design Agent on retry

`ToolCallRecord` type:
```python
@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    success: bool
    error: Optional[str] = None
```

#### Fields Removed (additional)
- `iteration_count`, `max_iterations` -- replaced by `retry_count` (max retries lives in `LLMConfig.max_validation_retries`)

#### Fields Kept
- `raw_input`, `requirements`, `vehicle_type`, `classification_confidence`, `classification_reasoning`
- `design_output`, `errors`, `warnings`, `metadata`, `intermediate_results`
- `phase` -- simplified to: `UNDERSTANDING`, `PARAMETERIZING`, `DESIGNING`, `VALIDATING`, `SYNTHESIZING`, `COMPLETE`, `ERROR`

#### Enum/String Fix
Keep `use_enum_values = True`. All code works with string values directly. No more scattered `hasattr(x, "value")` checks.

### Agent Specifications

#### Understand Agent (`agents/understand.py`)

**Job:** Classify vehicle type and extract explicitly stated requirements from natural language.

- Single `chat()` call with focused system prompt
- Outputs structured JSON: vehicle type, numeric values, reasoning
- No tools -- pure language understanding
- Low confidence sets a warning but doesn't error out

Replaces: `llm_supervisor.py` + `classify_vehicle()` + `parse_requirements()` (three steps collapsed into one)

#### Parameter Agent (`agents/parameter.py`)

**Job:** Fill every missing parameter needed for calculation using domain reasoning.

- Single `chat()` call
- System prompt includes vehicle-type-specific guidelines (e.g., "a 4kg drone cruises at 60-80 km/h, not 200 km/h")
- Receives what's specified, fills gaps with reasoning
- Outputs complete parameter set + reasoning in metadata

Replaces: `llm_parameter_completer.py`

#### Design Agent (`agents/design.py`)

**Job:** Core agentic node. Calls calculation tools via Ollama function-calling to produce a design.

- `chat_with_tools()` call with vehicle-type-specific tool schemas
- LLM decides which tools to call and with what parameters
- Tool results return to the LLM; it can inspect and call additional tools
- On retry, receives `validation_feedback` and adjusts before re-calling
- All tool calls logged to `state.tool_calls`

**Tool-call loop pseudocode:**

```
messages = [initial prompt with requirements + available tools]
if validation_feedback:
    messages.append(feedback context)

for i in range(max_tool_calls):
    response = client.chat_with_tools(messages, system_prompt, tools)

    if no tool_calls in response:
        # LLM produced a text-only answer — done
        extract design summary from response.message
        break

    for tool_call in response.tool_calls:
        # Validate parameter types against schema before execution
        validated_args = validate_tool_args(tool_call.name, tool_call.arguments)
        if validation error:
            tool_result = {"error": "Expected float for payload_kg, got string"}
        else:
            tool_result = execute_tool(tool_call.name, validated_args)

        # Log to state
        state.tool_calls.append(ToolCallRecord(...))

        # Append result as tool-role message so LLM sees it
        messages.append({"role": "tool", "content": json.dumps(tool_result), "tool_call_id": tool_call.id})

# After loop: populate state.intermediate_results["design"] from tool results
```

**Tool-call parameter validation:** Before executing any tool, the Design Agent validates parameter types against the schema from `llm/tools.py`. If a 9B model passes `"2.0"` (string) instead of `2.0` (float), the validator coerces it. If a parameter is completely wrong (e.g., a hallucinated name), the error is returned to the LLM as a tool result so it can self-correct.

**State output:** The Design Agent writes the final design data to `state.intermediate_results["design"]` (same key the Synthesize step reads from). Raw tool call logs go to `state.tool_calls` for the Validate Agent to inspect.

Replaces: the hardcoded `perform_calculations()` switch statement. The LLM chooses tools instead of an if/elif chain.

#### Validate Agent (`agents/validate.py`)

**Job:** Review design against original requirements, decide pass/fail.

- Single `chat()` call
- Sees original requirements, tool calls made, and results
- Returns structured JSON: pass/fail, checks, warnings, errors
- On fail: produces natural language feedback string for Design Agent retry

Replaces: `validate_design()` + `llm_validator.py` + `llm_data_validator.py` (three validation mechanisms unified)

#### Synthesize (code-only in `graph/workflow.py`)

Reads from `state.intermediate_results["design"]` (populated by the Design Agent) and formats it into `DesignOutput`. Same role as current `synthesize_output()`, cleaned up. No LLM needed.

**`DesignOutput` field changes** (in `graph/state.py`):
- `citations: List[str]` -- **removed** (no documents to cite with RAG gone)
- `components: List[DesignComponent]` -- **removed** (the `DesignComponent.source_citations` field is vestigial without RAG). Component breakdown is now captured in `specifications` dict directly.
- `confidence_score` -- **kept**, sourced from classification confidence + validation pass/fail
- All other fields kept as-is: `vehicle_type`, `summary`, `specifications`, `performance`, `weight_breakdown`, `validation`

### Error Handling

- **LLM parsing failures**: Shared client retries up to `max_retries`. On total failure, agent returns structured error in state. No silent fallbacks.
- **Tool-call failures**: Design Agent sees error as tool result, can retry with different params. Unrecoverable errors set state error.
- **Validation retry**: Max 2 retries. If still failing, output design with warnings. User sees what passed and what didn't.
- **Ollama not running**: Caught at client level, clear error message immediately.

## File Layout

```
capstone-496/
  agents/
    __init__.py
    understand.py
    parameter.py
    design.py
    validate.py
  llm/
    __init__.py
    client.py
    tools.py
  graph/
    __init__.py
    state.py
    workflow.py
  tools/                  # UNCHANGED
    __init__.py
    common_tools.py
    drone_tools.py
    fixed_wing_tools.py
    helicopter_tools.py
    rocket_tools.py
    satellite_tools.py
    glider_tools.py
  tests/
    test_all.py
  config.py
  main.py
  requirements.txt
```

## Deletions

- `rag/` -- entire directory
- `data/papers/` -- research papers and index (manually curated but no longer needed)
- `nodes/` -- entire directory including `synthesizer.py` (replaced by `agents/`)
- `utils/` -- empty package, unused
- `ingest_data.py`, `download_real_data.py`, `verify_updates.py`
- `graph/nodes.py` -- replaced by focused agent modules
- `graph/studio_graph.py`
- ChromaDB artifacts (`chroma_db/` directory if present)
- `design_report.json`, `langgraph.json`

## Migration Order

1. Build `llm/` layer (testable independently against Ollama)
2. Rewrite `graph/state.py` and `config.py`
3. Build agents: understand -> parameter -> design -> validate
4. Wire up `graph/workflow.py`
5. Simplify `main.py`
6. Rewrite tests
7. Delete old code

## Dependencies

### Add
- `ollama` -- Python client for Ollama API
- `langgraph` -- already present, retained

### Remove
- `chromadb` -- RAG vector store
- `sentence-transformers` -- embedding model
- `PyPDF2`, `pdfplumber` -- PDF parsing for papers
- `langchain-openai`, `langchain-anthropic` -- LLM provider integrations
- `langsmith` -- tracing (no longer needed without Anthropic)
- `anthropic` -- replaced by Ollama

### Keep
- `pydantic` -- state models
- `typer`, `rich` -- CLI
- `python-dotenv` -- env config
- `numpy`, `scipy` -- used by calculation tools

## Constraints

- **Model**: Ollama with Qwen 3.5 9B (or MiniMax M2.7) -- verify exact model tag with `ollama list` before implementation
- **Framework**: LangGraph retained for orchestration
- **Tool-calling reliability**: 9B models need focused prompts, small tool sets per agent, retry logic, and parameter type validation/coercion before tool execution
- **No API dependencies**: Fully local inference
