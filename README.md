# AI-Powered Aerospace Design Assistant

Automated preliminary design generation for any flying vehicle. Describe what you want, and the system produces a complete engineering design with specifications, weight breakdowns, and validation.

## Overview

The assistant uses a LangGraph state machine with four specialized LLM sub-agents running on local Ollama inference. Each agent has a focused task, and all routing between agents is deterministic (code-based, not LLM-decided). The Design Agent uses Ollama's native function-calling to invoke physics-based calculation tools.

```
User Input
    |
    v
[Understand Agent] -- classifies vehicle type, extracts requirements
    |
    v
[Parameter Agent]  -- fills missing parameters via engineering reasoning
    |
    v
[Design Agent]     -- calls calculation tools via Ollama function-calling
    |
    v
[Validate Agent]   -- reviews design against requirements (retries up to 2x)
    |
    v
[Synthesize]       -- formats final output (pure code, no LLM)
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

### Setup

```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama serve
```

### Run

```bash
# Single design
python main.py --design "quadcopter drone with 1kg payload and 20 minute flight time"

# Interactive mode
python main.py --interactive

# Check config
python main.py --status
```

### Model override

```bash
OLLAMA_MODEL=qwen2.5:7b python main.py --design "..."
```

## Supported Vehicle Types

| Vehicle | Example Prompt | Calculation Tools |
|---------|---------------|-------------------|
| **Drone** | "quadcopter with 1kg payload, 20min flight" | `size_drone`, `calculate_hover_thrust`, `calculate_flight_time` |
| **Fixed Wing** | "surveillance aircraft, 5kg payload, 200km range" | `size_aircraft`, `calculate_lift`, `calculate_stall_speed` |
| **Helicopter** | "light helicopter, 400kg payload, 500km range" | `design_helicopter` |
| **Rocket** | "model rocket to 500m altitude, 0.5kg payload" | `design_rocket`, `tsiolkovsky_delta_v` |
| **Satellite** | "earth observation sat, 20kg payload, 600km orbit" | `design_satellite`, `calculate_orbital_velocity`, `calculate_orbital_period` |
| **Glider** | "competition glider, 15m class, cross-country" | `design_glider`, `calculate_glide_performance`, `calculate_best_glide_speed` |

## Example Output

```
$ python main.py --design "quadcopter drone with 1kg payload and 20 minute flight time"

Vehicle Type: DRONE
Confidence: 90%

Drone design for 1.0kg payload.

Specifications:
--------------------------------------------------
  frame_size: 395.13
  num_motors: 4
  motor_kv: 640.00
  prop_diameter: 10
  prop_pitch: 4.50
  battery_cells: 6
  battery_capacity: 8500.00
  total_weight: 2.49
  max_thrust: 81.72
  thrust_to_weight: 3.34
  hover_time: 36.94
  max_speed: 66.83

Weight Breakdown:
  total_weight: 2.49 kg
```

```
$ python main.py --design "model rocket to reach 500 meters altitude, 0.5kg payload"

Vehicle Type: ROCKET
Confidence: 90%

Rocket design for 0.5kg payload.

Specifications:
--------------------------------------------------
  total_delta_v: 133.69
  total_mass: 0.59
  payload_mass: 0.50
  payload_fraction: 0.85
  max_altitude: 500.00
  target_achieved: True
```

## Architecture

### RLM Sub-Agentic Workflow

The system was redesigned from a RAG pipeline to an RLM (Reasoning Language Model) sub-agentic architecture. Instead of retrieving formulas from research papers, the LLM reasons about design parameters using its trained knowledge, then delegates to deterministic calculation tools.

**Why this works better:**
- RAG retrieval was unreliable — frequently fell back to mock data
- 9B local models are better at focused reasoning tasks than multi-step retrieval
- Deterministic tools do the actual math — the LLM only decides *which* tool to call and *what parameters* to pass
- Each agent has a single focused task with a small prompt

### Key Design Decisions

- **Deterministic routing**: All graph edges are code-based. The LLM never decides the next step — only the Understand Agent's output determines the path.
- **`use_enum_values = True`**: All Pydantic enums stored as strings. Route comparisons use `state.phase == "error"` not enum objects.
- **Type coercion**: `validate_tool_args()` handles 9B model quirks like returning `"2.0"` (string) instead of `2.0` (float).
- **Retry loop**: Validate Agent can send designs back to Design Agent up to 2 times with feedback.
- **Merged tool results**: Design Agent merges all successful tool results, so the main sizing tool provides the foundation and utility tools add supplementary data.

### Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph (StateGraph) |
| LLM Inference | Ollama (local) |
| State Management | Pydantic BaseModel |
| Calculation Tools | NumPy, SciPy |
| Models | qwen2.5:7b (fast), qwen3.5:latest (quality) |

### Project Structure

```
main.py                 # CLI entry point (--design, --interactive, --status)
config.py               # Ollama-only configuration with env var overrides
llm/
  client.py             # Shared OllamaClient (chat, chat_json, chat_with_tools)
  tools.py              # Tool schema generation, registry, arg validation
agents/
  understand.py         # Vehicle classification + requirement extraction
  parameter.py          # Missing parameter completion
  design.py             # Agentic tool-calling loop
  validate.py           # Design review against requirements
graph/
  state.py              # DesignState, enums, Pydantic models
  workflow.py           # LangGraph StateGraph, routing, synthesize
tools/
  drone_tools.py        # Multirotor sizing, hover, battery
  fixed_wing_tools.py   # Aircraft sizing, lift, stall speed
  helicopter_tools.py   # Rotor design, hover, forward flight
  rocket_tools.py       # Staging, trajectory, delta-v
  satellite_tools.py    # Orbit, power, thermal
  glider_tools.py       # Glide performance, sink rate
  common_tools.py       # Shared constants and utilities
tests/
  test_all.py           # 28 unit tests (all mocked, no Ollama needed)
```

## Development

### Tests

```bash
python -m pytest tests/test_all.py -v
```

28 tests covering config, OllamaClient, tool schemas, state management, all 4 agents, and workflow integration. All tests use mocked LLM calls.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3.5:latest` | Model for inference |

### Model Recommendations

| Model | Speed | Notes |
|-------|-------|-------|
| `qwen2.5:7b` | ~20s/request | Recommended for development and general use |
| `qwen3.5:latest` | ~2-5min | Better reasoning, heavier on hardware |

## Project History

This project was originally built for MAT496 (capstone course) using:
- Claude Sonnet API for all LLM calls
- ChromaDB + sentence-transformers for RAG over aerospace papers
- 6 independent Anthropic API calls per design

It was redesigned to run entirely locally with:
- Ollama for all inference (no API keys needed)
- 4 focused sub-agents replacing 6 monolithic LLM calls
- Agentic tool-calling replacing hardcoded tool selection
- All RAG/embedding infrastructure removed (~7300 lines deleted)

The calculation tools (physics formulas, sizing algorithms) were preserved — only the LLM orchestration layer changed.
