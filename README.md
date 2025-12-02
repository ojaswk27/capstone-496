Template for creating and submitting MAT496 capstone project.

# Overview of MAT496

In this course, we have primarily learned Langgraph. This is helpful tool to build apps which can process unstructured `text`, find information we are looking for, and present the format we choose. Some specific topics we have covered are:

- Prompting
- Structured Output 
- Semantic Search
- Retreaval Augmented Generation (RAG)
- Tool calling LLMs & MCP
- Langgraph: State, Nodes, Graph

We also learned that Langsmith is a nice tool for debugging Langgraph codes.

------

# Capstone Project objective

The first purpose of the capstone project is to give a chance to revise all the major above listed topics. The second purpose of the capstone is to show your creativity. Think about all the problems which you can not have solved earlier, but are not possible to solve with the concepts learned in this course. For example, We can use LLM to analyse all kinds of news: sports news, financial news, political news. Another example, we can use LLMs to build a legal assistant. Pretty much anything which requires lots of reading, can be outsourced to LLMs. Let your imagination run free.


-------------------------

# Project report

## Title: AI-Powered Aerospace Design Assistant: Automated Design Generation for Any Flying Vehicle

## Overview

My project is an AI assistant that designs flying vehicles automatically. You give it requirements like "I need a drone that flies for 30 minutes with 2kg payload" or "design a small rocket to reach 1km altitude", and it figures out what type of vehicle you need, searches through research papers to find the right formulas and design methods, does all the math calculations, and gives you a complete design with component specifications.

It uses LangGraph to manage the whole process: first it figures out what kind of vehicle you're asking for (drone, plane, helicopter, rocket, satellite, etc.), then searches for relevant papers, extracts the important formulas, calls calculation tools to do the math, checks if the design works, and outputs everything in a nice structured format with citations to the papers it used.

## Reason for picking up this project

This project covers all the main topics we learned in MAT496:

**Prompting**: I use prompts to tell the LLM how to extract equations from research papers, understand what the user wants, and explain why certain design choices were made. Different prompts work for different types of vehicles.

**Structured Output**: All the design specs come out in a clean JSON format with component details, performance numbers, and cost estimates. I'm using Pydantic models to make sure everything is organized properly.

**Semantic Search**: The system searches through aerospace research papers to find relevant information. It filters by vehicle type (drone, plane, rocket, etc.) so it only looks at papers that actually matter for the design problem.

**RAG (Retrieval Augmented Generation)**: After finding the right papers, it pulls out the specific formulas, data, and design methods from them. This information guides which calculations to run, and everything gets cited in the final output.

**Tool calling & MCP**: I built a bunch of calculation tools for different types of vehicles - drones, planes, rockets, helicopters, satellites. The LLM picks which tools to use based on what it's designing. All the actual math happens in these tools using real aerospace formulas.

**Langgraph (State, Nodes, Graph)**: The whole design process is a LangGraph with different nodes for each step: figuring out vehicle type, understanding requirements, searching papers, extracting formulas, picking tools, running calculations, checking if it works, and putting together the final design. The state keeps track of everything as it moves through the graph.

I picked this project because it's creative - instead of just summarizing information, it actually creates new designs by combining knowledge from multiple sources. Plus, it connects to my drone work and lets me apply course concepts to real aerospace problems.

## Plan


## Plan

I plan to execute these steps to complete my project.

- [✅ DONE] Step 1 involves setting up the project - fork the template repo, install langchain, langgraph, langsmith, and FAISS/ChromaDB, set up API keys.

- [✅ DONE] Step 2 involves collecting research papers - download 30-40 papers from arXiv and NASA on drones, planes, helicopters, rockets, and satellites, organize them by vehicle type. A lot harder than expected, had to use multiple layered RAG apps to fetch, read, filter and arrange bits of data that I will need.

- [✅ DONE] Step 3 involves building semantic search - extract text from PDFs, create embeddings, set up vector database, test if search actually finds relevant papers.

- [✅ DONE] Step 4 involves making the RAG system - write prompts to pull out formulas and data from papers, add citation tracking so we know where info came from, test on some known equations.

- [✅ DONE] Step 5 involves creating all the calculation tools - drone tools (thrust, battery life, propeller size), plane tools (lift, drag, wing area, range), rocket tools (delta-v, staging, burn time), helicopter tools (rotor power), satellite tools (orbit math, power budget), and common tools (weight, balance, stability).

- [✅ DONE] Step 6 involves connecting tools to the LLM - define tool schemas, set up MCP-style registration, make sure the LLM can actually call tools correctly.

- [✅ DONE] Step 7 involves designing the state - define what data the graph needs to track (requirements, vehicle type, search results, formulas, calculations, final design), create Pydantic models, add vehicle classification logic.

- [✅ DONE] Step 8 involves building all the LangGraph nodes - Vehicle Classifier (what type of vehicle?), Requirement Parser (understand user input), Search Agent (find relevant papers), Extraction Agent (pull out formulas using RAG), Tool Selector (pick the right calculation tools), Calculation Agent (run the math), Validator (does it meet requirements?), Synthesizer (make final design output)

- [✅ DONE] Step 9 involves connecting the graph - add edges between nodes, add routing for different vehicle types, add logic to loop back if validation fails, test the whole flow.

- [✅ DONE] Step 10 involves making nice output - define JSON schema for results, create templates for design reports, add tables/charts, include citations.

- [✅ DONE] Step 11 involves testing everything - test with surveillance drone (60min flight, 2kg payload), small plane (2 people, 500km range), model rocket (1km altitude), LEO satellite (100kg, 400km orbit), VTOL aircraft, check if answers make sense. They match my own calculations for these given parameters.

- [✅ DONE] Step 12 involves debugging with Langsmith - set up tracing, find slow parts, improve prompts, fix tool calling issues

- [✅ DONE] Step 13 involves writing documentation - README with examples, explain what each node does, add code comments, show example outputs

- [✅ DONE] Step 14 involves final prep - make demo video, finish conclusion, change all TODOs to DONE as I complete them, make sure commits are spread across multiple days

---

## Major Enhancements & Advanced Features

During development, several critical issues were discovered and resolved through intelligent LLM-powered solutions:

### 1. **LLM Supervisor Node** ✨
**Problem**: Initial keyword-based vehicle classification was unreliable and couldn't handle ambiguous requests.

**Solution**: Created `llm_supervisor.py` that uses Claude Sonnet 4.5 to intelligently classify vehicle types and extract all requirements from natural language input. The LLM returns structured JSON with vehicle category, payload, endurance, range, speed, and reasoning.

**Example**: 
- Input: "make a fixed wing drone with 4 hours flight time and 4kg payload"
- LLM Output: Classified as "fixed_wing", extracted 4.0kg payload, 4.0h endurance, and explained reasoning

### 2. **Intelligent Parameter Completion** 🤖
**Problem**: Users rarely specify ALL required parameters. Using hardcoded defaults (e.g., 200 km/h for all aircraft) led to absurd designs - a 4kg drone sizing at 2000kg MTOW because it used manned aircraft defaults.

**Solution**: Created `llm_parameter_completer.py` that uses LLM reasoning to intelligently fill missing parameters based on:
- Vehicle type and scale (small UAV vs manned aircraft)
- Specified constraints (4kg payload suggests small UAV, not jet)
- Engineering relationships (range = endurance × speed)
- Real-world examples of similar vehicles

**Example**:

User specifies: "drone with 2kg payload, 40 min flight"
LLM completes:
  - Speed: 60 km/h (appropriate for small UAV, not 200 km/h)
  - Range: 40 km (calculated from endurance)
  - Altitude: 500m (typical for this class)
  - Propulsion: electric
  - Reasoning: "Based on payload and endurance, this matches tactical UAV class..."


### 3. **LLM Data Validation Layer** 🔍
**Problem**: RAG system retrieved formulas from research papers about manned aircraft and applied them directly to small drones without scale adjustments.

**Solution**: Created `llm_data_validator.py` that reviews all retrieved data in context of the actual design requirements and corrects scale mismatches:
- Detects when manned aircraft cruise speeds (200 km/h) are being applied to 4kg drones
- Identifies weight estimation formulas meant for 1000kg aircraft being used for 15kg UAVs
- Flags high-severity issues as warnings
- Provides corrected parameters with reasoning

**Example Output**:

🔍 LLM Data Validation Results:
   Confidence: 85%
   Issues Found: 2
   🔴 Cruise speed of 200 km/h is inappropriate for 4kg UAV - should be 60-80 km/h
   🟡 Default range of 500km exceeds realistic capability for this endurance
   ✅ Corrected cruise speed to 70 km/h
   ✅ Recalculated range: 4h × 70 km/h = 280 km


### 4. **Context-Aware Aircraft Classification** 🛩️
**Problem**: Fixed-wing tool used one-size-fits-all approach - a 4kg UAV and a Cessna both used the same weight estimation formulas.

**Solution**: Enhanced `fixed_wing_tools.py` with LLM-based sub-classification:
- Categories: uav_small, uav_tactical, light_sport, single_engine_ga, twin_engine_ga, commuter, transport
- Each category has appropriate sizing parameters (MTOW multipliers, empty weight fractions, drag coefficients, aspect ratios)
- LLM analyzes user request + parameters to choose correct category
- Different propulsion modeling (electric vs ICE) based on mission profile

**Impact**: 4kg fixed-wing drone now correctly sized at ~12-18kg MTOW (not 2000kg!)

### 5. **Markdown JSON Response Handling** 📝
**Problem**: Claude API sometimes returns JSON wrapped in markdown code blocks (` ```json ... ``` `), causing parsing failures.

**Solution**: Added `_strip_markdown_json()` helper to all LLM nodes that:
- Detects markdown delimiters
- Strips ` ```json ` and ` ``` ` markers
- Returns clean JSON for parsing

Applied to: `llm_supervisor.py`, `llm_formula_extractor.py`, `llm_search_generator.py`, `llm_validator.py`, `llm_data_validator.py`, `llm_parameter_completer.py`

### 6. **Enum/String Type Safety** 🔒
**Problem**: LangGraph returns state as dict with enum values as strings, but code expected enum objects, causing AttributeError: 'str' object has no attribute 'value'.

**Solution**: 
- Created `dict_to_design_state()` helper in `graph/state.py` to properly convert dicts back to DesignState with enums
- Added safe accessor pattern throughout codebase: 
  ```python
  vtype = state.vehicle_type.value if hasattr(state.vehicle_type, 'value') else str(state.vehicle_type)
  
### 7. **Complete Workflow Integration** 🔄
The final workflow now includes:
User Input → LLM Supervisor (classify vehicle, extract requirements)
          → Parse Requirements (structure the data)
          → LLM Parameter Completer (intelligently fill missing params)
          → Search Documents (RAG retrieval)
          → Extract Formulas (pull equations from papers)
          → LLM Data Validator (verify scale appropriateness)
          → Perform Calculations (use correct tool with correct params)
          → Validate Design (check requirements met)
          → Synthesize Output (generate report)

Key Improvements**:
- Zero hardcoded defaults - all decisions made by LLM reasoning
- Scale-aware calculations - small UAVs vs manned aircraft handled correctly  
- Context validation - retrieved data verified against actual requirements
- Transparent reasoning - LLM explains all parameter choices

Technical Architecture

### LLM Integration
- **Model**: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
- **API**: Anthropic Messages API
- **Key Features**: Structured JSON output, system prompts, low temperature (0) for consistency

### Vector Database
- **Engine**: ChromaDB
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Collections**: Aerospace research papers organized by vehicle type

### State Management
- **Framework**: Pydantic BaseModel
- **Type Safety**: Enum-based vehicle types, strict validation
- **Serialization**: JSON-compatible with enum handling

### Calculation Tools
- Modular design per vehicle type (drones, fixed_wing, helicopters, rockets, satellites, gliders)
- Physics-based formulas from aerospace literature
- Unit-aware calculations with automatic conversions

#### Example Usage:
```python
# Run the design assistant
python main.py --design "fixed wing drone with 4 hours flight time and 4kg payload"
```
**Output**:

🤖 LLM completing missing parameters...
   ✅ Parameters completed:
      Payload: 4.0 kg
      Endurance: 4.0 hours
      Range: 320.0 km
      Cruise Speed: 80.0 km/h
      Altitude: 1000.0 m
      Vehicle-specific: {'propulsion_type': 'electric', 'wingspan_m': 3.2, ...}

   💭 Reasoning: This specification matches a tactical fixed-wing UAV in the 15-25kg 
   MTOW class. With 4kg payload and 4-hour endurance requirements, I selected 80 km/h 
   cruise speed (typical for efficient long-endurance fixed-wing UAVs)...

🤖 LLM classified fixed-wing as: uav_tactical (Medium payload suggests tactical UAV)

============================================================ \
DESIGN RESULT \
============================================================ 

🚀 Vehicle Type: FIXED_WING
📊 Classification Confidence: 90%

Specifications:
   • wing_span_m: 3.65
   • wing_area_m2: 1.11
   • aspect_ratio: 12.00
   • total_weight_kg: 18.2
   • empty_weight_kg: 9.8
   • battery_weight_kg: 4.4
   • power_required_w: 340.5
   • stall_speed_ms: 12.3
   • cruise_speed_ms: 22.2
   • range_km: 320


### Conclusion

This project successfully demonstrates the integration of all MAT496 course concepts into a practical aerospace design system. The key achievement is building an intelligent agent that doesn't just retrieve information but *reasons* about it - understanding scale, context, and engineering relationships to generate appropriate designs.

The most challenging aspect was handling the "impedance mismatch" between research literature (often focused on large aircraft) and user requests (often for small UAVs). The solution - multiple LLM reasoning layers - showcases how modern language models can act as intelligent mediators that understand both domain knowledge and practical constraints.

The system is production-ready for preliminary design work and educational use, with proper citations, transparent reasoning, and validated outputs. Future enhancements could include cost optimization, structural analysis, and CAD model generation.

**Key Takeaway**: LLMs excel not just at information retrieval but at *intelligent adaptation* - taking knowledge from one context and appropriately applying it to another through reasoning, not just pattern matching.