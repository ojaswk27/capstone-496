"""
Aerospace Design Assistant
==========================

AI-Powered Aerospace Design Assistant using LangGraph.

This package provides an intelligent agent that automatically designs
any type of flying vehicle based on user-specified requirements.

Supported Vehicle Types:
- Drones / Multicopters (UAV)
- Fixed-Wing Aircraft
- Helicopters / Rotorcraft
- Rockets
- Satellites
- Gliders / Sailplanes
- Hybrid VTOL

Course Alignment (MAT496):
- Prompting: LLM instructions for design extraction and synthesis
- Structured Output: Pydantic models for consistent specifications
- Semantic Search: Vector similarity search for research papers
- RAG: Formula and methodology extraction from literature
- Tool Calling: 20+ aerospace engineering calculation tools
- LangGraph: 8-node workflow for design orchestration

Usage:
    from aerospace_design_assistant import DesignAssistant
    
    assistant = DesignAssistant()
    design = assistant.design("surveillance drone, 2kg payload, 60min flight")
    print(design.specifications)
"""

__version__ = "1.0.0"
__author__ = "MAT496 Student"
__course__ = "MAT496 Capstone Project"

__all__ = [
    "__version__",
]
