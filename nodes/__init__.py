"""
LangGraph Workflow Nodes
========================

This package contains the node functions for the LangGraph
design workflow. Each node performs a specific step in the
aerospace design process.

Nodes:
- Node 0: Vehicle Classifier - Determines vehicle type from requirements
- Node 1: Requirement Parser - Extracts structured parameters
- Node 2: Search Agent - Performs semantic search for relevant papers
- Node 3: Extraction Agent - Extracts formulas using RAG
- Node 4: Tool Selector - Chooses appropriate calculation tools
- Node 5: Calculation Agent - Executes aerospace calculations
- Node 6: Validator - Checks design against requirements
- Node 7: Synthesizer - Generates final design specifications
"""

__all__ = []
