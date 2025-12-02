"""
LangGraph Studio Entry Point
Provides a clean graph definition for visualization and debugging
"""

from graph.workflow import build_design_graph

# Export the compiled graph for LangGraph Studio
graph = build_design_graph()
