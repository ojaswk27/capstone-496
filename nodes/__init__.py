"""
Node Implementations Package
=============================

This package contains individual node implementations that can be
used by the graph workflow or independently.

Individual node files:
- classifier.py: Vehicle classification
- parser.py: Requirement parsing
- search_agent.py: RAG search
- calculator.py: Calculations
- validator.py: Design validation
- synthesizer.py: Output synthesis
- llm_*.py: LLM-powered nodes
"""

# No state imports here - nodes import directly from graph.state
# This prevents circular import issues

__all__ = []
