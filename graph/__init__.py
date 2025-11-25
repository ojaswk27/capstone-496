"""
LangGraph Design Workflow
=========================

This package contains the LangGraph workflow definition that
orchestrates the aerospace design process.

The workflow consists of 8 nodes connected with conditional
edges for iteration and vehicle-specific routing.

Workflow Flow:
    classifier → parser → searcher → extractor → tool_selector
        → calculator → validator → [iterate or finalize] → synthesizer
"""

__all__ = []
