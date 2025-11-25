"""
Aerospace Calculation Tools
===========================

This package contains specialized calculation tools for different
aerospace vehicle types. Each tool implements real physics formulas
and engineering calculations.

Tool Categories:
- common_tools: Shared utilities (weight, CG, Reynolds number, units)
- drone_tools: Multicopter calculations (thrust, hover, battery)
- fixed_wing_tools: Aircraft calculations (lift, drag, range)
- helicopter_tools: Rotorcraft calculations (disk loading, autorotation)
- rocket_tools: Propulsion calculations (delta-v, staging, burn time)
- satellite_tools: Orbital mechanics (velocity, period, power budget)
- glider_tools: Soaring calculations (glide ratio, sink rate, thermals)
"""

from typing import List

# Will be populated as tool modules are created
__all__ = []
