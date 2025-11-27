# Aerospace Design Assistant - Document Index
## Technical Reference Library

This document catalogs all technical reference documents in the aerospace design assistant knowledge base. Each document contains real engineering formulas, methodologies, and performance data for use in the RAG system.

---

## Summary Statistics

- **Total Documents**: 34
- **Vehicle Types Covered**: 6 (Drones, Fixed-Wing, Helicopters, Rockets, Satellites, Gliders)
- **Content Type**: Technical reference documents with formulas and calculations

---

## Documents by Vehicle Type

### 🚁 Drones / Multicopters (7 documents)

| Document | Description | Key Topics |
|----------|-------------|------------|
| `multirotor_design_principles.txt` | Fundamental physics and design | Thrust, hover power, disk loading, dynamics |
| `uav_power_systems.txt` | Battery and power calculations | LiPo specs, endurance, power budget |
| `control_systems.txt` | Flight dynamics and control | PID control, state estimation, flight modes |
| `motor_propeller_selection.txt` | Motor/prop matching guide | KV rating, thrust coefficients, efficiency |
| `sensors_navigation.txt` | Sensor systems | IMU, GPS, barometer, optical flow |
| `fpv_racing_design.txt` | Racing quadcopter design | High-performance builds, tuning |
| `vtol_hybrid_design.txt` | Hybrid VTOL aircraft | Transition flight, configurations |

### ✈️ Fixed-Wing Aircraft (8 documents)

| Document | Description | Key Topics |
|----------|-------------|------------|
| `aircraft_aerodynamics.txt` | Fundamental aerodynamics | Lift, drag, L/D, performance |
| `propulsion_systems.txt` | Propeller and engine theory | Thrust coefficients, efficiency, SFC |
| `structural_design.txt` | Structural analysis | Loads, materials, wing bending |
| `atmospheric_properties.txt` | ISA and atmosphere | Density, temperature, Reynolds number |
| `stability_control.txt` | Flight stability | Static margin, control derivatives |
| `weight_estimation.txt` | Weight prediction methods | Statistical methods, CG calculation |
| `airfoil_wing_design.txt` | Airfoil selection | NACA profiles, wing geometry |
| `aerodynamic_coefficients.txt` | Coefficient estimation | C_L, C_D, C_m prediction |

### 🚁 Helicopters (5 documents)

| Document | Description | Key Topics |
|----------|-------------|------------|
| `helicopter_aerodynamics.txt` | Rotor aerodynamics | Momentum theory, forward flight |
| `rotor_system_design.txt` | Rotor configuration | Hub types, blade design |
| `rotor_design.txt` | Detailed rotor design | Blade element theory, twist |
| `powerplant_transmission.txt` | Engine and gearbox | Turboshaft, gear reduction |
| `transmission_systems.txt` | Transmission details | Gear types, lubrication |

### 🚀 Rockets (5 documents)

| Document | Description | Key Topics |
|----------|-------------|------------|
| `rocket_propulsion.txt` | Propulsion fundamentals | Tsiolkovsky, thrust, Isp |
| `structures_materials.txt` | Rocket structures | Tanks, body tubes, materials |
| `motor_selection.txt` | Motor selection guide | Classifications, performance |
| `trajectory_guidance.txt` | Flight trajectory | Gravity losses, staging |
| `recovery_systems.txt` | Parachute recovery | Sizing, ejection, descent rate |

### 🛸 Satellites (6 documents)

| Document | Description | Key Topics |
|----------|-------------|------------|
| `orbital_mechanics.txt` | Orbit fundamentals | Kepler, Hohmann, station keeping |
| `power_thermal_systems.txt` | Power and thermal | Solar arrays, batteries, thermal control |
| `attitude_control.txt` | ADCS systems | Sensors, actuators, control laws |
| `communications.txt` | Link budget | Antennas, data rates, margins |
| `cubesat_standards.txt` | CubeSat specifications | Form factors, interfaces |
| `launch_deployment.txt` | Launch operations | Environment, separation, LEOP |

### 🪂 Gliders (3 documents)

| Document | Description | Key Topics |
|----------|-------------|------------|
| `glider_aerodynamics.txt` | Glide performance | L/D, sink rate, polar |
| `thermal_soaring.txt` | Cross-country flying | Thermal centering, MacCready |
| `structural_design.txt` | Glider structures | Composite construction, loads |

---

## Document Format

Each document follows a consistent structure:
1. **Introduction** - Overview of the topic
2. **Fundamental Equations** - Core formulas with variable definitions
3. **Design Parameters** - Typical values and ranges
4. **Calculation Methods** - Step-by-step procedures
5. **Example Data** - Real-world specifications
6. **References** - Source citations

---

## Usage Notes

### For RAG System
- Documents are plain text for easy parsing
- Equations use ASCII notation for compatibility
- Each formula includes variable definitions and units
- Example values provided for validation

### Vehicle Type Tags
Each document is tagged by vehicle type via directory structure:
- `/drones/` - Multicopter UAVs
- `/fixed_wing/` - Airplanes and UAV fixed-wing
- `/helicopters/` - Rotorcraft
- `/rockets/` - Launch vehicles and model rockets
- `/satellites/` - Spacecraft and CubeSats
- `/gliders/` - Sailplanes

### Key Formulas Available

| Category | Formula | Document |
|----------|---------|----------|
| Lift | L = 0.5ρV²SC_L | aircraft_aerodynamics.txt |
| Drag | D = 0.5ρV²SC_D | aircraft_aerodynamics.txt |
| Hover thrust | T = 2ρAv_i² | multirotor_design_principles.txt |
| Battery endurance | t = C/(I×60) | uav_power_systems.txt |
| Delta-V | Δv = I_sp×g₀×ln(m₀/m_f) | rocket_propulsion.txt |
| Orbital velocity | v = √(μ/r) | orbital_mechanics.txt |
| Glide ratio | L/D = C_L/C_D | glider_aerodynamics.txt |
| Disk loading | DL = T/A | helicopter_aerodynamics.txt |

---

## Document Metadata Schema

```json
{
  "document_id": "string",
  "title": "string",
  "vehicle_type": "drone|fixed_wing|helicopter|rocket|satellite|glider",
  "topics": ["string"],
  "formulas_count": "integer",
  "word_count": "integer",
  "created_date": "ISO-8601",
  "version": "1.0"
}
```

---

## Version History

- **v1.0** (2024): Initial document collection - 34 documents covering 6 vehicle types

---

*This document library supports the MAT496 Capstone Project: AI-Powered Aerospace Design Assistant*
