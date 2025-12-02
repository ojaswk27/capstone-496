"""
Search Agent Node

Performs semantic search over aerospace research papers.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# FIXED: Import from graph.state
sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.state import (
    DesignPhase,
    DesignState,
    SearchResult,
    UserRequirements,
    VehicleType,
)

# ... rest of the file remains the same

# =============================================================================
# Query Generation
# =============================================================================

# Base queries for each vehicle type
BASE_QUERIES = {
    VehicleType.DRONE: [
        "multicopter design methodology",
        "drone motor propeller sizing",
        "UAV battery endurance calculation",
        "quadcopter hover power",
    ],
    VehicleType.FIXED_WING: [
        "aircraft wing design",
        "lift drag calculation aircraft",
        "aircraft performance range endurance",
        "wing loading stall speed",
    ],
    VehicleType.HELICOPTER: [
        "helicopter rotor design",
        "rotorcraft hover power",
        "disk loading helicopter",
        "autorotation performance",
    ],
    VehicleType.ROCKET: [
        "rocket equation delta-v",
        "rocket staging optimization",
        "thrust specific impulse",
        "recovery system parachute sizing",
    ],
    VehicleType.SATELLITE: [
        "orbital mechanics velocity period",
        "satellite power system solar array",
        "spacecraft thermal control",
        "communication link budget",
    ],
    VehicleType.GLIDER: [
        "glider polar curve performance",
        "thermal soaring climb rate",
        "glide ratio L/D optimization",
        "speed to fly MacCready",
    ],
}


def generate_queries(
    vehicle_type: VehicleType, requirements: Optional[UserRequirements]
) -> List[str]:
    """
    Generate search queries based on vehicle type and requirements.

    Args:
        vehicle_type: Type of vehicle
        requirements: Parsed requirements

    Returns:
        List of search queries
    """
    queries = list(BASE_QUERIES.get(vehicle_type, []))

    if not requirements:
        return queries

    # Add requirement-specific queries
    if requirements.payload_kg:
        queries.append(f"{vehicle_type.value} payload capacity design")

    if requirements.range_km:
        queries.append(f"{vehicle_type.value} range calculation fuel")

    if requirements.endurance_hours:
        queries.append(f"{vehicle_type.value} endurance flight time battery")

    if requirements.speed_kmh:
        queries.append(f"{vehicle_type.value} cruise speed aerodynamics")

    if requirements.mission_type:
        queries.append(f"{vehicle_type.value} {requirements.mission_type} mission")

    # Vehicle-specific queries
    if vehicle_type == VehicleType.DRONE:
        if requirements.endurance_hours and requirements.endurance_hours > 0.5:
            queries.append("long endurance drone battery optimization")

    elif vehicle_type == VehicleType.FIXED_WING:
        if requirements.num_passengers:
            queries.append(f"aircraft {requirements.num_passengers} seat design")

    elif vehicle_type == VehicleType.ROCKET:
        if requirements.target_altitude_m:
            if requirements.target_altitude_m > 100000:
                queries.append("suborbital rocket design high altitude")
            else:
                queries.append("model rocket altitude performance")

    elif vehicle_type == VehicleType.SATELLITE:
        if requirements.orbit_altitude_km:
            if requirements.orbit_altitude_km < 2000:
                queries.append("LEO satellite design constraints")
            elif requirements.orbit_altitude_km > 30000:
                queries.append("GEO satellite station keeping")

    return queries[:8]  # Limit to 8 queries


# =============================================================================
# Search Functions
# =============================================================================


class MockSearchSystem:
    """
    Mock search system for when RAG is not available.
    Returns predefined results based on vehicle type.
    """

    MOCK_RESULTS = {
        VehicleType.DRONE: [
            SearchResult(
                content="Hover power for multicopters: P = T^(3/2) / sqrt(2*rho*A). "
                "Figure of merit typically 0.5-0.7 for small multirotors.",
                source="drone_power_systems.txt",
                vehicle_type="drone",
                relevance_score=0.95,
            ),
            SearchResult(
                content="Battery sizing: Capacity = (Power × Time) / (Voltage × DOD × Efficiency). "
                "LiPo energy density ~150-200 Wh/kg.",
                source="drone_battery_sizing.txt",
                vehicle_type="drone",
                relevance_score=0.90,
            ),
        ],
        VehicleType.FIXED_WING: [
            SearchResult(
                content="Lift equation: L = 0.5 × ρ × V² × S × C_L. "
                "For level flight, L = W.",
                source="aircraft_aerodynamics.txt",
                vehicle_type="fixed_wing",
                relevance_score=0.95,
            ),
            SearchResult(
                content="Breguet range equation for propeller aircraft: "
                "R = (η/SFC) × (L/D) × ln(W_i/W_f).",
                source="aircraft_performance.txt",
                vehicle_type="fixed_wing",
                relevance_score=0.90,
            ),
        ],
        VehicleType.HELICOPTER: [
            SearchResult(
                content="Disk loading DL = T/A. Typical values: 200-400 N/m² for light helicopters. "
                "Hover power: P = T × v_i / FM.",
                source="helicopter_aerodynamics.txt",
                vehicle_type="helicopter",
                relevance_score=0.95,
            ),
        ],
        VehicleType.ROCKET: [
            SearchResult(
                content="Tsiolkovsky rocket equation: Δv = Isp × g0 × ln(m0/mf). "
                "For staging, sum Δv of each stage.",
                source="rocket_propulsion.txt",
                vehicle_type="rocket",
                relevance_score=0.95,
            ),
            SearchResult(
                content="Parachute sizing: V_descent = sqrt(2W/(ρ×Cd×A)). "
                "Target 5-7 m/s for safe recovery.",
                source="rocket_recovery.txt",
                vehicle_type="rocket",
                relevance_score=0.85,
            ),
        ],
        VehicleType.SATELLITE: [
            SearchResult(
                content="Orbital velocity: v = sqrt(μ/r). Period: T = 2π × sqrt(a³/μ). "
                "LEO at 400km: v ≈ 7.67 km/s, T ≈ 92 min.",
                source="orbital_mechanics.txt",
                vehicle_type="satellite",
                relevance_score=0.95,
            ),
            SearchResult(
                content="Solar array sizing: P = S × A × η × cos(θ) × degradation. "
                "Include eclipse fraction in power budget.",
                source="satellite_power.txt",
                vehicle_type="satellite",
                relevance_score=0.90,
            ),
        ],
        VehicleType.GLIDER: [
            SearchResult(
                content="Maximum L/D occurs when CD0 = CDi. "
                "L/D_max = 0.5 × sqrt(π × e × AR / CD0).",
                source="glider_performance.txt",
                vehicle_type="glider",
                relevance_score=0.95,
            ),
            SearchResult(
                content="Thermal climb: Net climb = thermal_strength - sink_rate. "
                "Bank angle affects turn radius and sink rate.",
                source="thermal_soaring.txt",
                vehicle_type="glider",
                relevance_score=0.90,
            ),
        ],
    }

    def search(
        self, query: str, vehicle_type: VehicleType, top_k: int = 3
    ) -> List[SearchResult]:
        """Perform mock search."""
        results = self.MOCK_RESULTS.get(vehicle_type, [])
        return results[:top_k]


def create_search_system():
    """
    Create the search system.

    Tries to use the real RAG system if available,
    otherwise falls back to mock results.
    """
    try:
        # Try to import and create real search system
        try:
            from rag.document_store import DocumentStore
            from rag.search import create_search_system as create_rag_search
        except ImportError:
            from ..rag.document_store import DocumentStore
            from ..rag.search import create_search_system as create_rag_search

        papers_dir = os.path.join(os.path.dirname(__file__), "..", "data", "papers")

        if os.path.exists(papers_dir):
            return create_rag_search(papers_dir)
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: Could not initialize RAG system: {e}")

    # Fall back to mock
    return MockSearchSystem()


def perform_search(
    queries: List[str], vehicle_type: VehicleType, search_system: Any = None
) -> List[SearchResult]:
    """
    Perform searches and collect results.

    Args:
        queries: List of search queries
        vehicle_type: Vehicle type for filtering
        search_system: Search system to use

    Returns:
        List of SearchResult
    """
    if search_system is None:
        search_system = MockSearchSystem()

    all_results = []
    seen_sources = set()

    for query in queries:
        try:
            if hasattr(search_system, "search"):
                # Real search system
                results = search_system.search(
                    query,
                    vehicle_type=vehicle_type.value
                    if hasattr(vehicle_type, "value")
                    else str(vehicle_type),
                    top_k=3,
                )

                # Convert to SearchResult if needed
                for r in results:
                    if hasattr(r, "content"):
                        source = getattr(r, "source", "") or r.metadata.get(
                            "title", "Unknown"
                        )
                        if source not in seen_sources:
                            seen_sources.add(source)
                            all_results.append(
                                SearchResult(
                                    content=r.content,
                                    source=source,
                                    vehicle_type=vehicle_type.value,
                                    relevance_score=getattr(r, "score", 0.8),
                                    metadata=getattr(r, "metadata", {}),
                                )
                            )
            else:
                # Mock system
                results = search_system.search(query, vehicle_type)
                for r in results:
                    if r.source not in seen_sources:
                        seen_sources.add(r.source)
                        all_results.append(r)

        except Exception as e:
            print(f"Search error for query '{query}': {e}")

    # Sort by relevance
    all_results.sort(key=lambda x: x.relevance_score, reverse=True)

    return all_results[:10]  # Return top 10


# =============================================================================
# Node Function
# =============================================================================


def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node for searching aerospace information.

    Args:
        state: Current graph state

    Returns:
        State updates
    """
    vehicle_type = state["vehicle_type"]
    requirements = state.get("requirements")

    # Generate queries
    queries = generate_queries(vehicle_type, requirements)

    # Perform search
    search_system = create_search_system()
    results = perform_search(queries, vehicle_type, search_system)

    # Build message
    if results:
        sources = list(set(r.source for r in results))
        message = (
            f"Found {len(results)} relevant results from: {', '.join(sources[:5])}"
        )
    else:
        message = "No search results found, using built-in knowledge"

    return {
        "search_results": results,
        "search_queries": queries,
        "current_phase": DesignPhase.EXTRACTING,
        "messages": [{"role": "system", "content": message}],
    }


if __name__ == "__main__":
    # Test search
    print("=== Search Node Tests ===\n")

    for vtype in [VehicleType.DRONE, VehicleType.ROCKET, VehicleType.SATELLITE]:
        print(f"Vehicle Type: {vtype.value}")

        queries = generate_queries(vtype, None)
        print(f"  Queries: {queries[:3]}")

        results = perform_search(queries, vtype)
        print(f"  Results: {len(results)}")

        if results:
            print(f"  Top result: {results[0].source}")
            print(f"    Content: {results[0].content[:100]}...")
        print()
