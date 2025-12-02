"""
Semantic Search Interface for Aerospace Design Assistant

This module provides a high-level search interface that combines
semantic similarity search with metadata filtering and result ranking.

Features:
- Multi-query search (expand query with related terms)
- Vehicle type filtering
- Formula extraction from results
- Result aggregation and ranking
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from .document_store import DocumentStore, SearchResult


@dataclass
class FormulaMatch:
    """Represents an extracted formula from search results."""
    formula: str
    context: str
    variables: Dict[str, str]
    source_doc: str
    relevance_score: float


@dataclass 
class SearchResponse:
    """Complete search response with results and metadata."""
    query: str
    vehicle_type: Optional[str]
    results: List[SearchResult]
    formulas: List[FormulaMatch]
    total_results: int
    search_time_ms: float = 0.0


class SemanticSearch:
    """
    High-level semantic search interface for aerospace documents.
    
    Provides:
    - Query expansion for better recall
    - Multi-vehicle type search
    - Formula extraction from results
    - Result re-ranking
    """
    
    # Aerospace domain synonyms for query expansion
    QUERY_EXPANSIONS = {
        "thrust": ["force", "propulsion", "power"],
        "lift": ["aerodynamic force", "wing loading"],
        "drag": ["resistance", "friction", "parasitic"],
        "battery": ["power", "energy", "capacity", "endurance"],
        "motor": ["engine", "propulsion", "actuator"],
        "propeller": ["prop", "rotor", "blade"],
        "weight": ["mass", "loading", "payload"],
        "speed": ["velocity", "airspeed", "cruise"],
        "altitude": ["height", "elevation", "ceiling"],
        "range": ["distance", "endurance", "flight time"],
        "stability": ["control", "trim", "balance"],
        "orbit": ["orbital", "trajectory", "periapsis", "apoapsis"],
        "satellite": ["spacecraft", "cubesat", "LEO", "GEO"],
        "rocket": ["launch vehicle", "propellant", "staging"],
        "helicopter": ["rotorcraft", "rotor", "hover"],
        "glider": ["sailplane", "soaring", "glide ratio"],
        "drone": ["UAV", "multirotor", "quadcopter", "UAS"],
    }
    
    # Vehicle type aliases
    VEHICLE_ALIASES = {
        "drone": ["drones", "uav", "multirotor", "quadcopter", "hexacopter", "octocopter"],
        "fixed_wing": ["airplane", "aircraft", "plane", "fixed-wing"],
        "helicopter": ["helicopters", "rotorcraft", "helo", "chopper"],
        "rocket": ["rockets", "launch vehicle", "missile"],
        "satellite": ["satellites", "spacecraft", "cubesat", "smallsat"],
        "glider": ["gliders", "sailplane", "soaring"],
    }
    
    def __init__(self, document_store: DocumentStore):
        """
        Initialize semantic search with a document store.
        
        Args:
            document_store: Initialized DocumentStore instance
        """
        self.store = document_store
    
    def search(
        self,
        query: str,
        vehicle_type: Optional[str] = None,
        top_k: int = 5,
        expand_query: bool = True,
        extract_formulas: bool = True
    ) -> SearchResponse:
        """
        Perform semantic search with optional query expansion.
        
        Args:
            query: Search query
            vehicle_type: Filter by vehicle type (optional)
            top_k: Number of results to return
            expand_query: Whether to expand query with synonyms
            extract_formulas: Whether to extract formulas from results
            
        Returns:
            SearchResponse with results and metadata
        """
        import time
        start_time = time.time()
        
        # Normalize vehicle type
        vehicle_type = self._normalize_vehicle_type(vehicle_type)
        
        # Expand query if enabled
        queries = [query]
        if expand_query:
            queries.extend(self._expand_query(query))
        
        # Search with all query variants
        all_results = []
        for q in queries[:3]:  # Limit to 3 query variants
            results = self.store.search(
                query=q,
                vehicle_type=vehicle_type,
                top_k=top_k
            )
            all_results.extend(results)
        
        # Deduplicate and re-rank
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rerank_results(unique_results, query)[:top_k]
        
        # Extract formulas if requested
        formulas = []
        if extract_formulas:
            formulas = self._extract_formulas(ranked_results)
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=query,
            vehicle_type=vehicle_type,
            results=ranked_results,
            formulas=formulas,
            total_results=len(ranked_results),
            search_time_ms=search_time
        )
    
    def search_for_design(
        self,
        requirements: Dict[str, Any],
        vehicle_type: str,
        aspects: List[str] = None
    ) -> Dict[str, SearchResponse]:
        """
        Search for information needed for a design task.
        
        Args:
            requirements: Design requirements dict
            vehicle_type: Type of vehicle being designed
            aspects: Design aspects to search for (e.g., ["propulsion", "structure"])
            
        Returns:
            Dict mapping aspect to SearchResponse
        """
        if aspects is None:
            aspects = self._get_default_aspects(vehicle_type)
        
        results = {}
        
        for aspect in aspects:
            # Build aspect-specific query
            query = self._build_aspect_query(aspect, requirements, vehicle_type)
            
            # Search for this aspect
            response = self.search(
                query=query,
                vehicle_type=vehicle_type,
                top_k=3,
                extract_formulas=True
            )
            
            results[aspect] = response
        
        return results
    
    def find_formulas(
        self,
        topic: str,
        vehicle_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[FormulaMatch]:
        """
        Search specifically for formulas related to a topic.
        
        Args:
            topic: Topic to find formulas for
            vehicle_type: Optional vehicle type filter
            top_k: Maximum number of formulas to return
            
        Returns:
            List of FormulaMatch objects
        """
        # Search with formula-focused query
        query = f"{topic} formula equation calculation"
        
        response = self.search(
            query=query,
            vehicle_type=vehicle_type,
            top_k=top_k * 2,  # Get more results to find formulas
            extract_formulas=True
        )
        
        return response.formulas[:top_k]
    
    def _normalize_vehicle_type(self, vehicle_type: Optional[str]) -> Optional[str]:
        """Normalize vehicle type string to standard form."""
        if vehicle_type is None:
            return None
        
        vehicle_type = vehicle_type.lower().strip()
        
        # Check if it's already a standard type
        standard_types = ["drone", "drones", "fixed_wing", "helicopter", "helicopters", 
                         "rocket", "rockets", "satellite", "satellites", "glider", "gliders"]
        if vehicle_type in standard_types:
            # Normalize plurals
            if vehicle_type.endswith('s') and vehicle_type != "drones":
                return vehicle_type[:-1] + 's'
            return vehicle_type
        
        # Check aliases
        for standard, aliases in self.VEHICLE_ALIASES.items():
            if vehicle_type in aliases:
                return standard
        
        return vehicle_type
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with domain synonyms."""
        expanded = []
        query_lower = query.lower()
        
        for term, synonyms in self.QUERY_EXPANSIONS.items():
            if term in query_lower:
                for syn in synonyms[:2]:  # Limit synonyms per term
                    expanded_query = query_lower.replace(term, syn)
                    if expanded_query != query_lower:
                        expanded.append(expanded_query)
        
        return expanded[:3]  # Return max 3 expansions
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on doc_id."""
        seen = set()
        unique = []
        
        for result in results:
            if result.doc_id not in seen:
                seen.add(result.doc_id)
                unique.append(result)
        
        return unique
    
    def _rerank_results(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """Re-rank results based on multiple factors."""
        query_terms = set(query.lower().split())
        
        def score_result(result: SearchResult) -> float:
            score = result.score
            
            # Boost for title match
            title = result.metadata.get('title', '').lower()
            title_match = sum(1 for term in query_terms if term in title)
            score += title_match * 0.1
            
            # Boost for formula presence
            if self._contains_formula(result.content):
                score += 0.05
            
            # Boost for exact phrase match
            if query.lower() in result.content.lower():
                score += 0.1
            
            return score
        
        # Sort by adjusted score
        results.sort(key=score_result, reverse=True)
        return results
    
    def _contains_formula(self, text: str) -> bool:
        """Check if text contains a formula."""
        formula_patterns = [
            r'[A-Za-z_]+\s*=\s*[^=\n]+',  # Variable = expression
            r'\b[A-Z][a-z]?\s*=',  # Physics-style (F =, L =)
            r'[×÷√∫∑∏]',  # Math symbols
            r'\^[0-9]',  # Exponents
        ]
        
        for pattern in formula_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _extract_formulas(self, results: List[SearchResult]) -> List[FormulaMatch]:
        """Extract formulas from search results."""
        formulas = []
        
        # Pattern to match formulas
        formula_pattern = r'([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([^\n]+)'
        
        for result in results:
            matches = re.finditer(formula_pattern, result.content)
            
            for match in matches:
                var_name = match.group(1)
                expression = match.group(2).strip()
                
                # Skip if too short or looks like a definition
                if len(expression) < 3 or expression.startswith('"'):
                    continue
                
                # Get context around the formula
                start = max(0, match.start() - 100)
                end = min(len(result.content), match.end() + 100)
                context = result.content[start:end]
                
                # Try to extract variable definitions
                variables = self._extract_variables(result.content, match.end())
                
                formulas.append(FormulaMatch(
                    formula=f"{var_name} = {expression}",
                    context=context,
                    variables=variables,
                    source_doc=result.metadata.get('title', result.doc_id),
                    relevance_score=result.score
                ))
        
        # Sort by relevance and deduplicate
        formulas.sort(key=lambda f: f.relevance_score, reverse=True)
        
        seen_formulas = set()
        unique_formulas = []
        for f in formulas:
            formula_key = f.formula.split('=')[0].strip()
            if formula_key not in seen_formulas:
                seen_formulas.add(formula_key)
                unique_formulas.append(f)
        
        return unique_formulas
    
    def _extract_variables(self, text: str, formula_end: int) -> Dict[str, str]:
        """Extract variable definitions following a formula."""
        variables = {}
        
        # Look for "Where:" or "-" definitions after formula
        search_text = text[formula_end:formula_end + 500]
        
        # Pattern for variable definitions like "- V = velocity" or "V: velocity"
        var_pattern = r'[-•]\s*([A-Za-z_][A-Za-z_0-9]*)\s*[=:]\s*([^\n-•]+)'
        
        matches = re.finditer(var_pattern, search_text)
        for match in matches:
            var_name = match.group(1).strip()
            var_def = match.group(2).strip()
            if len(var_def) > 2:
                variables[var_name] = var_def
        
        return variables
    
    def _get_default_aspects(self, vehicle_type: str) -> List[str]:
        """Get default design aspects for a vehicle type."""
        aspects_by_type = {
            "drone": ["propulsion", "power system", "frame design", "flight controller"],
            "drones": ["propulsion", "power system", "frame design", "flight controller"],
            "fixed_wing": ["aerodynamics", "propulsion", "structure", "stability"],
            "helicopter": ["rotor design", "power system", "transmission", "control"],
            "helicopters": ["rotor design", "power system", "transmission", "control"],
            "rocket": ["propulsion", "structure", "trajectory", "recovery"],
            "rockets": ["propulsion", "structure", "trajectory", "recovery"],
            "satellite": ["orbit", "power system", "thermal", "attitude control"],
            "satellites": ["orbit", "power system", "thermal", "attitude control"],
            "glider": ["aerodynamics", "structure", "performance"],
            "gliders": ["aerodynamics", "structure", "performance"],
        }
        
        return aspects_by_type.get(vehicle_type, ["design", "performance", "structure"])
    
    def _build_aspect_query(
        self,
        aspect: str,
        requirements: Dict[str, Any],
        vehicle_type: str
    ) -> str:
        """Build a search query for a specific design aspect."""
        # Start with aspect and vehicle type
        query_parts = [vehicle_type, aspect]
        
        # Add relevant requirements
        relevant_keys = {
            "propulsion": ["thrust", "power", "motor", "engine"],
            "power system": ["battery", "endurance", "voltage", "capacity"],
            "aerodynamics": ["lift", "drag", "wing", "airfoil"],
            "structure": ["weight", "material", "frame", "load"],
            "stability": ["control", "trim", "margin"],
            "orbit": ["altitude", "inclination", "period"],
            "rotor design": ["rotor", "blade", "rpm", "diameter"],
        }
        
        if aspect in relevant_keys:
            for key in relevant_keys[aspect]:
                if key in requirements:
                    query_parts.append(f"{key} {requirements[key]}")
        
        return " ".join(query_parts[:5])  # Limit query length


def create_search_interface(
    papers_directory: str = None,
    persist_directory: str = "./chroma_db"
) -> SemanticSearch:
    """
    Create a semantic search interface.
    
    Args:
        papers_directory: Directory containing papers (for initial ingestion)
        persist_directory: ChromaDB storage directory
        
    Returns:
        Initialized SemanticSearch instance
    """
    from .document_store import DocumentStore
    
    store = DocumentStore(persist_directory=persist_directory)
    
    # Ingest documents if directory provided and store is empty
    if papers_directory and store.get_stats()["total_documents"] == 0:
        store.add_documents_from_directory(papers_directory)
    
    return SemanticSearch(store)


if __name__ == "__main__":
    # Test the semantic search
    print("Testing Semantic Search Interface...")
    
    from .document_store import DocumentStore
    
    # Create store with test document
    store = DocumentStore(persist_directory="./test_chroma_db")
    
    test_content = """
    # Quadcopter Thrust Calculation
    
    ## Motor Thrust Formula
    
    The thrust produced by a motor-propeller combination:
    
    T = C_T * rho * n^2 * D^4
    
    Where:
    - T = Thrust force (N)
    - C_T = Thrust coefficient (dimensionless)
    - rho = Air density (kg/m³)
    - n = Propeller speed (rev/s)
    - D = Propeller diameter (m)
    
    ## Hover Thrust Requirement
    
    For stable hover, total thrust must equal weight:
    
    T_hover = m * g
    
    For a quadcopter with 4 motors:
    T_per_motor = (m * g) / 4
    """
    
    store.add_document(
        content=test_content,
        metadata={"vehicle_type": "drone", "title": "Thrust Calculations"},
        doc_id="thrust_test"
    )
    
    # Create search interface
    search = SemanticSearch(store)
    
    # Test search
    response = search.search("motor thrust calculation", vehicle_type="drone")
    print(f"\nSearch: 'motor thrust calculation'")
    print(f"Results: {response.total_results}")
    print(f"Time: {response.search_time_ms:.1f}ms")
    
    if response.formulas:
        print(f"\nExtracted Formulas:")
        for f in response.formulas:
            print(f"  {f.formula}")
            if f.variables:
                print(f"    Variables: {f.variables}")
