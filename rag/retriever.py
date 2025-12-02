"""
Retriever Module for Aerospace Design Assistant

This module provides high-level retrieval functionality combining:
- Semantic search using embeddings
- Keyword-based filtering
- Vehicle type routing
- Formula extraction from retrieved documents
"""

import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from .document_store import DocumentStore, SearchResult


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with extracted information."""
    content: str
    metadata: Dict[str, Any]
    score: float
    doc_id: str
    formulas: List[str]
    key_values: Dict[str, str]
    

class FormulaExtractor:
    """Extracts mathematical formulas and key values from text."""
    
    # Patterns for common formula formats
    FORMULA_PATTERNS = [
        # Equations with equals sign: L = 0.5 * rho * V^2
        r'([A-Za-z_][A-Za-z0-9_]*\s*=\s*[^.;\n]+)',
        # Greek letter equations: Δv = Isp × g0 × ln(m0/mf)
        r'([ΔδθφψωρσμλαβγΩ][A-Za-z0-9_]*\s*=\s*[^.;\n]+)',
        # Subscript notation: C_L = ...
        r'([A-Z]_[A-Za-z0-9]+\s*=\s*[^.;\n]+)',
    ]
    
    # Patterns for key-value pairs
    VALUE_PATTERNS = [
        # Range patterns: 0.7-0.8, 200-400 m/s
        r'(\d+\.?\d*)\s*[-–to]\s*(\d+\.?\d*)\s*([A-Za-z/%°³²]+)?',
        # Single values with units: 1.225 kg/m³
        r'(\d+\.?\d*)\s+([A-Za-z/%°³²/]+)',
        # Percentages: 15-20%
        r'(\d+\.?\d*)\s*%',
    ]
    
    def extract_formulas(self, text: str) -> List[str]:
        """Extract mathematical formulas from text."""
        formulas = []
        
        for pattern in self.FORMULA_PATTERNS:
            matches = re.findall(pattern, text)
            for match in matches:
                formula = match.strip() if isinstance(match, str) else match[0].strip()
                # Clean up and validate
                if len(formula) > 5 and '=' in formula:
                    formulas.append(formula)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_formulas = []
        for f in formulas:
            if f not in seen:
                seen.add(f)
                unique_formulas.append(f)
        
        return unique_formulas
    
    def extract_key_values(self, text: str) -> Dict[str, str]:
        """Extract key numerical values and parameters."""
        values = {}
        
        # Look for labeled values
        labeled_patterns = [
            r'([A-Za-z_][A-Za-z0-9_\s]+):\s*(\d+\.?\d*(?:\s*[-–]\s*\d+\.?\d*)?)\s*([A-Za-z/%°³²/]+)?',
            r'([A-Za-z_][A-Za-z0-9_\s]+)\s*=\s*(\d+\.?\d*(?:\s*[-–]\s*\d+\.?\d*)?)\s*([A-Za-z/%°³²/]+)?',
        ]
        
        for pattern in labeled_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                key = match[0].strip()
                value = match[1].strip()
                unit = match[2].strip() if len(match) > 2 else ""
                if key and value:
                    values[key] = f"{value} {unit}".strip()
        
        return values


class AerospaceRetriever:
    """
    High-level retriever for aerospace design queries.
    
    Combines semantic search with domain-specific processing:
    - Routes queries to appropriate vehicle type collections
    - Extracts formulas and numerical values
    - Ranks results by relevance to design tasks
    """
    
    # Mapping of keywords to vehicle types
    VEHICLE_KEYWORDS = {
        'drone': ['drone', 'quadcopter', 'multirotor', 'uav', 'unmanned', 'fpv', 'quadrotor', 'hexacopter', 'octocopter'],
        'fixed_wing': ['airplane', 'aircraft', 'plane', 'wing', 'airfoil', 'cessna', 'jet', 'turboprop', 'fixed-wing', 'fixed wing'],
        'helicopter': ['helicopter', 'rotor', 'rotorcraft', 'heli', 'chopper', 'vtol', 'hovering'],
        'rocket': ['rocket', 'launch', 'propellant', 'thrust', 'staging', 'delta-v', 'deltav', 'motor', 'nozzle', 'model rocket'],
        'satellite': ['satellite', 'spacecraft', 'orbit', 'cubesat', 'space', 'leo', 'geo', 'solar panel', 'attitude control'],
        'glider': ['glider', 'sailplane', 'soaring', 'thermal', 'glide ratio', 'sink rate']
    }
    
    # Design task keywords for query enhancement
    DESIGN_TASKS = {
        'sizing': ['size', 'sizing', 'dimension', 'how big', 'calculate size'],
        'power': ['power', 'battery', 'energy', 'endurance', 'flight time', 'range'],
        'propulsion': ['thrust', 'motor', 'propeller', 'engine', 'propulsion'],
        'aerodynamics': ['lift', 'drag', 'aerodynamic', 'airfoil', 'coefficient'],
        'structures': ['weight', 'structure', 'load', 'stress', 'material'],
        'stability': ['stability', 'control', 'cg', 'center of gravity', 'moment'],
        'performance': ['speed', 'climb', 'cruise', 'performance', 'payload']
    }
    
    def __init__(self, document_store: DocumentStore):
        self.store = document_store
        self.formula_extractor = FormulaExtractor()
    
    def detect_vehicle_type(self, query: str) -> Optional[str]:
        """Detect the vehicle type from a query string."""
        query_lower = query.lower()
        
        # Check for vehicle type keywords
        for vehicle_type, keywords in self.VEHICLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return vehicle_type
        
        return None
    
    def detect_design_tasks(self, query: str) -> List[str]:
        """Detect what design tasks are being requested."""
        query_lower = query.lower()
        tasks = []
        
        for task, keywords in self.DESIGN_TASKS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    tasks.append(task)
                    break
        
        return tasks
    
    def enhance_query(self, query: str, vehicle_type: Optional[str] = None) -> str:
        """Enhance query with domain-specific terms."""
        enhanced = query
        
        # Add vehicle type context if detected
        if vehicle_type:
            type_terms = {
                'drone': 'multirotor UAV quadcopter',
                'fixed_wing': 'aircraft airplane wing',
                'helicopter': 'rotorcraft rotor',
                'rocket': 'launch vehicle propulsion',
                'satellite': 'spacecraft orbital',
                'glider': 'sailplane soaring'
            }
            if vehicle_type in type_terms:
                enhanced = f"{query} {type_terms[vehicle_type]}"
        
        return enhanced
    
    def retrieve(
        self,
        query: str,
        vehicle_type: Optional[str] = None,
        top_k: int = 5,
        extract_formulas: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a design query.
        
        Args:
            query: The search query
            vehicle_type: Optional vehicle type filter (auto-detected if None)
            top_k: Number of results to return
            extract_formulas: Whether to extract formulas from results
            
        Returns:
            List of RetrievalResult objects with extracted information
        """
        # Auto-detect vehicle type if not provided
        if vehicle_type is None:
            vehicle_type = self.detect_vehicle_type(query)
        
        # Enhance query
        enhanced_query = self.enhance_query(query, vehicle_type)
        
        # Perform search
        search_results = self.store.search(
            query=enhanced_query,
            vehicle_type=vehicle_type,
            top_k=top_k
        )
        
        # Process results
        retrieval_results = []
        for result in search_results:
            formulas = []
            key_values = {}
            
            if extract_formulas:
                formulas = self.formula_extractor.extract_formulas(result.content)
                key_values = self.formula_extractor.extract_key_values(result.content)
            
            retrieval_results.append(RetrievalResult(
                content=result.content,
                metadata=result.metadata,
                score=result.score,
                doc_id=result.doc_id,
                formulas=formulas,
                key_values=key_values
            ))
        
        return retrieval_results
    
    def retrieve_for_design(
        self,
        requirements: Dict[str, Any],
        vehicle_type: str,
        top_k: int = 10
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve documents for a complete design task.
        
        Args:
            requirements: Design requirements dictionary
            vehicle_type: Type of vehicle being designed
            top_k: Results per category
            
        Returns:
            Dictionary of results organized by design aspect
        """
        results = {}
        
        # Build queries for different design aspects
        queries = self._build_design_queries(requirements, vehicle_type)
        
        for aspect, query in queries.items():
            aspect_results = self.retrieve(
                query=query,
                vehicle_type=vehicle_type,
                top_k=top_k // len(queries) + 1
            )
            results[aspect] = aspect_results
        
        return results
    
    def _build_design_queries(
        self,
        requirements: Dict[str, Any],
        vehicle_type: str
    ) -> Dict[str, str]:
        """Build search queries for different design aspects."""
        queries = {}
        
        # Common queries by vehicle type
        if vehicle_type == 'drone':
            queries['propulsion'] = 'motor propeller thrust power quadcopter sizing'
            queries['power'] = 'battery capacity endurance flight time LiPo'
            queries['frame'] = 'frame design weight structure'
            
            if requirements.get('payload'):
                queries['payload'] = f"payload {requirements['payload']} kg lifting capacity"
            if requirements.get('flight_time'):
                queries['endurance'] = f"flight time endurance {requirements['flight_time']} minutes"
                
        elif vehicle_type == 'fixed_wing':
            queries['aerodynamics'] = 'lift drag coefficient wing design'
            queries['propulsion'] = 'propeller engine thrust power'
            queries['performance'] = 'cruise speed range stall'
            queries['stability'] = 'stability control CG static margin'
            
            if requirements.get('range'):
                queries['range'] = f"range {requirements['range']} km fuel consumption Breguet"
            if requirements.get('payload'):
                queries['payload'] = f"payload {requirements['payload']} kg weight estimation"
                
        elif vehicle_type == 'helicopter':
            queries['rotor'] = 'main rotor disk loading hover power'
            queries['propulsion'] = 'turboshaft engine power transmission'
            queries['performance'] = 'hover ceiling forward flight'
            
        elif vehicle_type == 'rocket':
            queries['propulsion'] = 'motor thrust specific impulse delta-v'
            queries['trajectory'] = 'trajectory altitude staging'
            queries['recovery'] = 'parachute recovery descent'
            queries['structures'] = 'body tube fins materials'
            
        elif vehicle_type == 'satellite':
            queries['orbital'] = 'orbit altitude velocity period'
            queries['power'] = 'solar array battery power budget'
            queries['attitude'] = 'attitude control reaction wheels sensors'
            queries['thermal'] = 'thermal control temperature'
            
        elif vehicle_type == 'glider':
            queries['aerodynamics'] = 'glide ratio lift drag wing'
            queries['performance'] = 'sink rate speed polar thermal'
            queries['structures'] = 'composite construction weight'
        
        return queries
    
    def get_formulas_for_topic(
        self,
        topic: str,
        vehicle_type: Optional[str] = None,
        top_k: int = 5
    ) -> List[str]:
        """
        Get relevant formulas for a specific topic.
        
        Args:
            topic: The topic to find formulas for
            vehicle_type: Optional vehicle type filter
            top_k: Number of documents to search
            
        Returns:
            List of extracted formulas
        """
        results = self.retrieve(
            query=f"{topic} equation formula calculation",
            vehicle_type=vehicle_type,
            top_k=top_k,
            extract_formulas=True
        )
        
        all_formulas = []
        for result in results:
            all_formulas.extend(result.formulas)
        
        # Deduplicate
        seen = set()
        unique_formulas = []
        for f in all_formulas:
            if f not in seen:
                seen.add(f)
                unique_formulas.append(f)
        
        return unique_formulas


def create_retriever(
    papers_directory: str = None,
    persist_directory: str = "./chroma_db"
) -> AerospaceRetriever:
    """
    Create a configured retriever instance.
    
    Args:
        papers_directory: Directory containing papers to index
        persist_directory: ChromaDB storage location
        
    Returns:
        Configured AerospaceRetriever
    """
    from .document_store import create_document_store
    
    store = create_document_store(
        papers_directory=papers_directory,
        persist_directory=persist_directory
    )
    
    return AerospaceRetriever(store)


if __name__ == "__main__":
    # Test the retriever
    print("Testing Aerospace Retriever...")
    
    from .document_store import DocumentStore
    
    # Create a test store
    store = DocumentStore(persist_directory="./test_chroma_db")
    
    # Add test content
    test_doc = """
    # Lift and Drag Calculations
    
    ## Lift Equation
    L = 0.5 * rho * V^2 * S * C_L
    
    Where:
    - rho = 1.225 kg/m³ at sea level
    - C_L = 0.5-1.5 for typical airfoils
    
    ## Drag Equation  
    D = 0.5 * rho * V^2 * S * C_D
    
    Typical drag coefficient: 0.02-0.05 for clean aircraft
    
    ## L/D Ratio
    L/D = C_L / C_D
    
    Typical values: 10-20 for general aviation
    """
    
    store.add_document(
        content=test_doc,
        metadata={"vehicle_type": "fixed_wing", "title": "Lift and Drag"},
        doc_id="test_ld"
    )
    
    # Create retriever
    retriever = AerospaceRetriever(store)
    
    # Test vehicle type detection
    print(f"\nVehicle type for 'drone battery sizing': {retriever.detect_vehicle_type('drone battery sizing')}")
    print(f"Vehicle type for 'rocket delta-v': {retriever.detect_vehicle_type('rocket delta-v')}")
    
    # Test retrieval
    results = retriever.retrieve("calculate lift force", vehicle_type="fixed_wing")
    print(f"\nResults for 'calculate lift force':")
    for r in results:
        print(f"  Score: {r.score:.3f}")
        print(f"  Formulas: {r.formulas}")
        print(f"  Key values: {r.key_values}")
