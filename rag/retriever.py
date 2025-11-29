from typing import List, Optional, Dict, Any
from rag.document_store import DocumentStore, SearchResult


class AerospaceRetriever:
    """
    Retrieves relevant aerospace engineering context from the document store.
    Handles query optimization and metadata filtering.
    """

    def __init__(self, doc_store: DocumentStore):
        self.store = doc_store

    def query(self, user_query: str, vehicle_type: Optional[str] = None, k: int = 5) -> List[SearchResult]:
        """
        Main search method. Retrieves documents relevant to the query.

        Args:
            user_query: The search query string (e.g., "momentum theory")
            vehicle_type: Optional filter (e.g., 'drones', 'rockets')
            k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        # Build metadata filter if a specific vehicle type is requested
        filter_dict = {"category": vehicle_type} if vehicle_type else None

        # Perform search using the underlying document store
        results = self.store.search(
            query=user_query,
            n_results=k,
            metadata_filter=filter_dict
        )

        return results

    def format_results(self, results: List[SearchResult]) -> str:
        """
        Helper to get a formatted string of context for LLM prompts.
        """
        if not results:
            return "No relevant technical documents found."

        context_parts = []
        for i, res in enumerate(results, 1):
            source = res.metadata.get('source', 'Unknown')
            context_parts.append(f"Source {i} ({source}):\n{res.content}")

        return "\n\n".join(context_parts)
