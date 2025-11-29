from typing import List, Dict, Optional
from rag.document_store import DocumentStore, SearchResult


class AerospaceRetriever:
    """
    Retrieves relevant aerospace engineering context from the document store.
    Handles query optimization and metadata filtering.
    """

    def __init__(self, doc_store: DocumentStore):
        self.store = doc_store

    def get_relevant_documents(self, query: str, limit: int = 5, vehicle_type: Optional[str] = None) -> List[
        SearchResult]:
        """
        Retrieve documents relevant to the query, optionally filtering by vehicle type.

        Args:
            query: The search query string
            limit: Number of results to return
            vehicle_type: Optional filter (e.g., 'drone', 'rocket')

        Returns:
            List of SearchResult objects
        """
        # If a vehicle type is specified, we can boost results from that domain
        # or strictly filter them depending on DocumentStore implementation.
        # For now, we append the context to the query to improve semantic matching.

        enhanced_query = query
        if vehicle_type:
            enhanced_query = f"{vehicle_type} design: {query}"

        # Perform search
        # Assuming DocumentStore.search(query, n_results, metadata_filter) signature
        # based on typical ChromaDB implementations
        results = self.store.search(
            query=enhanced_query,
            n_results=limit,
            metadata_filter={"category": vehicle_type} if vehicle_type else None
        )

        return results

    def get_context_string(self, query: str, limit: int = 3, vehicle_type: str = None) -> str:
        """
        Helper to get a formatted string of context for LLM prompts.
        Updated to accept vehicle_type for filtering.
        """
        # Pass vehicle_type to get_relevant_documents
        results = self.get_relevant_documents(query, limit=limit, vehicle_type=vehicle_type)

        context_parts = []
        for i, res in enumerate(results, 1):
            source = res.metadata.get('source', 'Unknown')
            context_parts.append(f"Source {i} ({source}):\n{res.content}")

        return "\n\n".join(context_parts)

