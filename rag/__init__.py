"""
RAG (Retrieval Augmented Generation) System
============================================

This package implements the RAG pipeline for extracting
design methodologies and formulas from aerospace research papers.

Components:
- document_store: Vector database management with ChromaDB
- retriever: High-level retrieval with formula extraction
- search: Semantic search with query expansion
- ingest: Document ingestion utilities
"""

from .document_store import (
    DocumentStore,
    Document,
    SearchResult,
    TextChunker,
    EmbeddingModel,
    create_document_store,
    CHROMADB_AVAILABLE,
    SENTENCE_TRANSFORMERS_AVAILABLE
)

from .retriever import (
    AerospaceRetriever,
    RetrievalResult,
    FormulaExtractor,
    create_retriever
)

from .search import (
    SemanticSearch,
    SearchResponse,
    FormulaMatch,
    create_search_interface
)

__all__ = [
    # Document Store
    "DocumentStore",
    "Document", 
    "SearchResult",
    "TextChunker",
    "EmbeddingModel",
    "create_document_store",
    
    # Retriever
    "AerospaceRetriever",
    "RetrievalResult",
    "FormulaExtractor",
    "create_retriever",
    
    # Search
    "SemanticSearch",
    "SearchResponse",
    "FormulaMatch",
    "create_search_interface",
    
    # Availability flags
    "CHROMADB_AVAILABLE",
    "SENTENCE_TRANSFORMERS_AVAILABLE"
]
