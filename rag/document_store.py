"""
Document Store for Aerospace Design Assistant

This module handles document ingestion, embedding generation, and storage
using ChromaDB vector database with sentence-transformers embeddings.

Supports:
- Text file ingestion with chunking
- PDF extraction (optional)
- Metadata tagging by vehicle type
- Persistent storage
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

# Vector database
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed. Run: pip install chromadb")

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed.")


@dataclass
class Document:
    """Represents a document chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    doc_id: str = ""
    
    def __post_init__(self):
        if not self.doc_id:
            # Generate ID from content hash
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class SearchResult:
    """Represents a search result with relevance score."""
    content: str
    metadata: Dict[str, Any]
    score: float
    doc_id: str


class TextChunker:
    """Splits documents into chunks for embedding."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        
        # First, try to split by major sections (###)
        sections = re.split(r'\n(?=###)', text)
        
        current_chunk = ""
        for section in sections:
            if len(current_chunk) + len(section) < self.chunk_size:
                current_chunk += section
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If section itself is too long, split further
                if len(section) > self.chunk_size:
                    sub_chunks = self._split_long_section(section)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = section
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        return [c for c in chunks if c.strip()]
    
    def _split_long_section(self, text: str) -> List[str]:
        """Split a long section into smaller chunks."""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap text from previous chunk."""
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            # Get last N characters from previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk
            # Find a good break point
            break_point = overlap_text.rfind('. ')
            if break_point > 0:
                overlap_text = overlap_text[break_point + 2:]
            
            result.append(overlap_text + "\n" + chunks[i])
        
        return result


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                print("Embedding model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print("Sentence transformers not available. Using fallback.")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if self.model is None:
            # Fallback: return simple hash-based vectors (for testing only)
            return [self._fallback_embed(t) for t in texts]
        
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embed([query])[0]
    
    def _fallback_embed(self, text: str) -> List[float]:
        """Simple fallback embedding for testing without model."""
        import hashlib
        # Create a deterministic 384-dim vector from text hash
        hash_bytes = hashlib.sha384(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_bytes]


class DocumentStore:
    """
    Vector database for aerospace documents using ChromaDB.
    
    Supports:
    - Document ingestion with automatic chunking
    - Semantic similarity search
    - Metadata filtering by vehicle type
    - Persistent storage
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "aerospace_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize components
        self.chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.embedder = EmbeddingModel(embedding_model)
        
        # Initialize ChromaDB
        self.client = None
        self.collection = None
        self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        if not CHROMADB_AVAILABLE:
            print("ChromaDB not available. Document store will not persist.")
            return
        
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Aerospace design reference documents"}
            )
            
            print(f"ChromaDB initialized. Collection '{self.collection_name}' has {self.collection.count()} documents.")
            
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            self.client = None
            self.collection = None
    
    def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None
    ) -> List[str]:
        """
        Add a document to the store with automatic chunking.
        
        Args:
            content: Document text content
            metadata: Document metadata (vehicle_type, title, etc.)
            doc_id: Optional document ID prefix
            
        Returns:
            List of chunk IDs added
        """
        # Split into chunks
        chunks = self.chunker.split_text(content)
        
        if not chunks:
            print("Warning: No chunks generated from document")
            return []
        
        # Generate IDs and prepare data
        chunk_ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Create unique ID for each chunk
            chunk_id = f"{doc_id or 'doc'}_{i:04d}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            chunk_ids.append(chunk_id)
            documents.append(chunk)
            
            # Add chunk-specific metadata
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            metadatas.append(chunk_metadata)
        
        # Generate embeddings
        embeddings = self.embedder.embed(documents)
        
        # Add to ChromaDB
        if self.collection is not None:
            try:
                self.collection.add(
                    ids=chunk_ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                print(f"Added {len(chunks)} chunks from document '{metadata.get('title', doc_id)}'")
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")
        
        return chunk_ids
    
    def add_documents_from_directory(
        self,
        directory: str,
        vehicle_type_from_path: bool = True
    ) -> int:
        """
        Add all text documents from a directory structure.
        
        Args:
            directory: Root directory containing documents
            vehicle_type_from_path: Extract vehicle type from subdirectory name
            
        Returns:
            Total number of chunks added
        """
        directory = Path(directory)
        total_chunks = 0
        
        # Find all text files
        for filepath in directory.rglob("*.txt"):
            # Extract metadata from path
            relative_path = filepath.relative_to(directory)
            
            # Get vehicle type from parent directory
            vehicle_type = "general"
            if vehicle_type_from_path and len(relative_path.parts) > 1:
                vehicle_type = relative_path.parts[0]
            
            # Read content
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
            
            # Extract title from first line or filename
            lines = content.strip().split('\n')
            title = lines[0].strip('# ') if lines else filepath.stem
            
            # Create metadata
            metadata = {
                "vehicle_type": vehicle_type,
                "title": title,
                "filename": filepath.name,
                "filepath": str(relative_path),
                "source": "local_document"
            }
            
            # Add document
            doc_id = f"{vehicle_type}_{filepath.stem}"
            chunk_ids = self.add_document(content, metadata, doc_id)
            total_chunks += len(chunk_ids)
        
        print(f"\nTotal: Added {total_chunks} chunks from {directory}")
        return total_chunks
    
    def search(
        self,
        query: str,
        vehicle_type: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query text
            vehicle_type: Optional filter by vehicle type
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult objects
        """
        if self.collection is None:
            print("Warning: No collection available for search")
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Build where filter
        where_filter = None
        if vehicle_type:
            where_filter = {"vehicle_type": vehicle_type}
        
        # Search ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"Search error: {e}")
            return []
        
        # Convert to SearchResult objects
        search_results = []
        
        if results and results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
            distances = results['distances'][0] if results['distances'] else [0] * len(documents)
            ids = results['ids'][0] if results['ids'] else [''] * len(documents)
            
            for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids):
                # Convert distance to similarity score (ChromaDB uses L2 distance)
                score = 1.0 / (1.0 + dist)
                
                if score >= min_score:
                    search_results.append(SearchResult(
                        content=doc,
                        metadata=meta,
                        score=score,
                        doc_id=doc_id
                    ))
        
        return search_results
    
    def hybrid_search(
        self,
        query: str,
        vehicle_types: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Perform hybrid search across multiple vehicle types.
        
        Args:
            query: Search query
            vehicle_types: List of vehicle types to search (None for all)
            top_k: Results per vehicle type
            
        Returns:
            Combined and re-ranked results
        """
        all_results = []
        
        if vehicle_types:
            # Search each vehicle type separately
            for vtype in vehicle_types:
                results = self.search(query, vehicle_type=vtype, top_k=top_k)
                all_results.extend(results)
        else:
            # Search all documents
            all_results = self.search(query, top_k=top_k * 2)
        
        # Sort by score and deduplicate
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Remove duplicates (same doc_id)
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.doc_id not in seen_ids:
                seen_ids.add(result.doc_id)
                unique_results.append(result)
        
        return unique_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the document store."""
        stats = {
            "total_documents": 0,
            "vehicle_types": {},
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }
        
        if self.collection is not None:
            stats["total_documents"] = self.collection.count()
            
            # Count by vehicle type (sample-based estimate)
            try:
                sample = self.collection.get(limit=1000, include=["metadatas"])
                if sample and sample['metadatas']:
                    for meta in sample['metadatas']:
                        vtype = meta.get('vehicle_type', 'unknown')
                        stats["vehicle_types"][vtype] = stats["vehicle_types"].get(vtype, 0) + 1
            except:
                pass
        
        return stats
    
    def clear(self):
        """Clear all documents from the collection."""
        if self.client is not None:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Aerospace design reference documents"}
                )
                print(f"Collection '{self.collection_name}' cleared.")
            except Exception as e:
                print(f"Error clearing collection: {e}")


# Convenience function for quick setup
def create_document_store(
    papers_directory: str = None,
    persist_directory: str = "./chroma_db"
) -> DocumentStore:
    """
    Create and populate a document store.
    
    Args:
        papers_directory: Directory containing papers (optional)
        persist_directory: Where to store the database
        
    Returns:
        Initialized DocumentStore
    """
    store = DocumentStore(persist_directory=persist_directory)
    
    if papers_directory and Path(papers_directory).exists():
        store.add_documents_from_directory(papers_directory)
    
    return store


if __name__ == "__main__":
    # Test the document store
    print("Testing Document Store...")
    
    # Create store
    store = DocumentStore(persist_directory="./test_chroma_db")
    
    # Add a test document
    test_content = """
    # Test Aerospace Document
    
    ## Introduction
    This is a test document about aircraft design.
    
    ## Lift Equation
    The lift force is calculated as:
    L = 0.5 * rho * V^2 * S * C_L
    
    Where:
    - L = Lift force (N)
    - rho = Air density (kg/m³)
    - V = Velocity (m/s)
    - S = Wing area (m²)
    - C_L = Lift coefficient
    
    ## Drag Equation
    D = 0.5 * rho * V^2 * S * C_D
    """
    
    store.add_document(
        content=test_content,
        metadata={"vehicle_type": "fixed_wing", "title": "Test Document"},
        doc_id="test_001"
    )
    
    # Test search
    results = store.search("lift force calculation", top_k=3)
    print(f"\nSearch results for 'lift force calculation':")
    for r in results:
        print(f"  Score: {r.score:.3f} | {r.metadata.get('title', 'Unknown')}")
        print(f"  Content: {r.content[:100]}...")
    
    # Print stats
    print(f"\nStore stats: {store.get_stats()}")
