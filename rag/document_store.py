import os
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    doc_id: str


class TextChunker:
    """Splits text into manageable chunks for embedding."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Simple character-based splitter.
        For production, consider using recursive splitters from LangChain.
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

        return chunks


class EmbeddingModel:
    """Wrapper for sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Lazy load model to avoid overhead if not used
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print(f"⚙️ Loading embedding model: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()


class DocumentStore:
    """
    Manages document storage and retrieval using ChromaDB.
    """

    def __init__(self, collection_name: str = "aerospace_docs", persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.chunker = TextChunker()
        self.embedder = EmbeddingModel()

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """
        Chunks and adds a document to the vector store.
        """
        if metadata is None:
            metadata = {}

        chunks = self.chunker.split_text(content)

        if not chunks:
            return

        # Generate embeddings for all chunks
        embeddings = self.embedder.encode(chunks)

        # Prepare IDs and metadata for each chunk
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [metadata.copy() for _ in chunks]

        # Add to ChromaDB
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, n_results: int = 5, metadata_filter: Dict = None) -> List[SearchResult]:
        """
        Semantic search for relevant document chunks.
        """
        query_embedding = self.embedder.encode([query])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=metadata_filter
        )

        search_results = []
        if results['documents']:
            # Chroma returns lists of lists
            for i in range(len(results['documents'][0])):
                search_results.append(SearchResult(
                    content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=results['distances'][0][i] if 'distances' in results else 0.0,
                    doc_id=results['ids'][0][i]
                ))

        return search_results

    def get_stats(self) -> Dict:
        return {
            "count": self.collection.count(),
            "collection_name": self.collection_name
        }

    def clear(self):
        """Deletes the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
