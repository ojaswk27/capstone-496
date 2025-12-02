#!/usr/bin/env python3
"""
Document Ingestion Script for Aerospace Design Assistant

This script loads all aerospace reference documents into the ChromaDB
vector database for semantic search.

Usage:
    python -m rag.ingest_documents [--papers-dir PATH] [--db-dir PATH] [--clear]
"""

import os
import sys
import argparse
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.document_store import DocumentStore, CHROMADB_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE


def ingest_documents(
    papers_dir: str,
    db_dir: str = "./chroma_db",
    clear_existing: bool = False
) -> dict:
    """
    Ingest all documents from the papers directory.
    
    Args:
        papers_dir: Directory containing the papers
        db_dir: ChromaDB storage directory
        clear_existing: Whether to clear existing documents first
        
    Returns:
        Statistics about the ingestion
    """
    print("=" * 60)
    print("Aerospace Design Assistant - Document Ingestion")
    print("=" * 60)
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not CHROMADB_AVAILABLE:
        print("ERROR: chromadb is not installed.")
        print("Run: pip install chromadb")
        return {"error": "chromadb not installed"}
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("WARNING: sentence-transformers not installed.")
        print("Using fallback embeddings (not recommended for production)")
    
    # Verify papers directory
    papers_path = Path(papers_dir)
    if not papers_path.exists():
        print(f"ERROR: Papers directory not found: {papers_dir}")
        return {"error": "papers directory not found"}
    
    # Count documents
    txt_files = list(papers_path.rglob("*.txt"))
    print(f"\nFound {len(txt_files)} documents to process")
    
    # Show document breakdown by vehicle type
    by_type = {}
    for f in txt_files:
        parts = f.relative_to(papers_path).parts
        vtype = parts[0] if len(parts) > 1 else "general"
        by_type[vtype] = by_type.get(vtype, 0) + 1
    
    print("\nDocuments by vehicle type:")
    for vtype, count in sorted(by_type.items()):
        print(f"  {vtype}: {count}")
    
    # Initialize document store
    print(f"\nInitializing ChromaDB at: {db_dir}")
    store = DocumentStore(persist_directory=db_dir)
    
    # Clear if requested
    if clear_existing:
        print("Clearing existing documents...")
        store.clear()
    
    # Check current state
    initial_count = store.collection.count() if store.collection else 0
    print(f"Current document chunks in database: {initial_count}")
    
    # Ingest documents
    print("\nIngesting documents...")
    start_time = time.time()
    
    total_chunks = 0
    processed_files = 0
    errors = []
    
    for filepath in txt_files:
        relative_path = filepath.relative_to(papers_path)
        
        # Get vehicle type from parent directory
        parts = relative_path.parts
        vehicle_type = parts[0] if len(parts) > 1 else "general"
        
        try:
            # Read content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from first line
            lines = content.strip().split('\n')
            title = lines[0].strip('# ') if lines else filepath.stem
            
            # Create metadata
            metadata = {
                "vehicle_type": vehicle_type,
                "title": title,
                "filename": filepath.name,
                "filepath": str(relative_path),
                "source": "aerospace_reference"
            }
            
            # Create document ID
            doc_id = f"{vehicle_type}_{filepath.stem}"
            
            # Add to store
            chunk_ids = store.add_document(content, metadata, doc_id)
            total_chunks += len(chunk_ids)
            processed_files += 1
            
        except Exception as e:
            errors.append(f"{filepath}: {str(e)}")
            print(f"  ERROR processing {filepath.name}: {e}")
    
    elapsed_time = time.time() - start_time
    
    # Final statistics
    final_count = store.collection.count() if store.collection else 0
    
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"Files processed: {processed_files}/{len(txt_files)}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Total chunks in database: {final_count}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    # Get detailed stats
    stats = store.get_stats()
    
    print("\nChunks by vehicle type:")
    for vtype, count in sorted(stats.get("vehicle_types", {}).items()):
        print(f"  {vtype}: {count}")
    
    return {
        "files_processed": processed_files,
        "total_chunks": total_chunks,
        "final_count": final_count,
        "elapsed_time": elapsed_time,
        "errors": len(errors)
    }


def test_search(db_dir: str = "./chroma_db"):
    """Run some test searches to verify the ingestion."""
    print("\n" + "=" * 60)
    print("Testing Search Functionality")
    print("=" * 60)
    
    store = DocumentStore(persist_directory=db_dir)
    
    test_queries = [
        ("lift equation", "fixed_wing"),
        ("battery endurance calculation", "drone"),
        ("delta-v rocket equation", "rocket"),
        ("orbital velocity", "satellite"),
        ("rotor disk loading", "helicopter"),
        ("glide ratio", "glider"),
    ]
    
    for query, expected_type in test_queries:
        print(f"\nQuery: '{query}'")
        print(f"Expected vehicle type: {expected_type}")
        
        results = store.search(query, vehicle_type=expected_type, top_k=2)
        
        if results:
            for i, r in enumerate(results):
                print(f"  Result {i+1}:")
                print(f"    Score: {r.score:.3f}")
                print(f"    Type: {r.metadata.get('vehicle_type', 'unknown')}")
                print(f"    Title: {r.metadata.get('title', 'unknown')[:50]}")
                print(f"    Preview: {r.content[:100]}...")
        else:
            print("  No results found!")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest aerospace documents into ChromaDB"
    )
    parser.add_argument(
        "--papers-dir",
        default="./data/papers",
        help="Directory containing paper documents"
    )
    parser.add_argument(
        "--db-dir",
        default="./chroma_db",
        help="ChromaDB storage directory"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing documents before ingesting"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test searches after ingestion"
    )
    
    args = parser.parse_args()
    
    # Run ingestion
    stats = ingest_documents(
        papers_dir=args.papers_dir,
        db_dir=args.db_dir,
        clear_existing=args.clear
    )
    
    # Run tests if requested
    if args.test and "error" not in stats:
        test_search(args.db_dir)
    
    return 0 if "error" not in stats else 1


if __name__ == "__main__":
    sys.exit(main())
