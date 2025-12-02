#!/usr/bin/env python3
"""
Document Ingestion Script for Aerospace Design Assistant

This script ingests all aerospace reference documents into the ChromaDB
vector database for semantic search.

Usage:
    python -m rag.ingest [--papers-dir PATH] [--db-dir PATH] [--clear]
    
Options:
    --papers-dir    Directory containing paper documents (default: data/papers)
    --db-dir        ChromaDB storage directory (default: data/chroma_db)
    --clear         Clear existing database before ingestion
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.document_store import DocumentStore, CHROMADB_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    if not CHROMADB_AVAILABLE:
        missing.append("chromadb")
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing.append("sentence-transformers")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def count_files(directory: Path) -> dict:
    """Count files by vehicle type."""
    counts = {}
    
    for subdir in directory.iterdir():
        if subdir.is_dir():
            txt_files = list(subdir.glob("*.txt"))
            if txt_files:
                counts[subdir.name] = len(txt_files)
    
    return counts


def ingest_documents(
    papers_dir: str,
    db_dir: str,
    clear_existing: bool = False
) -> dict:
    """
    Ingest all documents from papers directory into ChromaDB.
    
    Args:
        papers_dir: Directory containing paper documents
        db_dir: ChromaDB storage directory
        clear_existing: Whether to clear existing database
        
    Returns:
        Dictionary with ingestion statistics
    """
    papers_path = Path(papers_dir)
    
    # Verify papers directory exists
    if not papers_path.exists():
        print(f"Error: Papers directory not found: {papers_dir}")
        return {"success": False, "error": "Papers directory not found"}
    
    # Count files before ingestion
    file_counts = count_files(papers_path)
    total_files = sum(file_counts.values())
    
    print(f"\n{'='*60}")
    print("AEROSPACE DOCUMENT INGESTION")
    print(f"{'='*60}")
    print(f"\nPapers directory: {papers_dir}")
    print(f"Database directory: {db_dir}")
    print(f"\nDocuments by vehicle type:")
    for vtype, count in sorted(file_counts.items()):
        print(f"  {vtype}: {count} files")
    print(f"\nTotal documents to ingest: {total_files}")
    print(f"{'='*60}\n")
    
    # Create document store
    store = DocumentStore(
        persist_directory=db_dir,
        collection_name="aerospace_documents"
    )
    
    # Clear if requested
    if clear_existing:
        print("Clearing existing database...")
        store.clear()
    
    # Check current state
    initial_count = store.get_stats()["total_documents"]
    if initial_count > 0:
        print(f"Existing documents in database: {initial_count}")
        print("Use --clear to remove existing documents first.\n")
    
    # Ingest documents
    print("Starting ingestion...\n")
    start_time = datetime.now()
    
    total_chunks = store.add_documents_from_directory(
        directory=papers_dir,
        vehicle_type_from_path=True
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Get final stats
    final_stats = store.get_stats()
    
    # Print summary
    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nTime elapsed: {duration:.1f} seconds")
    print(f"Documents processed: {total_files}")
    print(f"Chunks created: {total_chunks}")
    print(f"Total chunks in database: {final_stats['total_documents']}")
    print(f"\nChunks by vehicle type:")
    for vtype, count in sorted(final_stats.get("vehicle_types", {}).items()):
        print(f"  {vtype}: {count} chunks")
    print(f"{'='*60}\n")
    
    return {
        "success": True,
        "files_processed": total_files,
        "chunks_created": total_chunks,
        "total_chunks": final_stats["total_documents"],
        "duration_seconds": duration,
        "vehicle_types": final_stats.get("vehicle_types", {})
    }


def verify_ingestion(db_dir: str) -> bool:
    """
    Verify the ingestion by running test searches.
    
    Args:
        db_dir: ChromaDB storage directory
        
    Returns:
        True if verification passes
    """
    print("\nVerifying ingestion with test searches...\n")
    
    from rag.search import SemanticSearch
    
    store = DocumentStore(persist_directory=db_dir)
    search = SemanticSearch(store)
    
    # Test queries for each vehicle type
    test_queries = [
        ("drone", "motor thrust calculation"),
        ("fixed_wing", "lift coefficient"),
        ("helicopter", "rotor disk loading"),
        ("rocket", "specific impulse"),
        ("satellite", "orbital velocity"),
        ("glider", "glide ratio"),
    ]
    
    all_passed = True
    
    for vehicle_type, query in test_queries:
        response = search.search(
            query=query,
            vehicle_type=vehicle_type,
            top_k=2
        )
        
        status = "✓" if response.total_results > 0 else "✗"
        print(f"  {status} {vehicle_type}: '{query}' -> {response.total_results} results")
        
        if response.total_results == 0:
            all_passed = False
    
    print()
    
    if all_passed:
        print("All verification tests passed!")
    else:
        print("Warning: Some verification tests failed.")
    
    return all_passed


def main():
    """Main entry point for ingestion script."""
    parser = argparse.ArgumentParser(
        description="Ingest aerospace documents into vector database"
    )
    parser.add_argument(
        "--papers-dir",
        default="data/papers",
        help="Directory containing paper documents"
    )
    parser.add_argument(
        "--db-dir",
        default="data/chroma_db",
        help="ChromaDB storage directory"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing database before ingestion"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification tests after ingestion"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run ingestion
    result = ingest_documents(
        papers_dir=args.papers_dir,
        db_dir=args.db_dir,
        clear_existing=args.clear
    )
    
    if not result["success"]:
        print(f"Ingestion failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Run verification if requested
    if args.verify:
        if not verify_ingestion(args.db_dir):
            sys.exit(1)
    
    # Save ingestion report
    report_path = Path(args.db_dir) / "ingestion_report.json"
    with open(report_path, 'w') as f:
        result["timestamp"] = datetime.now().isoformat()
        json.dump(result, f, indent=2)
    print(f"Ingestion report saved to: {report_path}")


if __name__ == "__main__":
    main()
