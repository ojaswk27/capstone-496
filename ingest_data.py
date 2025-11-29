#!/usr/bin/env python3
"""
Enhanced document ingestion script for the Aerospace Design Assistant.
Processes both .txt and .pdf files from the data directory into the vector database.

Usage:
    python ingest_data.py              # Normal run (prompts before clearing)
    python ingest_data.py --clear      # Force clear without prompt
    python ingest_data.py --no-clear   # Add new docs without clearing
    python ingest_data.py --test       # Run test search after ingestion
    python ingest_data.py --clear --test  # Clear, ingest, and test
"""

import os
import sys
from pathlib import Path
from rag.document_store import DocumentStore
from pypdf import PdfReader

DATA_DIR = "data/papers"


def ingest_all(clear_existing: bool = True, verbose: bool = True):
    """
    Main ingestion function that processes all documents in the data directory.

    Args:
        clear_existing: If True, clears the database before ingesting
        verbose: If True, prints detailed progress information

    Returns:
        dict: Statistics about the ingestion process
    """

    if verbose:
        print("🚀 Starting Document Ingestion...")
        print(f"📂 Data directory: {DATA_DIR}")

    # Initialize document store
    try:
        store = DocumentStore()
    except Exception as e:
        print(f"❌ Error initializing DocumentStore: {e}")
        return {"success": False, "error": str(e)}

    # Clear existing data if requested
    if clear_existing:
        if verbose:
            print("🗑️  Clearing existing database...")
        try:
            store.clear()
            if verbose:
                print("✅ Database cleared")
        except Exception as e:
            print(f"❌ Error clearing database: {e}")
            return {"success": False, "error": str(e)}

    # Statistics tracking
    stats = {
        "total_files": 0,
        "txt_files": 0,
        "pdf_files": 0,
        "failed_files": 0,
        "categories": {},
        "success": True
    }

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print(f"   Please create it and add your documents.")
        return {"success": False, "error": "Data directory not found"}

    # Process all files
    if verbose:
        print(f"\n📚 Scanning for documents...\n")

    for root, dirs, files in os.walk(DATA_DIR):
        category = os.path.basename(root)

        # Skip the root papers directory itself
        if root == DATA_DIR:
            continue

        # Initialize category counter
        if category not in stats["categories"]:
            stats["categories"][category] = {"txt": 0, "pdf": 0, "failed": 0}

        # Process each file
        for filename in files:
            file_path = os.path.join(root, filename)
            content = ""
            file_type = None

            try:
                # Handle Text Files
                if filename.endswith(".txt"):
                    file_type = "txt"
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if not content.strip():
                        if verbose:
                            print(f"   ⚠️  [{category}] Skipping empty file: {filename}")
                        continue

                # Handle PDF Files
                elif filename.endswith(".pdf"):
                    file_type = "pdf"
                    try:
                        reader = PdfReader(file_path)
                        pages_text = []
                        for page_num, page in enumerate(reader.pages, 1):
                            try:
                                page_text = page.extract_text()
                                if page_text:
                                    pages_text.append(page_text)
                            except Exception as e:
                                if verbose:
                                    print(f"   ⚠️  [{category}] Error reading page {page_num} of {filename}: {e}")

                        content = "\n".join(pages_text)

                        if not content.strip():
                            if verbose:
                                print(f"   ⚠️  [{category}] No text extracted from PDF: {filename}")
                            stats["categories"][category]["failed"] += 1
                            stats["failed_files"] += 1
                            continue

                    except Exception as e:
                        if verbose:
                            print(f"   ❌ [{category}] Error reading PDF {filename}: {e}")
                        stats["categories"][category]["failed"] += 1
                        stats["failed_files"] += 1
                        continue

                else:
                    # Skip non-txt, non-pdf files
                    continue

                # Add document to store
                if content:
                    metadata = {
                        "source": filename,
                        "category": category,
                        "filepath": file_path,
                        "file_type": file_type
                    }

                    store.add_document(content, metadata=metadata)

                    # Update statistics
                    stats["total_files"] += 1
                    if file_type == "txt":
                        stats["txt_files"] += 1
                        stats["categories"][category]["txt"] += 1
                    elif file_type == "pdf":
                        stats["pdf_files"] += 1
                        stats["categories"][category]["pdf"] += 1

                    if verbose:
                        print(f"   ✅ [{category.upper()}] Ingested: {filename}")

            except Exception as e:
                if verbose:
                    print(f"   ❌ [{category}] Unexpected error processing {filename}: {e}")
                stats["categories"][category]["failed"] += 1
                stats["failed_files"] += 1

    # Get final database statistics
    try:
        db_stats = store.get_stats()
        stats["total_chunks"] = db_stats.get("count", 0)
    except Exception as e:
        print(f"⚠️  Could not retrieve database stats: {e}")
        stats["total_chunks"] = "Unknown"

    # Print summary
    if verbose:
        print_summary(stats)

    return stats


def print_summary(stats):
    """Print a formatted summary of the ingestion process."""
    print(f"\n{'=' * 60}")
    print(f"📊 Ingestion Complete!")
    print(f"{'=' * 60}")
    print(f"Total files processed: {stats['total_files']}")
    print(f"  - Text files (.txt): {stats['txt_files']}")
    print(f"  - PDF files (.pdf): {stats['pdf_files']}")
    print(f"  - Failed files: {stats['failed_files']}")
    print(f"\nTotal chunks in database: {stats['total_chunks']}")

    if stats['categories']:
        print(f"\n📂 By Category:")
        for category, counts in stats['categories'].items():
            total = counts['txt'] + counts['pdf']
            if total > 0:
                print(f"  - {category}: {total} files ({counts['txt']} txt, {counts['pdf']} pdf)")
                if counts['failed'] > 0:
                    print(f"    ⚠️  {counts['failed']} failed")

    print(f"\n✅ Vector database ready at: ./chroma_db/")


def test_search():
    """Run a test search to verify the database is working correctly."""
    print(f"\n{'=' * 60}")
    print(f"🔍 Running Test Search...")
    print(f"{'=' * 60}")

    try:
        store = DocumentStore()

        # Test queries for different categories
        test_queries = [
            ("hover thrust momentum theory", "drones"),
            ("lift coefficient", "fixed_wing"),
            ("delta v rocket equation", "rockets"),
        ]

        for query, expected_category in test_queries:
            print(f"\nQuery: '{query}'")
            results = store.search(query, n_results=3)

            if results:
                print(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    source = result.metadata.get('source', 'Unknown')
                    category = result.metadata.get('category', 'Unknown')
                    print(f"   {i}. [{category}] {source}")
                    print(f"      Preview: {result.content[:100]}...")
            else:
                print(f"⚠️  No results found")

        print(f"\n✅ Test search completed")

    except Exception as e:
        print(f"❌ Test search failed: {e}")


def main():
    """Main entry point with command-line argument handling."""

    # Parse command-line arguments
    args = sys.argv[1:]

    clear_db = True  # Default behavior
    run_test = False

    # Check for flags
    if "--no-clear" in args:
        clear_db = False
    elif "--clear" in args or "-c" in args:
        clear_db = True
    else:
        # Default: prompt user
        if os.path.exists("./chroma_db"):
            response = input("🗑️  Clear existing database before ingesting? (yes/no) [yes]: ").strip().lower()
            clear_db = response in ["", "y", "yes"]
        else:
            print("ℹ️  No existing database found, will create new one")

    if "--test" in args or "-t" in args:
        run_test = True

    # Show help
    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    # Run ingestion
    stats = ingest_all(clear_existing=clear_db, verbose=True)

    # Check if ingestion was successful
    if not stats.get("success", False):
        print("\n❌ Ingestion failed!")
        sys.exit(1)

    # Run test if requested
    if run_test:
        test_search()

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()