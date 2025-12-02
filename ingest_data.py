#!/usr/bin/env python3
"""
Enhanced ingestion for Aerospace Design Assistant
- OCR for scanned PDFs
- Math-aware page filter
- Duplicate detection via SHA-256
- Rich progress bar (optional)
"""

import os
import re
import json
import math  # Add this import
from typing import Dict, List, Optional, Any, Tuple

# LangGraph
from langgraph.graph import StateGraph, MessageGraph, END, CompiledGraph

# Rich progress bar (optional)
try:
    from rich.progress import track, Progress, TaskID
    from rich.console import Console
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

# PDF processing
import shutil, subprocess
try:
    from pypdf import PdfReader
    from pdf2image import convert_from_path
    import pytesseract
    
    # Check if tesseract binary is available
    tesseract_available = shutil.which("tesseract") is not None
    if tesseract_available:
        try:
            result = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True,
                timeout=5,
                check=False
            )
            tesseract_works = (result.returncode == 0)
        except Exception as e:
            if console:
                console.print(f"[yellow]Tesseract check failed: {e}[/yellow]")
            tesseract_works = False
    else:
        tesseract_works = False
    
    PDF_OK = tesseract_works
    
    if PDF_OK and console:
        console.print("[green]✓ PDF processing with OCR available[/green]")
    elif console:
        console.print("[yellow]⚠️  Tesseract not available - PDFs will be skipped[/yellow]")
        
except ImportError as e:
    PDF_OK = False
    if console:
        console.print(f"[yellow]⚠️  PDF libraries not installed: {e}[/yellow]")

from rag.document_store import DocumentStore

DATA_DIR = Path("data/papers")
MAX_PAGES_PER_PDF = int(os.getenv("MAX_PAGES", 50))   # safety cap
MATH_KEYWORDS = {
    "equation", "coefficient", "lift", "drag", "thrust", "moment",
    "ρ=", "C_L=", "C_D=", "Δv=", "Re=", "Ma=", "q=", "L/D", "Isp"
}

# ------------------------------------------------------------------
# OCR helper
# ------------------------------------------------------------------
def ocr_page(page_image) -> str:
    """Tesseract OCR on a single PIL image."""
    return pytesseract.image_to_string(page_image, lang="eng+equ")

# ------------------------------------------------------------------
# Math-aware page filter
# ------------------------------------------------------------------
def keep_page(text: str) -> bool:
    """Return True if page contains engineering math."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in MATH_KEYWORDS)

# ------------------------------------------------------------------
# Process single file
# ------------------------------------------------------------------
def process_file(file_path: Path, category: str) -> Optional[Dict]:
    """Return dict ready for DocumentStore or None if skipped."""
    content_parts: List[str] = []
    meta = {
        "source": file_path.name,
        "category": category,
        "filepath": str(file_path),
        "file_type": file_path.suffix.lower(),
        "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest()
    }

    if file_path.suffix.lower() == ".pdf":
        if not PDF_OK:
            if console:
                console.print(f"[yellow]⚠️  OCR libs missing – skipping {file_path.name}[/yellow]")
            return None
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            pages_to_read = min(total_pages, MAX_PAGES_PER_PDF)
            for page_num in range(pages_to_read):
                text = reader.pages[page_num].extract_text() or ""
                if not text and PDF_OK:                       # fallback to OCR
                    images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1)
                    text = ocr_page(images[0])
                if keep_page(text):
                    content_parts.append(text)
            content = "\n".join(content_parts)
        except Exception as e:
            if console:
                console.print(f"[red]❌ PDF read error {file_path.name}: {e}[/red]")
            return None
    elif file_path.suffix.lower() == ".txt":
        content = file_path.read_text(encoding="utf-8")
    else:
        return None

    if not content.strip():
        return None
    meta["content_size"] = len(content)
    return {"content": content, "metadata": meta}

# ------------------------------------------------------------------
# Main ingestion
# ------------------------------------------------------------------
def ingest_all(clear_existing: bool = True, verbose: bool = True) -> Dict:
    store = DocumentStore()
    if clear_existing:
        store.clear()
        if verbose and console:
            console.print("[bold green]✅ Database cleared[/bold green]")

    stats = {"total": 0, "pdf": 0, "txt": 0, "skipped": 0, "chunks": 0, "duplicates": 0}
    
    # Track processed SHA-256 hashes to avoid duplicates
    processed_hashes = set()

    all_files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in {".pdf", ".txt"}]
    if not all_files:
        if console:
            console.print("[red]No PDF/TXT files found – nothing to ingest[/red]")
        return stats

    iterator = track(all_files, description="Ingesting") if RICH else all_files
    for file_path in iterator:
        category = file_path.parent.name
        record = process_file(file_path, category)
        if not record:
            stats["skipped"] += 1
            continue
        
        # de-duplicate on SHA-256 using local tracking
        doc_hash = record["metadata"]["sha256"]
        if doc_hash in processed_hashes:
            stats["duplicates"] += 1
            continue
        
        processed_hashes.add(doc_hash)
        store.add_document(record["content"], metadata=record["metadata"])
        stats["total"] += 1
        stats[file_path.suffix.lower().strip(".")] += 1

    db_info = store.get_stats()
    stats["chunks"] = db_info.get("count", 0)
    if verbose and console:
        console.print(f"[bold green]Ingest complete:[/bold green] {stats}")
    return stats

# ------------------------------------------------------------------
# Test search
# ------------------------------------------------------------------
def test_search():
    if not console:
        return
    store = DocumentStore()
    queries = ["hover thrust momentum theory", "lift coefficient", "delta v rocket equation"]
    for q in queries:
        console.print(f"\n[bold]Query:[/bold] {q}")
        hits = store.search(q, top_k=2)
                console.print(f"  [cyan]{hit.metadata['category']}[/cyan] {hit.metadata['source']}")
            
# ------------------------------------------------------------------
# Design graph
# ------------------------------------------------------------------
def classify_vehicle(state):
    return "OK"

def parse_requirements(state):
    return "OK"
    
def validate_and_assume_parameters(state):
    return "OK"

def search_documents(state):
    return "OK"

def extract_formulas(state):
    return "OK"
    
def select_tools(state):
    return "OK"

def perform_calculations(state):
    return "OK"

def validate_design(state):
    return "OK"

def synthesize_output(state):
    return "OK"

START = "entrypoint"

def build_design_graph(
    verbose: bool = True,
    interactive: bool = True  # Add this parameter
) -> CompiledGraph:
    """
    Build the aerospace design LangGraph.
    
    Args:
        verbose: Whether to print progress messages
        interactive: Whether to pause for user confirmation on assumptions
        
    Returns:
        Compiled LangGraph ready for execution
    """
    # Create state graph
    graph = StateGraph(str)   # FIX ME: need a proper state class
    
    # Add nodes
    graph.add_node("classify", classify_vehicle)
    graph.add_node("parse", parse_requirements)
    graph.add_node("validate_params", validate_and_assume_parameters)  # NEW NODE
    graph.add_node("search", search_documents)
    graph.add_node("extract", extract_formulas)
    graph.add_node("select_tools", select_tools)
    graph.add_node("calculate", perform_calculations)
    graph.add_node("validate", validate_design)
    graph.add_node("synthesize", synthesize_output)
    
    # Define edges
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "parse")
    graph.add_edge("parse", "validate_params")  # NEW EDGE
    graph.add_edge("validate_params", "search")  # Modified edge
    graph.add_edge("search", "extract")
    graph.add_edge("extract", "select_tools")
    graph.add_edge("select_tools", "calculate")
    graph.add_edge("calculate", "validate")
    graph.add_edge("validate", "synthesize")
    graph.add_edge("synthesize", END)

    # Compile graph
    return graph.compile()

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    clear = "--no-clear" not in args
    if not clear and "--clear" not in args:      # ask
        clear = input("Clear DB before ingest? [Y/n] ").strip().lower() != "n"
    test = "--test" in args

    stats = ingest_all(clear_existing=clear, verbose=True)
    if test:
        test_search()

if __name__ == "__main__":
    main()