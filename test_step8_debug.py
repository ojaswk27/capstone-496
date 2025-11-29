from rag.document_store import DocumentStore
from rag.retriever import AerospaceRetriever


def diagnostic():
    print("🔍 Running RAG Diagnostic...")

    # 1. Check DB Stats
    store = DocumentStore()
    stats = store.get_stats()
    print(f"📊 Vector DB contains {stats['count']} chunks")

    if stats['count'] == 0:
        print("❌ ERROR: Database is empty!")
        print("   → Run: python ingest_data.py")
        return

    # 2. Test Search with Simple Query
    retriever = AerospaceRetriever(store)

    # Use a simpler, more generic query that WILL match your drone PDFs
    test_queries = [
        "momentum theory",
        "hover thrust",
        "quadcopter",
        "drone"
    ]

    for q in test_queries:
        print(f"\n🔎 Testing query: '{q}'")
        results = retriever.query(q, vehicle_type="drones", k=2)

        if results:
            print(f"✅ Found {len(results)} results")
            for r in results:
                print(f"   - {r.metadata['source']}")
        else:
            print("❌ No results")


if __name__ == "__main__":
    diagnostic()
