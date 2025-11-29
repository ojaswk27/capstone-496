from rag.document_store import DocumentStore
from rag.retriever import AerospaceRetriever


def test_system():
    # 1. Load Database
    store = DocumentStore()
    retriever = AerospaceRetriever(store)

    # 2. Test Query (Specific to a category)
    print("\n--- TEST: Drone Battery Search ---")
    results = retriever.query("calculating battery endurance", vehicle_type="drones")

    for r in results:
        print(f"Found: {r.metadata['source']} (Score: {r.score:.4f})")
        print(f"Snippet: {r.content[:100]}...\n")


if __name__ == "__main__":
    test_system()
