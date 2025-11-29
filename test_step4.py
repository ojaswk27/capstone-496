import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from rag.document_store import DocumentStore
from rag.retriever import AerospaceRetriever
from rag.extractor import AerospaceExtractor


def test_extraction():
    # 1. Setup
    print("⚙️  Initializing System...")
    store = DocumentStore()
    retriever = AerospaceRetriever(store)
    extractor = AerospaceExtractor(retriever)

    # 2. Test Queries
    scenarios = [
        {
            "query": "How do I calculate the lift coefficient?",
            "type": "fixed_wing"
        },
        {
            "query": "rocket equation delta v",
            "type": "rockets"
        }
    ]

    for scen in scenarios:
        print(f"\n🧪 Testing: {scen['query']} ({scen['type']})")
        print("-" * 50)

        formulas = extractor.extract_formulas(scen['query'], scen['type'])

        for f in formulas:
            print(f"📌 Name: {f.get('name')}")
            print(f"   Code: {f.get('expression')}")
            print(f"   Vars: {f.get('variables')}")
            print(f"   Desc: {f.get('description')}")
            print("-" * 20)


if __name__ == "__main__":
    test_extraction()
