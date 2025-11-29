import os
from rag.document_store import DocumentStore
from pypdf import PdfReader

DATA_DIR = "data/papers"


def ingest_all():
    store = DocumentStore()
    store.clear()  # Clear old dummy data from DB

    total_files = 0

    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            file_path = os.path.join(root, filename)
            category = os.path.basename(root)

            content = ""

            # Handle Text Files
            if filename.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            # Handle PDF Files (The Real Data)
            elif filename.endswith(".pdf"):
                try:
                    reader = PdfReader(file_path)
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
                except Exception as e:
                    print(f"❌ Error reading PDF {filename}: {e}")
                    continue

            if content:
                store.add_document(content, metadata={"source": filename, "category": category})
                print(f"✅ [{category.upper()}] Ingested: {filename}")
                total_files += 1

    print(f"\n🎉 Ingested {total_files} real documents.")


if __name__ == "__main__":
    ingest_all()
