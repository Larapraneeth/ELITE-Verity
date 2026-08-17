from pathlib import Path

from app.services.document_processor import process_pdf
from app.services.embedding_service import generate_embeddings
from app.services.vector_store import VectorStore


UPLOAD_DIR = Path("data/uploads")


def main():
    pdf_files = list(UPLOAD_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/uploads.")
        return

    store = VectorStore()

    total_files = 0
    total_chunks = 0

    print("=" * 70)
    print("PDF DOCUMENT INGESTION")
    print("=" * 70)

    for pdf_path in pdf_files:

        print(f"\nProcessing: {pdf_path.name}")

        try:
            chunks = process_pdf(pdf_path)

            if not chunks:
                print("Skipped: PDF contains no extractable text.")
                continue

            texts = [chunk["text"] for chunk in chunks]

            embeddings = generate_embeddings(texts)

            store.add_chunks(
                chunks,
                embeddings
            )

            total_files += 1
            total_chunks += len(chunks)

            print(f"Status: Success")
            print(f"Chunks: {len(chunks)}")

        except Exception as e:
            print(f"Status: Failed")
            print(f"Reason: {e}")

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"Files processed : {total_files}")
    print(f"Total chunks    : {total_chunks}")
    print("=" * 70)


if __name__ == "__main__":
    main()