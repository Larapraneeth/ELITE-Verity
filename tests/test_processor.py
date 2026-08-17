from app.services.document_processor import process_pdf


chunks = process_pdf("sample.pdf")

print("Total chunks:", len(chunks))

for chunk in chunks[:5]:
    print("\n----------------")
    print("Filename:", chunk["filename"])
    print("Page:", chunk["page"])
    print("Chunk ID:", chunk["chunk_id"])
    print("Text:", chunk["text"][:200])