from app.services.chunker import chunk_text


text = """
Artificial intelligence is a field of computer science.
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks to learn patterns from data.
Natural language processing allows computers to work with human language.
Retrieval augmented generation combines information retrieval
with language model generation.
""" * 20


chunks = chunk_text(text)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
    print("Length:", len(chunk))