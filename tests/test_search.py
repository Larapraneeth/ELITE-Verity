from app.services.embedding_service import generate_embeddings
from app.services.vector_store import VectorStore


query = "What is this document about?"

query_embedding = generate_embeddings([query])[0]

store = VectorStore()

result = store.search(
    query_embedding,
    n_results=3
)

print("QUERY:", query)
print("\nRESULTS:")

for metadata, document, distance in zip(
    result["metadatas"][0],
    result["documents"][0],
    result["distances"][0]
):
    print("\n----------------")
    print("Filename:", metadata["filename"])
    print("Page:", metadata["page"])
    print("Chunk ID:", metadata["chunk_id"])
    print("Distance:", distance)
    print("Text:", document[:300])