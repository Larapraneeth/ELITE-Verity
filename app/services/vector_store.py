import chromadb


class VectorStore:

    def __init__(
        self,
        persist_directory="data/chroma"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="pdf_documents"
            )
        )

    def add_chunks(
        self,
        chunks,
        embeddings
    ):

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "filename": chunk["filename"],
                "page": chunk.get(
                    "page",
                    1
                ),
                "section": chunk.get(
                    "section",
                    "General"
                ),
                "file_type": chunk.get(
                    "file_type",
                    "unknown"
                ),
                "sheet": chunk.get(
                    "sheet",
                    ""
                ),
                "chunk_id": chunk["chunk_id"]
            }
            for chunk in chunks
        ]

        ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding,
        n_results=3,
        filename=None
    ):

        query_kwargs = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": n_results
        }

        if filename:

            query_kwargs["where"] = {
                "filename": filename
            }

        return self.collection.query(
            **query_kwargs
        )