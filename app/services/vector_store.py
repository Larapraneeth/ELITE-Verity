import logging

import chromadb

from app.config import CHROMA_HOST, CHROMA_PERSIST_DIRECTORY, CHROMA_PORT


logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(
        self,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    ):
        if CHROMA_HOST:
            self.client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
            )
        else:
            self.client = chromadb.PersistentClient(
                path=persist_directory
            )

        self.collection = (
            self.client.get_or_create_collection(
                name="pdf_documents"
            )
        )

    def health_check(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:
            logger.warning(
                "Chroma health check failed",
                extra={"event": "chroma.health_failed"},
            )
            return False

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
