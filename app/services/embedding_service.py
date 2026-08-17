from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts):

    if not texts:
        return []

    model = get_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings.tolist()