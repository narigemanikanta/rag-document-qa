import numpy as np

from sentence_transformers import SentenceTransformer


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model():
    """
    Load the embedding model only once.
    """

    global _model

    if _model is None:

        print(
            f"Loading embedding model: {MODEL_NAME}"
        )

        _model = SentenceTransformer(
            MODEL_NAME
        )

    return _model


def create_embeddings(texts):
    """
    Convert text into normalized embeddings.
    """

    if not texts:
        return np.array([])


    model = get_model()


    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


    return embeddings