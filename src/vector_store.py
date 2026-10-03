import json
from pathlib import Path

import numpy as np


VECTOR_STORE_DIR = Path("data/vector_store")
EMBEDDINGS_FILE = VECTOR_STORE_DIR / "embeddings.npy"
CHUNKS_FILE = VECTOR_STORE_DIR / "chunks.json"


def save_vector_store(chunks, embeddings):
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_FILE, embeddings)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print("Vector store saved successfully.")
    print(f"Embeddings saved to: {EMBEDDINGS_FILE}")
    print(f"Chunks saved to: {CHUNKS_FILE}")


def load_vector_store():
    embeddings = np.load(EMBEDDINGS_FILE)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks, embeddings


if __name__ == "__main__":
    print("Vector store module ready.")