import numpy as np
from sentence_transformers import SentenceTransformer

from vector_store import load_vector_store


MODEL_NAME = "all-MiniLM-L6-v2"


def search(query, top_k=2, min_score=0.40):
    chunks, embeddings = load_vector_store()

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode([query])[0]

    # Normalize the query vector
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Normalize document vectors
    embeddings = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    # Calculate cosine similarity
    similarities = embeddings @ query_embedding

    # Get the most similar chunk indexes
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        score = float(similarities[index])

        if score >= min_score:
            results.append(
                {
                    "chunk": chunks[index],
                    "score": score,
                }
            )

    return results


if __name__ == "__main__":
    query = "What topics are covered in Data Science?"

    results = search(query)

    print(f"Query: {query}")

    if not results:
        print("\nNo sufficiently relevant results found.")

    for i, result in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(f"Similarity score: {result['score']:.4f}")
        print(result["chunk"])