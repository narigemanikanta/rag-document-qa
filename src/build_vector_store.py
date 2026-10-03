from embeddings import (
    PDF_PATH,
    extract_text,
    create_chunks,
    create_embeddings,
)

from vector_store import save_vector_store


if __name__ == "__main__":
    print("Loading PDF...")

    text = extract_text(PDF_PATH)

    print(f"Extracted {len(text)} characters.")

    print("\nCreating chunks...")

    chunks = create_chunks(text)

    print(f"Created {len(chunks)} chunks.")

    print("\nCreating embeddings...")

    embeddings = create_embeddings(chunks)

    print(f"Created embeddings with shape: {embeddings.shape}")

    print("\nSaving vector store...")

    save_vector_store(chunks, embeddings)

    print("\nVector store build completed successfully!")