import numpy as np
import ollama

from document_loader import load_document
from chunking import create_chunks
from embeddings import create_embeddings


OLLAMA_MODEL = "llama3.2:3b"

TOP_K = 5
MIN_SIMILARITY = 0.15


# ============================================================
# CREATE INDEX
# ============================================================

def create_document_index(pdf_files, pdf_names):

    all_chunks = []
    all_sources = []

    for pdf_path, pdf_name in zip(pdf_files, pdf_names):

        print("\n========================================")
        print(f"Processing: {pdf_name}")
        print("========================================")

        pages = load_document(pdf_path)

        print(f"Pages extracted: {len(pages)}")

        chunks = create_chunks(pages)

        print(f"Chunks created: {len(chunks)}")

        for chunk in chunks:
            all_chunks.append(chunk)
            all_sources.append(pdf_name)

    if not all_chunks:
        raise ValueError(
            "No text could be extracted from the uploaded documents."
        )

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    print("\nCreating embeddings...")

    embeddings = create_embeddings(texts)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    return (
        all_chunks,
        embeddings,
        all_sources
    )


# ============================================================
# DOCUMENT RELEVANCE
# ============================================================

def find_relevant_documents(
    query,
    sources,
    embeddings
):

    query_embedding = create_embeddings(
        [query]
    )[0]

    query_norm = np.linalg.norm(
        query_embedding
    )

    if query_norm == 0:
        return []

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    embedding_norms = np.maximum(
        embedding_norms,
        1e-12
    )

    similarities = (
        embeddings @ query_embedding
    ) / (
        embedding_norms * query_norm
    )

    document_scores = {}

    for index, source in enumerate(sources):

        score = float(
            similarities[index]
        )

        if source not in document_scores:
            document_scores[source] = []

        document_scores[source].append(score)

    # Use the strongest chunk from each document.
    ranked_documents = []

    for source, scores in document_scores.items():

        best_score = max(scores)

        ranked_documents.append(
            (
                source,
                best_score
            )
        )

    ranked_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\n========== DOCUMENT RANKING ==========")

    for source, score in ranked_documents:

        print(
            f"{score:.4f} | {source}"
        )

    return ranked_documents


# ============================================================
# SEARCH
# ============================================================

def search_documents(
    query,
    chunks,
    embeddings,
    sources,
    top_k=TOP_K
):

    query_embedding = create_embeddings(
        [query]
    )[0]

    query_norm = np.linalg.norm(
        query_embedding
    )

    if query_norm == 0:
        return []

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1
    )

    embedding_norms = np.maximum(
        embedding_norms,
        1e-12
    )

    similarities = (
        embeddings @ query_embedding
    ) / (
        embedding_norms * query_norm
    )

    # --------------------------------------------------------
    # Rank documents first
    # --------------------------------------------------------

    ranked_documents = find_relevant_documents(
        query,
        sources,
        embeddings
    )

    if not ranked_documents:
        return []

    # --------------------------------------------------------
    # Select the most relevant document
    # --------------------------------------------------------

    best_document = ranked_documents[0][0]

    best_document_score = ranked_documents[0][1]

    print(
        f"\nSelected document: {best_document}"
    )

    print(
        f"Document relevance: "
        f"{best_document_score:.4f}"
    )

    # --------------------------------------------------------
    # Retrieve chunks ONLY from selected document
    # --------------------------------------------------------

    candidate_indices = [
        i
        for i, source in enumerate(sources)
        if source == best_document
    ]

    candidate_indices.sort(
        key=lambda i: similarities[i],
        reverse=True
    )

    results = []

    print("\n========== CHUNK RETRIEVAL ==========")

    for index in candidate_indices:

        score = float(
            similarities[index]
        )

        print(
            f"{score:.4f} | "
            f"{sources[index]} | "
            f"Page {chunks[index]['page']}"
        )

        if score < MIN_SIMILARITY:
            continue

        results.append(
            {
                "chunk": chunks[index]["text"],
                "page": chunks[index]["page"],
                "score": score,
                "source": sources[index]
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}

DOCUMENT:
{result['source']}

PAGE:
{result['page']}

CONTENT:
{result['chunk']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    chunks,
    embeddings,
    sources,
    chat_history=None
):

    if chat_history is None:
        chat_history = []

    results = search_documents(
        question,
        chunks,
        embeddings,
        sources,
        TOP_K
    )

    if not results:

        return {
            "answer":
                "I could not find relevant "
                "information in the documents.",
            "sources": []
        }

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    conversation_text = ""

    for item in chat_history[-4:]:

        conversation_text += (
            f"User: {item['question']}\n"
        )

        conversation_text += (
            f"Assistant: {item['answer']}\n"
        )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a Retrieval-Augmented Generation
document question-answering assistant.

Answer the user's question using ONLY the
document context provided below.

Do not use outside knowledge.

Do not invent information.

Do not combine unrelated documents.

The retrieved context comes from the document
that is most relevant to the current question.

If the requested information is not contained
in the context, respond exactly:

I could not find relevant information in the documents.

Previous conversation:

{conversation_text}

Retrieved document context:

{context}

Current question:

{question}

Answer:
"""

    # --------------------------------------------------------
    # Ollama
    # --------------------------------------------------------

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response[
        "message"
    ][
        "content"
    ].strip()

    return {
        "answer": answer,
        "sources": results
    }