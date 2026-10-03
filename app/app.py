import sys
import hashlib
import tempfile

from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


SRC_FOLDER = (
    PROJECT_ROOT / "src"
)


if str(SRC_FOLDER) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_FOLDER)
    )


# ============================================================
# RAG IMPORTS
# ============================================================

from rag_pipeline import (
    create_document_index,
    generate_answer
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📚 RAG Document Q&A"
)


st.write(
    "Upload one or more PDF documents and "
    "ask questions about their content "
    "using Retrieval-Augmented Generation."
)


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:

    st.session_state.chunks = None


if "embeddings" not in st.session_state:

    st.session_state.embeddings = None


if "sources" not in st.session_state:

    st.session_state.sources = None


if "document_names" not in st.session_state:

    st.session_state.document_names = []


if "conversation" not in st.session_state:

    st.session_state.conversation = []


if "upload_signature" not in st.session_state:

    st.session_state.upload_signature = None


if "processed" not in st.session_state:

    st.session_state.processed = False


if "last_result" not in st.session_state:

    st.session_state.last_result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Controls"
    )


    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.conversation = []

        st.session_state.last_result = None

        st.rerun()


    if st.button(
        "🔄 Clear Documents",
        use_container_width=True
    ):

        st.session_state.chunks = None

        st.session_state.embeddings = None

        st.session_state.sources = None

        st.session_state.document_names = []

        st.session_state.upload_signature = None

        st.session_state.processed = False

        st.session_state.conversation = []

        st.session_state.last_result = None

        st.rerun()


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    signature_parts = []


    for uploaded_file in uploaded_files:

        file_bytes = (
            uploaded_file.getvalue()
        )


        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()


        signature_parts.append(
            f"{uploaded_file.name}:{file_hash}"
        )


    upload_signature = "|".join(
        signature_parts
    )


    # --------------------------------------------------------
    # Only process when files change
    # --------------------------------------------------------

    if (
        st.session_state.upload_signature
        != upload_signature
    ):

        with st.spinner(
            "Processing PDF documents..."
        ):

            try:

                temp_folder = (
                    Path(
                        tempfile.gettempdir()
                    )
                    / "rag_document_qa"
                )


                temp_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )


                pdf_paths = []

                pdf_names = []


                # ------------------------------------------------
                # Save uploaded files
                # ------------------------------------------------

                for uploaded_file in uploaded_files:

                    file_bytes = (
                        uploaded_file.getvalue()
                    )


                    file_hash = hashlib.md5(
                        file_bytes
                    ).hexdigest()[:12]


                    safe_filename = (
                        uploaded_file.name
                        .replace(
                            "\\",
                            "_"
                        )
                        .replace(
                            "/",
                            "_"
                        )
                    )


                    pdf_path = (
                        temp_folder
                        / f"{file_hash}_{safe_filename}"
                    )


                    with open(
                        pdf_path,
                        "wb"
                    ) as file:

                        file.write(
                            file_bytes
                        )


                    pdf_paths.append(
                        str(pdf_path)
                    )


                    pdf_names.append(
                        uploaded_file.name
                    )


                # ------------------------------------------------
                # Create RAG index
                # ------------------------------------------------

                (
                    chunks,
                    embeddings,
                    sources
                ) = create_document_index(
                    pdf_paths,
                    pdf_names
                )


                # ------------------------------------------------
                # Save state
                # ------------------------------------------------

                st.session_state.chunks = chunks

                st.session_state.embeddings = embeddings

                st.session_state.sources = sources

                st.session_state.document_names = pdf_names

                st.session_state.upload_signature = (
                    upload_signature
                )

                st.session_state.processed = True

                st.session_state.conversation = []

                st.session_state.last_result = None


            except Exception as error:

                st.error(
                    f"Error processing documents: {error}"
                )

                st.exception(error)


# ============================================================
# SHOW DOCUMENTS
# ============================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} "
        f"document(s) uploaded."
    )


    if st.session_state.processed:

        st.info(
            f"Processed "
            f"{len(st.session_state.document_names)} "
            f"document(s) and created "
            f"{len(st.session_state.chunks)} chunks."
        )


        st.markdown(
            "### 📄 Uploaded Documents"
        )


        for name in (
            st.session_state.document_names
        ):

            st.write(
                f"📄 {name}"
            )


# ============================================================
# QUESTION AREA
# ============================================================

if (
    st.session_state.processed
    and st.session_state.chunks
):

    st.markdown("---")


    st.markdown(
        "### 💬 Ask a question"
    )


    question = st.text_input(
        "Ask a question about your documents:",
        placeholder=(
            "Example: What position is "
            "mentioned in the cover letter?"
        )
    )


    ask_button = st.button(
        "🔎 Ask Question",
        type="primary"
    )


    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching documents..."
            ):

                try:

                    result = generate_answer(
                        question,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        st.session_state.sources,
                        st.session_state.conversation
                    )


                    answer = result[
                        "answer"
                    ]


                    sources = result.get(
                        "sources",
                        []
                    )


                    # ------------------------------------------------
                    # Save conversation
                    # ------------------------------------------------

                    st.session_state.conversation.append(
                        {
                            "question": question,
                            "answer": answer
                        }
                    )


                    # ------------------------------------------------
                    # Save result
                    # ------------------------------------------------

                    st.session_state.last_result = {
                        "question": question,
                        "answer": answer,
                        "sources": sources
                    }


                except Exception as error:

                    st.error(
                        f"Error generating answer: {error}"
                    )

                    st.exception(error)


# ============================================================
# CURRENT ANSWER
# ============================================================

if st.session_state.last_result:

    result = (
        st.session_state.last_result
    )


    st.markdown("---")


    st.markdown(
        "## 🤖 Answer"
    )


    st.write(
        result["answer"]
    )


    # ========================================================
    # SOURCES
    # ========================================================

    source_results = result.get(
        "sources",
        []
    )


    if source_results:

        st.markdown(
            "### 📚 Sources"
        )


        for i, source in enumerate(
            source_results,
            start=1
        ):

            source_name = source.get(
                "source",
                "Unknown document"
            )


            page = source.get(
                "page",
                "Unknown"
            )


            score = source.get(
                "score",
                0.0
            )


            st.write(
                f"**Source {i}** — "
                f"{source_name} — "
                f"Page {page} — "
                f"Similarity: {score:.4f}"
            )


            with st.expander(
                f"View source {i}"
            ):

                st.write(
                    source.get(
                        "chunk",
                        "Source text unavailable."
                    )
                )


    else:

        st.info(
            "No relevant sources were found "
            "in the documents."
        )


# ============================================================
# CONVERSATION
# ============================================================

if st.session_state.conversation:

    st.markdown("---")


    st.markdown(
        "## 💬 Conversation"
    )


    for chat in (
        st.session_state.conversation
    ):

        st.markdown(
            "### 🧑 You"
        )


        st.write(
            chat["question"]
        )


        st.markdown(
            "### 🤖 Answer"
        )


        st.write(
            chat["answer"]
        )


        st.markdown("---")