def create_chunks(
    pages,
    chunk_size=1000,
    chunk_overlap=150
):
    """
    Split PDF pages into overlapping text chunks.

    Each chunk keeps its original page number.
    """

    chunks = []

    for page in pages:

        page_number = page["page"]

        text = page["text"].strip()

        if not text:
            continue

        start = 0

        text_length = len(text)

        while start < text_length:

            end = start + chunk_size

            chunk_text = text[start:end]

            chunk_text = chunk_text.strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_number
                    }
                )

            if end >= text_length:
                break

            start = end - chunk_overlap

    return chunks