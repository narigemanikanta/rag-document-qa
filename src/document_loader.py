from pathlib import Path
from pypdf import PdfReader


def load_document(pdf_path):
    """
    Extract text from a PDF.

    Returns:
        list of dictionaries:
        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        text = text.strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages