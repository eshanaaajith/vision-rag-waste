import os
from pathlib import Path

from pypdf import PdfReader
from bs4 import BeautifulSoup


DOCUMENTS_DIR = Path("rag/documents")


def extract_pdf(pdf_path):
    """Extract text from a PDF."""
    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append(
                f"\n--- PAGE {page_number} ---\n{text}"
            )

    return "\n".join(pages)


def extract_html(html_path):
    """Extract visible text from an HTML webpage."""
    with open(
        html_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove elements that don't contain useful page content
    for element in soup(
        ["script", "style", "noscript", "nav", "footer"]
    ):
        element.decompose()

    text = soup.get_text(
        separator="\n",
        strip=True
    )

    return text


def clean_text(text):
    """Basic text cleanup."""
    lines = []

    for line in text.splitlines():
        line = " ".join(line.split())

        if line:
            lines.append(line)

    return "\n".join(lines)


def process_document(path):
    """Extract text based on file type."""

    if path.suffix.lower() == ".pdf":
        text = extract_pdf(path)

    elif path.suffix.lower() == ".html":
        text = extract_html(path)

    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}"
        )

    return clean_text(text)


def main():

    print("=" * 60)
    print("RAG DOCUMENT TEXT EXTRACTION")
    print("=" * 60)

    documents = sorted(
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.suffix.lower() in [".pdf", ".html"]
    )

    if not documents:
        print("No PDF/HTML documents found.")
        return

    total_chars = 0

    for document in documents:

        print(f"\nProcessing: {document.name}")

        text = process_document(document)

        output_path = document.with_suffix(".txt")

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:
            f.write(text)

        char_count = len(text)
        total_chars += char_count

        print(f"Characters extracted: {char_count:,}")
        print(f"Saved to: {output_path}")

    print("\n" + "=" * 60)
    print(f"Documents processed: {len(documents)}")
    print(f"Total extracted characters: {total_chars:,}")
    print("Extraction complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()