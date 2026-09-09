from pathlib import Path
import json
import re


DOCUMENTS_DIR = Path("rag/documents")
OUTPUT_FILE = Path("rag/documents/chunks.json")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def clean_text(text):
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def create_chunks(text, chunk_size=1000, overlap=200):
    """Create overlapping text chunks."""
    chunks = []

    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def main():

    print("=" * 60)
    print("RAG DOCUMENT CHUNKING")
    print("=" * 60)

    text_files = sorted(DOCUMENTS_DIR.glob("*.txt"))

    all_chunks = []

    for text_file in text_files:

        print(f"\nProcessing: {text_file.name}")

        with open(
            text_file,
            "r",
            encoding="utf-8"
        ) as f:
            text = f.read()

        text = clean_text(text)

        chunks = create_chunks(
            text,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        print(f"Characters: {len(text):,}")
        print(f"Chunks: {len(chunks):,}")

        source_id = text_file.stem.split("_")[0]

        for i, chunk in enumerate(chunks):

            all_chunks.append({
                "chunk_id": f"{source_id}_{i:05d}",
                "source_id": source_id,
                "source_file": text_file.name,
                "chunk_index": i,
                "text": chunk
            })

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print(f"Total chunks: {len(all_chunks):,}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("Chunking complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()