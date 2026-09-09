import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_FILE = Path("rag/documents/chunks.json")
VECTOR_STORE = Path("rag/vector_store")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():

    print("=" * 60)
    print("BUILDING RAG VECTOR INDEX")
    print("=" * 60)

    # -----------------------------
    # Load chunks
    # -----------------------------
    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    print(f"Loaded chunks: {len(chunks)}")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # -----------------------------
    # Load embedding model
    # -----------------------------
    print(f"\nLoading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    # -----------------------------
    # Generate embeddings
    # -----------------------------
    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype(
        np.float32
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # -----------------------------
    # Build FAISS index
    # -----------------------------
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(
        f"FAISS vectors indexed: {index.ntotal}"
    )

    # -----------------------------
    # Save vector store
    # -----------------------------
    VECTOR_STORE.mkdir(
        parents=True,
        exist_ok=True
    )

    index_path = VECTOR_STORE / "waste.index"
    metadata_path = VECTOR_STORE / "metadata.json"

    faiss.write_index(
        index,
        str(index_path)
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            chunks,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("VECTOR INDEX COMPLETE")
    print("=" * 60)

    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Vectors: {index.ntotal}")
    print(f"Dimensions: {dimension}")


if __name__ == "__main__":
    main()