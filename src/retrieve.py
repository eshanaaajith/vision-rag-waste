import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("rag/vector_store/waste.index")
METADATA_FILE = Path("rag/vector_store/metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CANDIDATE_K = 30
TOP_K = CANDIDATE_K


def load_resources():

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    model = SentenceTransformer(
        MODEL_NAME
    )

    return model, index, metadata


def retrieve(question, model, index, metadata):

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        CANDIDATE_K
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        results.append({
            "score": float(score),
            "chunk_id": metadata[idx]["chunk_id"],
            "source_id": metadata[idx]["source_id"],
            "source_file": metadata[idx]["source_file"],
            "text": metadata[idx]["text"]
        })

    return results


def main():

    print("=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)

    model, index, metadata = load_resources()

    print(
        f"Indexed chunks: {index.ntotal}"
    )

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        print("No question provided.")
        return

    results = retrieve(
        question,
        model,
        index,
        metadata
    )

    print("\n" + "=" * 60)
    print("TOP RETRIEVED PASSAGES")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- RESULT {rank} ---"
        )

        print(
            f"Score: {result['score']:.4f}"
        )

        print(
            f"Source: {result['source_id']}"
        )

        print(
            f"File: {result['source_file']}"
        )

        print(
            f"Chunk: {result['chunk_id']}"
        )

        print(
            f"\n{result['text'][:1200]}"
        )


if __name__ == "__main__":
    main()