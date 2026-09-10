import sys
from pathlib import Path

from classify_image import load_classifier, classify_image
from retrieve import load_resources
from rag_answer import generate_answer, gather_evidence


def main():

    print("=" * 70)
    print("END-TO-END WASTE CLASSIFICATION + RAG")
    print("=" * 70)

    # --------------------------------------------------
    # Load classifier
    # --------------------------------------------------

    print("\n[1/3] Loading image classifier...")

    classifier = load_classifier()

    # --------------------------------------------------
    # Load RAG retrieval resources
    # --------------------------------------------------

    print("\n[2/3] Loading FAISS retrieval resources...")

    retrieval_model, index, metadata = load_resources()

    print(
        f"Indexed chunks: {index.ntotal}"
    )

    # --------------------------------------------------
    # Input
    # --------------------------------------------------

    print("\n[3/3] Input")

    while True:
        image_path = input(
            "Enter image path: "
        ).strip()
        
        if not image_path:
            print("No image path provided.")
            return
            
        # Strip Windows quotes
        image_path = image_path.strip('"\'')
        
        if image_path.endswith("..."):
            print("Invalid image path. Please enter a valid image file.")
            continue
            
        try:
            p = Path(image_path)
            if not p.exists() or not p.is_file():
                print("Invalid image path. Please enter a valid image file.")
                continue
                
            valid_exts = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
            if p.suffix.lower() not in valid_exts:
                print("Invalid image path. Please enter a valid image file.")
                continue
            
            from PIL import Image
            # Check if PIL can open it
            with Image.open(p) as img:
                img.verify()
                
        except Exception:
            print("Invalid image path. Please enter a valid image file.")
            continue
            
        break

    question = input(
        "Enter your question: "
    ).strip()

    if not image_path:
        print("No image path provided.")
        return

    if not question:
        print("No question provided.")
        return

    # --------------------------------------------------
    # Classify image
    # --------------------------------------------------

    classification = classify_image(
        image_path,
        classifier
    )

    predicted_class = classification[
        "predicted_class"
    ]

    confidence = classification[
        "confidence"
    ]

    print("\n" + "=" * 70)
    print("CLASSIFICATION")
    print("=" * 70)

    print(
        f"Predicted class : {predicted_class}"
    )

    print(
        f"Confidence      : "
        f"{confidence:.4f} "
        f"({confidence * 100:.2f}%)"
    )

    # --------------------------------------------------
    # Retrieve evidence
    # --------------------------------------------------

    print("\nRetrieving evidence...")

    results = gather_evidence(
        question,
        retrieval_model,
        index,
        metadata,
        waste_class=predicted_class
    )

    print("\nRetrieved sources:")

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. "
            f"{result['source_id']} "
            f"(score={result['score']:.4f})"
        )

    # --------------------------------------------------
    # Generate grounded answer
    # --------------------------------------------------

    print("\nGenerating grounded answer...")

    grounded_question = (
        f"The classified waste item belongs to the "
        f"'{predicted_class}' class.\n\n"
        f"User question: {question}"
    )

    answer = generate_answer(
        grounded_question,
        results
    )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("END-TO-END RESULT")
    print("=" * 70)

    print(
        f"Image            : {image_path}"
    )

    print(
        f"Predicted class  : {predicted_class}"
    )

    print(
        f"Confidence       : "
        f"{confidence:.4f} "
        f"({confidence * 100:.2f}%)"
    )

    print(
        f"\nQuestion         : {question}"
    )

    print(
        f"\nAnswer:\n{answer}"
    )


if __name__ == "__main__":
    main()