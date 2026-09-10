from pathlib import Path
import sys
import pandas as pd

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from classify_image import classify_image, load_classifier
from retrieve import load_resources
from rag_answer import generate_answer, gather_evidence, format_retrieved_sources


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TEST_CSV = Path("data/processed/test.csv")
OUTPUT_CSV = Path("results/end_to_end_evaluation.csv")

CASES_PER_CLASS = 3

QUESTIONS = {
    "Glass": "How should glass waste be handled?",
    "Leather": "How should leather waste be handled?",
    "Organic": "How should organic waste be handled?",
    "Paper": "How should paper waste be handled?",
    "Plastic": "How should plastic waste be recycled?",
}


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("15-CASE END-TO-END EVALUATION")
    print("=" * 70)

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------

    print("\n[1/3] Loading test dataset...")

    df = pd.read_csv(TEST_CSV)

    selected_cases = []

    for waste_class in QUESTIONS:

        class_df = df[df["class"] == waste_class]

        if len(class_df) < CASES_PER_CLASS:
            raise ValueError(
                f"Not enough test images for {waste_class}"
            )

        selected_cases.append(
            class_df.sample(
                n=CASES_PER_CLASS,
                random_state=42
            )
        )

    evaluation_df = pd.concat(
        selected_cases,
        ignore_index=True
    )

    print(
        f"Selected {len(evaluation_df)} test images "
        f"({CASES_PER_CLASS} per class)."
    )
    classifier = load_classifier()

    # --------------------------------------------------
    # Load models
    # --------------------------------------------------

    print("\n[2/3] Loading retrieval resources...")

    retrieval_model, index, metadata = load_resources()

    print(f"Indexed chunks: {index.ntotal}")

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------

    print("\n[3/3] Running 15 end-to-end cases...")

    results = []

    for i, row in evaluation_df.iterrows():

        image_path = str(row["image_path"])
        true_class = row["class"]
        question = QUESTIONS[true_class]

        print("\n" + "=" * 70)
        print(f"CASE {i + 1}/15")
        print("=" * 70)

        print(f"True class : {true_class}")
        print(f"Image      : {image_path}")
        print(f"Question   : {question}")

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        prediction = classify_image(image_path, classifier)

        predicted_class = prediction["predicted_class"]
        confidence = prediction["confidence"]

        classification_correct = (
            predicted_class == true_class
        )

        print(
            f"Prediction : {predicted_class} "
            f"({confidence:.4f})"
        )

        print(
            f"Correct    : {classification_correct}"
        )

        # --------------------------------------------------
        # Retrieval
        # --------------------------------------------------

        retrieved = gather_evidence(
            question,
            retrieval_model,
            index,
            metadata,
            waste_class=predicted_class
        )

        displayed_sources = format_retrieved_sources(retrieved)

        print(
            "Retrieved  : "
            + ", ".join(displayed_sources)
        )

        # --------------------------------------------------
        # Grounded answer
        # --------------------------------------------------

        grounded_question = f"""
The classified waste item belongs to the
"{predicted_class}" class.

User question:
{question}
"""

        answer = generate_answer(
            grounded_question,
            retrieved
        )

        print(f"Answer     : {answer}")

        # --------------------------------------------------
        # Save result
        # --------------------------------------------------

        results.append({
            "case_id": i + 1,
            "image_path": image_path,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "classification_correct": classification_correct,
            "question": question,
            "retrieved_sources": " | ".join(
                displayed_sources
            ),
            "answer": answer
        })

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    output_df = pd.DataFrame(results)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8"
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    accuracy = (
        output_df["classification_correct"]
        .mean()
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Cases evaluated : {len(output_df)}"
    )

    print(
        f"Classification accuracy : "
        f"{accuracy:.2%}"
    )

    print(
        f"Results saved to : "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()