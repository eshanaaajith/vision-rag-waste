import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from model import WasteCNN
from dataset import create_dataloaders, CLASS_NAMES


# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 32
MODEL_PATH = "models/best_waste_cnn.pth"
TEST_CSV = "data/processed/test.csv"

RESULTS_DIR = "results"
PREDICTIONS_CSV = os.path.join(
    RESULTS_DIR, "test_predictions.csv"
)
CONFUSION_MATRIX_PNG = os.path.join(
    RESULTS_DIR, "confusion_matrix.png"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Create results directory
# -----------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------
# Load test data
# -----------------------------
_, _, test_loader = create_dataloaders(
    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/test.csv",
    BATCH_SIZE
)

test_df = pd.read_csv(TEST_CSV)

print(f"Device: {DEVICE}")
print(f"Test images: {len(test_df)}")


# -----------------------------
# Load trained model
# -----------------------------
model = WasteCNN(num_classes=len(CLASS_NAMES))

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()


# -----------------------------
# Run inference
# -----------------------------
all_true = []
all_pred = []
all_confidence = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        # Convert logits into probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Highest probability = predicted class
        confidence, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_true.extend(labels.cpu().numpy())
        all_pred.extend(predictions.cpu().numpy())
        all_confidence.extend(confidence.cpu().numpy())


# -----------------------------
# Safety check
# -----------------------------
assert len(all_true) == len(test_df), (
    "Number of predictions does not match test CSV."
)


# -----------------------------
# Convert IDs to class names
# -----------------------------
true_names = [
    CLASS_NAMES[i]
    for i in all_true
]

pred_names = [
    CLASS_NAMES[i]
    for i in all_pred
]


# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(
    all_true,
    all_pred
)

macro_f1 = f1_score(
    all_true,
    all_pred,
    average="macro"
)


print("\n==============================")
print("TEST SET RESULTS")
print("==============================")

print(
    f"Test Accuracy : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Macro F1      : {macro_f1:.4f}"
)


# -----------------------------
# Per-class metrics
# -----------------------------
print("\nPer-class classification report:\n")

report = classification_report(
    all_true,
    all_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)


# -----------------------------
# Save predictions
# -----------------------------
predictions_df = test_df.copy()

predictions_df["true_class"] = true_names
predictions_df["predicted_class"] = pred_names
predictions_df["confidence"] = all_confidence

predictions_df.to_csv(
    PREDICTIONS_CSV,
    index=False
)

print(
    f"Saved predictions to: {PREDICTIONS_CSV}"
)


# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(
    all_true,
    all_pred,
    labels=list(range(len(CLASS_NAMES)))
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

ax.set_title("Waste Classification - Test Set Confusion Matrix")

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved confusion matrix to: {CONFUSION_MATRIX_PNG}"
)

print("\nEvaluation complete.")