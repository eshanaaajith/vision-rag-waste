from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
import pillow_heif

from model import WasteCNN
from dataset import CLASS_NAMES


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/best_waste_cnn.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# HEIC support
# --------------------------------------------------

pillow_heif.register_heif_opener()


# --------------------------------------------------
# Same preprocessing used during evaluation
# --------------------------------------------------

eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# --------------------------------------------------
# Load trained classifier
# --------------------------------------------------

def load_classifier():

    model = WasteCNN(
        num_classes=len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# Classify one image
# --------------------------------------------------

def classify_image(
    image_path,
    model
):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = eval_transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = prediction.item()

    return {
        "image_path": str(image_path),
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(confidence.item()),
    }


# --------------------------------------------------
# Interactive test
# --------------------------------------------------

def main():

    print("=" * 60)
    print("WASTE IMAGE CLASSIFIER")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    model = load_classifier()

    image_path = input(
        "\nEnter image path: "
    ).strip()

    if not image_path:
        print("No image path provided.")
        return

    result = classify_image(
        image_path,
        model
    )

    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULT")
    print("=" * 60)

    print(
        f"Image      : {result['image_path']}"
    )

    print(
        f"Prediction : {result['predicted_class']}"
    )

    print(
        f"Confidence : {result['confidence']:.4f} "
        f"({result['confidence'] * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()