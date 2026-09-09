from pathlib import Path
from collections import Counter


# Dataset root
DATASET_DIR = Path(
    "data/raw/Automatic waste management dataset/"
    "Automatic waste management dataset"
)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".JPG", ".JPEG", ".PNG", ".HEIC"}


def inspect_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    class_counts = {}
    extension_counts = Counter()

    # Each subdirectory represents a class
    class_dirs = sorted(
        [directory for directory in DATASET_DIR.iterdir() if directory.is_dir()]
    )

    if not class_dirs:
        raise RuntimeError("No class directories found.")

    for class_dir in class_dirs:
        images = [
            file
            for file in class_dir.rglob("*")
            if file.is_file() and file.suffix in IMAGE_EXTENSIONS
        ]

        class_counts[class_dir.name] = len(images)

        for image in images:
            extension_counts[image.suffix.lower()] += 1

    total_images = sum(class_counts.values())

    print("\n=== DATASET INSPECTION ===\n")

    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name:10s}: {count}")

    print(f"{'-' * 25}")
    print(f"{'Total':10s}: {total_images}")

    print("\nFile extensions:")
    for extension, count in sorted(extension_counts.items()):
        print(f"{extension:10s}: {count}")

    print("\nDataset path:")
    print(DATASET_DIR.resolve())


if __name__ == "__main__":
    inspect_dataset()