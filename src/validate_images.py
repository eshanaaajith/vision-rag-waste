from pathlib import Path
from PIL import Image
import pillow_heif


DATASET_DIR = Path(
    "data/raw/Automatic waste management dataset/"
    "Automatic waste management dataset"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".HEIC",
}


def validate_images():
    # Register HEIC support with Pillow
    pillow_heif.register_heif_opener()

    total = 0
    valid = 0
    invalid = []

    class_counts = {}

    for class_dir in sorted(DATASET_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        class_total = 0
        class_valid = 0

        for image_path in class_dir.rglob("*"):

            if not image_path.is_file():
                continue

            if image_path.suffix not in IMAGE_EXTENSIONS:
                continue

            total += 1
            class_total += 1

            try:
                with Image.open(image_path) as image:
                    image.verify()

                # Re-open after verify because verify() invalidates
                # the image object.
                with Image.open(image_path) as image:
                    image.load()

                valid += 1
                class_valid += 1

            except Exception as error:
                invalid.append(
                    {
                        "path": str(image_path),
                        "error": str(error),
                    }
                )

        class_counts[class_dir.name] = (class_total, class_valid)

    print("\n=== IMAGE VALIDATION ===\n")

    print("Class validation:")
    for class_name, (class_total, class_valid) in class_counts.items():
        print(
            f"{class_name:10s}: "
            f"{class_valid}/{class_total} valid"
        )

    print("\nOverall:")
    print(f"Total images : {total}")
    print(f"Valid images : {valid}")
    print(f"Invalid      : {len(invalid)}")

    if invalid:
        print("\nInvalid images:")
        for item in invalid:
            print(f"\n{item['path']}")
            print(f"Error: {item['error']}")

    else:
        print("\nAll images passed validation.")


if __name__ == "__main__":
    validate_images()