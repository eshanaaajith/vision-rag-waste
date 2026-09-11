from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Dataset configuration
DATASET_DIR = Path(
    "data/raw/Automatic waste management dataset/"
    "Automatic waste management dataset"
)

OUTPUT_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
SPLIT_SUMMARY_CSV = RESULTS_DIR / "dataset_splits.csv"

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

RANDOM_SEED = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def collect_images():
    records = []

    for class_dir in sorted(DATASET_DIR.iterdir()):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for image_path in class_dir.rglob("*"):

            if not image_path.is_file():
                continue

            if image_path.suffix not in IMAGE_EXTENSIONS:
                continue

            records.append(
                {
                    "image_path": str(image_path),
                    "class": class_name,
                }
            )

    return pd.DataFrame(records)


def create_splits(df):
    # First split:
    # 70% train
    # 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_SIZE + TEST_SIZE),
        stratify=df["class"],
        random_state=RANDOM_SEED,
    )

    # Split the remaining 30% equally:
    # 15% validation
    # 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["class"],
        random_state=RANDOM_SEED,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_distribution(name, df):
    print(f"\n{name}")
    print("-" * 30)

    counts = df["class"].value_counts().sort_index()

    for class_name, count in counts.items():
        print(f"{class_name:10s}: {count}")

    print(f"{'Total':10s}: {len(df)}")


def save_split_summary(train_df, val_df, test_df):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = sorted(
        set(train_df["class"])
        | set(val_df["class"])
        | set(test_df["class"])
    )

    rows = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        counts = split_df["class"].value_counts()
        row = {"split": split_name, "images": len(split_df)}
        for class_name in class_names:
            row[class_name] = int(counts.get(class_name, 0))
        rows.append(row)

    total_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )
    total_counts = total_df["class"].value_counts()
    total_row = {"split": "total", "images": len(total_df)}
    for class_name in class_names:
        total_row[class_name] = int(total_counts.get(class_name, 0))
    rows.append(total_row)

    pd.DataFrame(rows).to_csv(
        SPLIT_SUMMARY_CSV,
        index=False
    )

    print(f"\nSaved split summary to: {SPLIT_SUMMARY_CSV}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting images...")

    df = collect_images()

    print(f"Total images found: {len(df)}")

    if len(df) != 1973:
        raise ValueError(
            f"Expected 1973 images, but found {len(df)}."
        )

    train_df, val_df, test_df = create_splits(df)

    # Save CSV files
    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    val_df.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    # Print distributions
    print_distribution("TRAIN", train_df)
    print_distribution("VALIDATION", val_df)
    print_distribution("TEST", test_df)

    print("\nSplit sizes")
    print("-" * 30)
    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")
    print(f"Total      : {len(train_df) + len(val_df) + len(test_df)}")

    save_split_summary(train_df, val_df, test_df)

    # Verify no overlap
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    print("\nNo image overlap between splits: PASS")


if __name__ == "__main__":
    main()