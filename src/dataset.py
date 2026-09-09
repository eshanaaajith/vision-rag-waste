from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pillow_heif


# Register HEIC support
pillow_heif.register_heif_opener()


# Class mapping
CLASS_NAMES = [
    "Glass",
    "Leather",
    "Organic",
    "Paper",
    "Plastic",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

INDEX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_INDEX.items()
}


class WasteDataset(Dataset):
    """
    PyTorch Dataset for the waste classification dataset.
    """

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        if "image_path" not in self.data.columns:
            raise ValueError(
                "CSV must contain an 'image_path' column."
            )

        if "class" not in self.data.columns:
            raise ValueError(
                "CSV must contain a 'class' column."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = Path(row["image_path"])
        class_name = row["class"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Convert class name to numerical label
        label = CLASS_TO_INDEX[class_name]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label


# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Validation/test transformations
eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def create_dataloaders(
    train_csv="data/processed/train.csv",
    val_csv="data/processed/val.csv",
    test_csv="data/processed/test.csv",
    batch_size=32,
):
    """
    Create train, validation, and test DataLoaders.
    """

    train_dataset = WasteDataset(
        train_csv,
        transform=train_transform,
    )

    val_dataset = WasteDataset(
        val_csv,
        transform=eval_transform,
    )

    test_dataset = WasteDataset(
        test_csv,
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    print("\n=== DATASET / DATALOADER TEST ===\n")

    train_loader, val_loader, test_loader = create_dataloaders()

    print(f"Train samples      : {len(train_loader.dataset)}")
    print(f"Validation samples : {len(val_loader.dataset)}")
    print(f"Test samples       : {len(test_loader.dataset)}")

    # Get one batch
    images, labels = next(iter(train_loader))

    print("\nFirst training batch:")
    print(f"Images shape : {images.shape}")
    print(f"Labels shape : {labels.shape}")

    print("\nLabels:")
    print(labels)

    print("\nClass mapping:")
    print(CLASS_TO_INDEX)

    # Basic checks
    assert images.shape[1:] == (3, 128, 128)
    assert images.shape[0] <= 32
    assert labels.shape[0] == images.shape[0]

    print("\nDataset/DataLoader test: PASS")