import torch
import torch.nn as nn


class WasteCNN(nn.Module):
    """
    CNN for five-class waste image classification.

    Classes:
        0 - Glass
        1 - Leather
        2 - Organic
        3 - Paper
        4 - Plastic
    """

    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Input: 128 x 128
        # After 3 pooling layers:
        # 128 -> 64 -> 32 -> 16
        #
        # Therefore:
        # 128 channels x 16 x 16
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),

            nn.Dropout(p=0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    # Create model
    model = WasteCNN()

    print("\n=== WASTE CNN ===\n")
    print(model)

    # Count trainable parameters
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"\nTrainable parameters: {total_params:,}")

    # Dummy input
    dummy_input = torch.randn(4, 3, 128, 128)

    # Forward pass
    output = model(dummy_input)

    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (4, 5)

    print("\nForward pass: PASS")