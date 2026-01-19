"""
Image Classification with PyTorch

This script demonstrates image classification using CIFAR-10 dataset and ResNet-18.
It covers computer vision fundamentals: CNNs, image preprocessing, and transfer learning.

Dataset: CIFAR-10 (60,000 32x32 color images in 10 classes)
Model: ResNet-18 (pre-trained on ImageNet, fine-tuned for CIFAR-10)
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from workshop.utils import get_device

# --- 1. CONFIGURATION CONSTANTS ---
BATCH_SIZE = 64  # Batch size for training and testing
NUM_EPOCHS = 5  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate for optimizer
DEVICE = "cpu"  # Default to CPU; will check for MPS acceleration
NUM_CLASSES = 10  # CIFAR-10 has 10 classes
IMAGE_SIZE = 32  # CIFAR-10 images are 32x32


def get_data_loaders() -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns train/test DataLoaders for CIFAR-10 dataset.

    Returns:
        Tuple of (train_loader, test_loader)
    """
    print("Setting up CIFAR-10 data loaders...")

    # Define transformations for training data (with augmentation)
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
            transforms.RandomCrop(IMAGE_SIZE, padding=4),  # Random crop with padding
            transforms.ToTensor(),  # Convert PIL image to tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 mean/std
        ]
    )

    # Define transformations for test data (no augmentation)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load CIFAR-10 datasets
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    return train_loader, test_loader


def create_model() -> nn.Module:
    """
    Creates and returns a ResNet-18 model fine-tuned for CIFAR-10.

    Returns:
        ResNet-18 model with modified final layer
    """
    print("Creating ResNet-18 model...")

    # Load pre-trained ResNet-18
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

    # Modify the final fully connected layer for CIFAR-10 (10 classes)
    # Original: 512 -> 1000 (ImageNet classes)
    # New: 512 -> 10 (CIFAR-10 classes)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


def train_model(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> None:
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to run on ('cpu' or 'mps')
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    print(f"Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")


def evaluate_model(model: nn.Module, test_loader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float]:
    """
    Evaluates the model on test data.

    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on

    Returns:
        Tuple of (test_loss, test_accuracy)
    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main() -> None:
    """
    Main entry point for the image classification project.
    """
    print("Image Classification with CIFAR-10 and ResNet-18")
    print("=" * 50)

    # 1. Device setup
    global DEVICE
    DEVICE = get_device()
    if DEVICE == "mps":
        print(f"🔥 Found MPS device. Using {DEVICE} for acceleration.")
    else:
        print("Using CPU for computation.")

    # 2. Load data
    train_loader, test_loader = get_data_loaders()

    # 3. Create model
    model = create_model()
    model.to(DEVICE)

    # 4. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Standard loss for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 5. Training loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_model(model, train_loader, optimizer, criterion, DEVICE)

        # Evaluate after each epoch
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    # 6. Final evaluation
    print("\nFinal Evaluation:")
    final_loss, final_acc = evaluate_model(model, test_loader, criterion, DEVICE)
    print(f"Final Test Loss: {final_loss:.4f}, Final Accuracy: {final_acc:.2f}%")

    print("\nTraining complete! 🎉")


if __name__ == "__main__":
    main()
