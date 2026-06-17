"""Convolutional Neural Network (CNN) in PyTorch using synthetic shapes for classification."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper to generate synthetic geometric shape images (28x28)
def generate_shape_dataset(num_samples=750):
    images = []
    labels = []

    np.random.seed(42)
    for _ in range(num_samples):
        # 0: Circle, 1: Square, 2: Triangle
        shape_type = np.random.randint(0, 3)
        img = np.zeros((28, 28), dtype=np.float32)

        center_x = np.random.randint(10, 18)
        center_y = np.random.randint(10, 18)
        size = np.random.randint(5, 9)

        if shape_type == 0:
            # Circle
            y, x = np.ogrid[:28, :28]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= size**2
            img[mask] = 1.0
        elif shape_type == 1:
            # Square
            img[center_y - size : center_y + size, center_x - size : center_x + size] = 1.0
        elif shape_type == 2:
            # Triangle
            for r in range(28):
                for c in range(28):
                    # Simple bounding triangle approximation
                    if (r >= center_y - size) and (r <= center_y + size):
                        width = int(size * (r - (center_y - size)) / (2 * size))
                        if abs(c - center_x) <= width:
                            img[r, c] = 1.0

        # Add noise
        img += np.random.normal(0.0, 0.1, img.shape)
        img = np.clip(img, 0.0, 1.0)

        images.append(img)
        labels.append(shape_type)

    # Reshape to (N, 1, 28, 28) for PyTorch Conv2d
    images = np.expand_dims(np.array(images), 1)
    labels = np.array(labels)

    return torch.tensor(images), torch.tensor(labels, dtype=torch.long)


# CNN Model
class ShapeClassifierCNN(nn.Module):
    def __init__(self, filter_count=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, filter_count, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(filter_count, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # 28x28 -> MaxPool(2x2) -> 14x14 -> MaxPool(2x2) -> 7x7
        self.fc = nn.Linear(16 * 7 * 7, 3)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main(hook=None, config=None):
    from workshop.utils.hooks import NoOpProgressHook

    config = config or {}
    hook = hook or NoOpProgressHook()

    epochs = int(config.get("epochs", 5))
    batch_size = int(config.get("batch_size", 32))
    learning_rate = float(config.get("learning_rate", 0.01))
    filter_count = int(config.get("filter_count", 8))

    print("PyTorch Convolutional Neural Network (CNN)")
    print("=" * 45)
    print(f"Epochs: {epochs}, Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}, Conv1 Filters: {filter_count}")
    print()

    if hook.is_cancelled():
        return
    hook.update_stage("Data Ingestion", 10)
    print("Generating synthetic shapes dataset...")
    X, Y = generate_shape_dataset(num_samples=750)

    # Train / Test split
    split = 600
    train_dataset = TensorDataset(X[:split], Y[:split])
    test_dataset = TensorDataset(X[split:], Y[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Dataset generated. Train samples: {split}, Test samples: {150}.")

    if hook.is_cancelled():
        return
    hook.update_stage("Model Initialization", 20)

    model = ShapeClassifierCNN(filter_count=filter_count).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("CNN model initialized.")

    if hook.is_cancelled():
        return
    hook.update_stage("Training", 30)

    for epoch in range(epochs):
        if hook.is_cancelled():
            return

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        progress = 30 + int(40 * ((epoch + 1) / epochs))
        hook.update_stage("Training", progress)
        hook.update_metrics({"epoch": epoch + 1, "loss": float(epoch_loss), "accuracy": float(epoch_acc)})
        print(f"Epoch {epoch + 1:2d}/{epochs}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.2%}")

    if hook.is_cancelled():
        return
    hook.update_stage("Testing", 75)

    model.eval()
    all_preds = []
    all_targets = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_accuracy = test_correct / test_total
    print(f"\nEvaluation Complete. Test Accuracy: {test_accuracy:.2%}")

    if hook.is_cancelled():
        return
    hook.update_stage("Filter Extraction", 85)
    print("Extracting convolutional feature activations...")

    # Grab a sample from each class
    sample_indices = [np.where(Y.numpy() == c)[0][0] for c in range(3)]
    samples = X[sample_indices].to(device)  # (3, 1, 28, 28)

    with torch.no_grad():
        # Get activations of the first Conv2d layer
        # shape: (3, filter_count, 28, 28)
        activations = model.conv1(samples)

    # 1. Feature Activations Plot
    # Plot input sample and its first 4 feature maps
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    class_labels = ["Circle", "Square", "Triangle"]

    for i in range(3):
        # Original Image
        axes[i, 0].imshow(samples[i, 0].cpu().numpy(), cmap="gray")
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title("Input Shape")
        axes[i, 0].set_ylabel(class_labels[i], rotation=0, labelpad=25, fontweight="bold")

        # Activations for first 4 filters
        for f in range(4):
            ax = axes[i, f + 1]
            if f < filter_count:
                act = activations[i, f].cpu().numpy()
                ax.imshow(act, cmap="viridis")
            else:
                ax.imshow(np.zeros((28, 28)), cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Filter {f + 1}")

    plt.suptitle("PyTorch CNN Convolutional Feature Activations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("cnn_feature_activations.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Confusion Matrix Plot
    conf_matrix = np.zeros((3, 3), dtype=int)
    for t, p in zip(all_targets, all_preds):
        conf_matrix[t, p] += 1

    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.colorbar(label="Sample Count")

    for i in range(3):
        for j in range(3):
            plt.text(
                j,
                i,
                str(conf_matrix[i, j]),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > (len(all_targets) / 6) else "black",
                fontweight="bold",
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("CNN Classifier Confusion Matrix")
    plt.xticks(range(3), class_labels)
    plt.yticks(range(3), class_labels)
    plt.tight_layout()
    plt.savefig("cnn_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    hook.update_stage("Complete", 100)
    print("✓ Saved CNN visualizations.")
