"""Evaluation utilities for ML examples."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def print_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> None:
    """
    Print comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional list of class names for readability
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print("Classification Metrics")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional list of class names
        title: Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float] | None = None,
    train_accs: list[float] | None = None,
    val_accs: list[float] | None = None,
) -> None:
    """
    Plot training history (loss and accuracy over epochs).

    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch (optional)
        train_accs: Training accuracy per epoch (optional)
        val_accs: Validation accuracy per epoch (optional)
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2 if train_accs else 1, figsize=(12 if train_accs else 5, 4))

    if train_accs:
        # Loss subplot
        ax_loss = axes[0]
        # Accuracy subplot
        ax_acc = axes[1]
    else:
        ax_loss = axes if not isinstance(axes, np.ndarray) else axes

    # Plot loss
    ax_loss.plot(epochs, train_losses, "b-", label="Train Loss")
    if val_losses:
        ax_loss.plot(epochs, val_losses, "r-", label="Val Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Plot accuracy if provided
    if train_accs:
        ax_acc.plot(epochs, train_accs, "b-", label="Train Accuracy")
        if val_accs:
            ax_acc.plot(epochs, val_accs, "r-", label="Val Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Training Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
