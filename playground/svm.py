"""
Support Vector Machine Example using Breast Cancer Dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Load the Breast Cancer dataset
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name="target")

    print("Breast Cancer Dataset:")
    print(f"Features: {len(X.columns)} features")
    print(f"Target classes: {cancer.target_names}")
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print()

    # Scale the features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the model
    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    model.fit(X_train_scaled, y_train)

    print("Model trained successfully!")
    print(f"Kernel: {model.kernel}")
    print(f"C (regularization): {model.C}")
    print(f"Gamma: {model.gamma}")
    print(f"Number of support vectors: {model.n_support_}")
    print()

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))
    print()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.title("Confusion Matrix - SVM")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("svm_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix plot saved as 'svm_confusion_matrix.png'")

    # Test different kernels
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    accuracies = []

    for kernel in kernels:
        svm_temp = SVC(kernel=kernel, C=1.0, gamma="scale", random_state=42)
        svm_temp.fit(X_train_scaled, y_train)
        y_pred_temp = svm_temp.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred_temp))

    # Plot accuracy vs kernel
    plt.figure(figsize=(10, 6))
    plt.bar(kernels, accuracies, color=["blue", "green", "red", "purple"])
    plt.xlabel("Kernel Type")
    plt.ylabel("Accuracy")
    plt.title("SVM Accuracy vs Kernel Type")
    plt.ylim(0.9, 1.0)  # Focus on high accuracy range
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.001, f"{acc:.4f}", ha="center", va="bottom")
    plt.savefig("svm_accuracy_vs_kernel.png", dpi=300, bbox_inches="tight")
    print("Accuracy vs kernel plot saved as 'svm_accuracy_vs_kernel.png'")

    best_kernel = kernels[np.argmax(accuracies)]
    best_accuracy = max(accuracies)
    print(f"\nBest kernel: {best_kernel} with accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
