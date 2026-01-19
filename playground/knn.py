"""
K-Nearest Neighbors Example using Breast Cancer Dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

    # Scale the features (important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the model
    k = 5  # Number of neighbors
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    print("Model trained successfully!")
    print(f"Number of neighbors (k): {k}")
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
    plt.title(f"Confusion Matrix - KNN (k={k})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("knn_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix plot saved as 'knn_confusion_matrix.png'")

    # Test different k values
    k_values = range(1, 21)
    accuracies = []

    for k_val in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k_val)
        knn_temp.fit(X_train_scaled, y_train)
        y_pred_temp = knn_temp.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred_temp))

    # Plot accuracy vs k
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker="o", linestyle="-", color="b")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title("KNN Accuracy vs Number of Neighbors")
    plt.grid(True)
    plt.xticks(k_values)
    plt.savefig("knn_accuracy_vs_k.png", dpi=300, bbox_inches="tight")
    print("Accuracy vs k plot saved as 'knn_accuracy_vs_k.png'")

    best_k = k_values[np.argmax(accuracies)]
    best_accuracy = max(accuracies)
    print(f"\nBest k value: {best_k} with accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
