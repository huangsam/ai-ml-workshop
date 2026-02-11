"""
Support Vector Machine Example using Breast Cancer Dataset

This script demonstrates SVM, which finds the optimal hyperplane to separate
classes with maximum margin, using kernel tricks for non-linear data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    # Step 1: Load and prepare the dataset
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name="target")

    print("Breast Cancer Dataset:")
    print(f"Features: {len(X.columns)} features")
    print(f"Target classes: {cancer.target_names}")
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print()

    # Step 2: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print()

    # Step 3: Define Pipeline and Hyperparameter Grid
    # Using a Pipeline is best practice to prevent data leakage during cross-validation.
    # SVM is sensitive to feature scales, so standardization is crucial.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Preprocessing step
            ("svm", SVC(random_state=42)),  # Model step
        ]
    )

    # Parameter grid for the pipeline steps
    # Note the double underscore notation: 'step_name__parameter_name'
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],  # Regularization parameter; smaller values specify stronger regularization
        "svm__gamma": ["scale", "auto", 0.01, 0.1, 1],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        "svm__kernel": ["linear", "rbf", "poly"],  # Specifies the kernel type to be used in the algorithm
    }

    # GridSearchCV will now cross-validate the entire pipeline
    search = RandomizedSearchCV(pipeline, param_grid, n_iter=20, cv=5, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    # Best model (it's a fitted pipeline ready for prediction)
    best_pipeline = search.best_estimator_
    print("Best hyperparameters found:")
    print(search.best_params_)
    print()

    print("Model trained successfully!")
    # Access the SVM model inside the pipeline
    final_model = best_pipeline.named_steps["svm"]
    print(f"Kernel: {final_model.kernel}")  # Type of kernel used
    print(f"C (regularization): {final_model.C}")  # Regularization strength
    print(f"Gamma: {final_model.gamma}")  # Kernel parameter
    print(f"Number of support vectors: {final_model.n_support_}")  # Key data points
    print()

    # Step 5: Make predictions
    # The pipeline automatically applies the scaler (fitted on train) to the test data
    y_pred = best_pipeline.predict(X_test)  # Class predictions

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))
    print()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()

    # Step 7: Visualize results
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.title("Confusion Matrix - SVM (Tuned)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("svm_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix plot saved as 'svm_confusion_matrix.png'")

    # Step 8: Additional analysis - test different kernels
    # Compare kernels using the best C and gamma found
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    accuracies = []
    scaler = best_pipeline.named_steps["scaler"]
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for kernel in kernels:
        svm_temp = SVC(kernel=kernel, C=final_model.C, gamma=final_model.gamma, random_state=42)
        svm_temp.fit(X_train_scaled, y_train)
        y_pred_temp = svm_temp.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred_temp))

    # Plot accuracy vs kernel
    plt.figure(figsize=(10, 6))
    plt.bar(kernels, accuracies, color=["blue", "green", "red", "purple"])
    plt.xlabel("Kernel Type")
    plt.ylabel("Accuracy")
    plt.title("SVM Accuracy vs Kernel Type (with tuned C and gamma)")
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
