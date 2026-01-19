"""
Logistic Regression Example using Breast Cancer Dataset
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


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

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning with RandomizedSearchCV
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength; smaller values specify stronger regularization
        "penalty": ["l1", "l2"],  # Regularization norm (l1 for Lasso, l2 for Ridge)
        "solver": ["liblinear", "lbfgs"],  # Algorithm to use in the optimization problem
    }

    search = RandomizedSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid, n_iter=20, cv=5, random_state=42, verbose=1)
    search.fit(X_train_scaled, y_train)

    # Best model
    model = search.best_estimator_
    print("Best hyperparameters found:")
    print(search.best_params_)
    print()

    print("Model trained successfully!")
    print(f"Coefficients shape: {model.coef_.shape}")
    print(f"Intercept: {model.intercept_}")
    print()

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    model.predict_proba(X_test_scaled)[:, 1]

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
    plt.title("Confusion Matrix - Logistic Regression (Tuned)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("logistic_regression_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix plot saved as 'logistic_regression_confusion_matrix.png'")

    # Feature importance (absolute coefficients)
    feature_importance = pd.DataFrame({"feature": X.columns, "importance": np.abs(model.coef_[0])}).sort_values("importance", ascending=False)

    print("Top 5 Most Important Features:")
    print(feature_importance.head())


if __name__ == "__main__":
    main()
