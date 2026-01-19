"""
Random Forest Example using Breast Cancer Dataset

This script demonstrates Random Forest, an ensemble method that combines
multiple decision trees for better accuracy and reduced overfitting.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split


def main():
    # Step 1: Load and prepare the dataset
    # Same Breast Cancer dataset as other examples
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

    # Step 3: Hyperparameter tuning
    # Random Forest has many parameters to control tree growth and ensemble
    param_grid = {
        "n_estimators": [50, 100, 200, 300],  # Number of trees in the forest
        "max_depth": [None, 10, 20, 30],  # Maximum depth of each tree; None means unlimited
        "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        "bootstrap": [True, False],  # Whether bootstrap samples are used when building trees
    }

    search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=20, cv=5, random_state=42, verbose=1)
    search.fit(X_train, y_train)  # Tune parameters

    # Best model
    model = search.best_estimator_
    print("Best hyperparameters found:")
    print(search.best_params_)
    print()

    print("Model trained successfully!")
    print(f"Number of trees: {model.n_estimators}")  # How many trees
    print(f"Max depth: {model.max_depth}")  # Tree depth limit
    print(f"Min samples split: {model.min_samples_split}")  # Split threshold
    print(f"Min samples leaf: {model.min_samples_leaf}")  # Leaf size minimum
    print()

    # Step 4: Make predictions
    y_pred = model.predict(X_test)  # Class predictions

    # Step 5: Evaluate the model
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

    # Step 6: Visualize results
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.title("Confusion Matrix - Random Forest (Tuned)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("random_forest_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix plot saved as 'random_forest_confusion_matrix.png'")

    # Step 7: Feature importance
    # Random Forest provides built-in feature importance
    feature_importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)

    print("Top 5 Most Important Features:")
    print(feature_importance.head())
    print()

    # Plot feature importance as a bar chart
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features["importance"])
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Feature Importance")
    plt.title("Top 10 Feature Importances - Random Forest")
    plt.gca().invert_yaxis()  # Highest at top
    plt.savefig("random_forest_feature_importance.png", dpi=300, bbox_inches="tight")
    print("Feature importance plot saved as 'random_forest_feature_importance.png'")


if __name__ == "__main__":
    main()
