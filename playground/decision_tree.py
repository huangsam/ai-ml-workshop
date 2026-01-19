"""
Decision Tree Example using Breast Cancer Dataset
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


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

    # Hyperparameter tuning with RandomizedSearchCV
    param_grid = {
        "max_depth": [None, 5, 10, 20, 30],  # Maximum depth of the tree; None means unlimited
        "min_samples_split": [2, 5, 10, 20],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1, 2, 4, 8],  # Minimum number of samples required to be at a leaf node
        "criterion": ["gini", "entropy"],  # Function to measure the quality of a split
    }

    search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_grid, n_iter=20, cv=5, random_state=42, verbose=1)
    search.fit(X_train, y_train)

    # Best model
    model = search.best_estimator_
    print("Best hyperparameters found:")
    print(search.best_params_)
    print()

    print("Model trained successfully!")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    print()

    # Make predictions
    y_pred = model.predict(X_test)

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
    plt.title("Confusion Matrix - Decision Tree (Tuned)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("decision_tree_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix plot saved as 'decision_tree_confusion_matrix.png'")

    # Feature importance
    feature_importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)

    print("Top 5 Most Important Features:")
    print(feature_importance.head())
    print()

    # Plot decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X.columns, class_names=cancer.target_names, filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree Visualization")
    plt.savefig("decision_tree_visualization.png", dpi=300, bbox_inches="tight")
    print("Decision tree visualization saved as 'decision_tree_visualization.png'")


if __name__ == "__main__":
    main()
