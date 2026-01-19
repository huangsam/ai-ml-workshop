"""
Decision Tree Example using Breast Cancer Dataset

This script demonstrates how to train and evaluate a Decision Tree classifier
on the Breast Cancer dataset. Decision Trees are simple, interpretable models
that split data based on feature values to make predictions.
"""

# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def main():
    # Step 1: Load and prepare the dataset
    # The Breast Cancer dataset is a classic binary classification problem
    # where we predict if a tumor is malignant (1) or benign (0)
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)  # Convert to DataFrame for easier handling
    y = pd.Series(cancer.target, name="target")  # Target variable (0 = benign, 1 = malignant)

    print("Breast Cancer Dataset:")
    print(f"Features: {len(X.columns)} features")  # Number of input features (e.g., tumor size, texture)
    print(f"Target classes: {cancer.target_names}")  # Class names: ['malignant', 'benign']
    print(f"Dataset shape: {X.shape}")  # Total samples and features
    print(f"Class distribution: {y.value_counts().to_dict()}")  # Count of each class
    print()

    # Step 2: Split the data into training and testing sets
    # We use 80% for training and 20% for testing to evaluate model performance
    # Stratified split ensures balanced class distribution in both sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set shape: {X_train.shape}")  # Data used to train the model
    print(f"Test set shape: {X_test.shape}")  # Data used to evaluate the model
    print()

    # Step 3: Hyperparameter tuning
    # Decision Trees can overfit if not tuned, so we search for optimal parameters
    # RandomizedSearchCV tries random combinations for efficiency
    param_grid = {
        "max_depth": [None, 5, 10, 20, 30],  # Maximum depth of the tree; None means unlimited
        "min_samples_split": [2, 5, 10, 20],  # Minimum number of samples required to split an internal node
        "min_samples_leaf": [1, 2, 4, 8],  # Minimum number of samples required to be at a leaf node
        "criterion": ["gini", "entropy"],  # Function to measure the quality of a split
    }

    search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_grid, n_iter=20, cv=5, random_state=42, verbose=1)
    search.fit(X_train, y_train)  # Fit the search on training data with cross-validation

    # Best model from hyperparameter search
    model = search.best_estimator_
    print("Best hyperparameters found:")
    print(search.best_params_)  # Print the optimal parameters
    print()

    print("Model trained successfully!")
    print(f"Tree depth: {model.get_depth()}")  # Depth of the trained tree
    print(f"Number of leaves: {model.get_n_leaves()}")  # Number of leaf nodes
    print()

    # Step 4: Make predictions on the test set
    y_pred = model.predict(X_test)  # Predict class labels for test data

    # Step 5: Evaluate the model
    # Accuracy measures overall correctness
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")  # Percentage of correct predictions
    print()
    # Detailed report including precision, recall, F1-score per class
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))
    print()

    # Confusion matrix shows true vs predicted labels
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)  # Raw matrix
    print()

    # Step 6: Visualize results
    # Plot confusion matrix as a heatmap for better understanding
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.title("Confusion Matrix - Decision Tree (Tuned)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("decision_tree_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix plot saved as 'decision_tree_confusion_matrix.png'")

    # Feature importance shows which features were most useful
    feature_importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)

    print("Top 5 Most Important Features:")
    print(feature_importance.head())  # Features ranked by importance
    print()

    # Visualize the decision tree structure
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X.columns, class_names=cancer.target_names, filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree Visualization")
    plt.savefig("decision_tree_visualization.png", dpi=300, bbox_inches="tight")
    print("Decision tree visualization saved as 'decision_tree_visualization.png'")


if __name__ == "__main__":
    main()
