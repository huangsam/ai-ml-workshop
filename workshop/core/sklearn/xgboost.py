"""
XGBoost Example using Breast Cancer Dataset

This script demonstrates XGBoost, a powerful gradient boosting framework
that provides state-of-the-art performance for classification tasks.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier


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
    # XGBoost has many parameters for controlling boosting and tree growth
    param_grid = {
        "n_estimators": [50, 100, 200, 300],  # Number of boosting rounds
        "max_depth": [3, 6, 9, 12],  # Maximum depth of each tree
        "learning_rate": [0.01, 0.1, 0.2, 0.3],  # Step size shrinkage
        "subsample": [0.6, 0.8, 1.0],  # Subsample ratio of training instances
        "colsample_bytree": [0.6, 0.8, 1.0],  # Subsample ratio of columns when constructing each tree
        "gamma": [0, 0.1, 0.2, 0.3],  # Minimum loss reduction required to make a further partition
    }

    search = RandomizedSearchCV(XGBClassifier(random_state=42, eval_metric="logloss"), param_grid, n_iter=20, cv=5, random_state=42, verbose=1)
    search.fit(X_train, y_train)  # Tune parameters

    # Best model
    model = search.best_estimator_
    print("Best hyperparameters found:")
    print(search.best_params_)
    print()

    print("Model trained successfully!")
    print(f"Number of estimators: {model.n_estimators}")  # Number of boosting rounds
    print(f"Max depth: {model.max_depth}")  # Maximum tree depth
    print(f"Learning rate: {model.learning_rate}")  # Learning rate
    print(f"Subsample: {model.subsample}")  # Subsample ratio
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
    plt.title("Confusion Matrix - XGBoost (Tuned)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("xgboost_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix plot saved as 'xgboost_confusion_matrix.png'")

    # Step 7: Feature importance
    # XGBoost provides built-in feature importance
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
    plt.title("Top 10 Feature Importances - XGBoost")
    plt.gca().invert_yaxis()  # Highest at top
    plt.savefig("xgboost_feature_importance.png", dpi=300, bbox_inches="tight")
    print("Feature importance plot saved as 'xgboost_feature_importance.png'")


if __name__ == "__main__":
    main()
