"""
Logistic Regression Example using Breast Cancer Dataset

This script demonstrates logistic regression for binary classification.
Logistic regression predicts probabilities and classifies based on a threshold.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # Step 1: Load and prepare the dataset
    # Breast Cancer dataset: features from tumor measurements
    # Target: 0 (benign) or 1 (malignant)
    cancer = load_breast_cancer()
    X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y = pd.Series(cancer.target, name="target")

    print("Breast Cancer Dataset:")
    print(f"Features: {len(X.columns)} features")  # Number of features
    print(f"Target classes: {cancer.target_names}")  # Class names
    print(f"Dataset shape: {X.shape}")  # Samples and features
    print(f"Class distribution: {y.value_counts().to_dict()}")  # Class counts
    print()

    # Step 2: Split the data
    # Stratified split keeps class proportions the same in train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print()

    # Step 3: Define Pipeline and Hyperparameter Grid
    # Using a Pipeline is best practice to prevent data leakage during cross-validation.
    # The scaler is fitted ONLY on the training fold within the CV loop.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Preprocessing step
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),  # Model step
        ]
    )

    # Parameter grid for the pipeline steps
    # Note the double underscore notation: 'step_name__parameter_name'
    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength; smaller values specify stronger regularization
        "classifier__solver": ["lbfgs", "liblinear"],  # Algorithm to use in the optimization problem
    }

    # GridSearchCV will now cross-validate the entire pipeline
    search = RandomizedSearchCV(pipeline, param_grid, n_iter=20, cv=5, random_state=42, verbose=1)
    search.fit(X_train, y_train)  # Search for best params

    # Best model (it's a fitted pipeline ready for prediction)
    best_pipeline = search.best_estimator_
    print("Best hyperparameters found:")
    print(search.best_params_)
    print()

    print("Model trained successfully!")
    # Access the logistic regression model inside the pipeline to view coefficients
    final_model = best_pipeline.named_steps["classifier"]
    print(f"Coefficients shape: {final_model.coef_.shape}")  # Weights for each feature
    print(f"Intercept: {final_model.intercept_}")  # Bias term
    print()

    # Step 5: Make predictions
    # The pipeline automatically applies the scaler (fitted on train) to the test data
    y_pred = best_pipeline.predict(X_test)  # Class predictions

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)  # Fraction of correct predictions
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification Report:")  # Precision, recall, F1 per class
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))
    print()

    # Confusion matrix: true vs predicted counts
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()

    # Step 7: Visualize results
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.title("Confusion Matrix - Logistic Regression (Tuned)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("logistic_regression_confusion_matrix.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix plot saved as 'logistic_regression_confusion_matrix.png'")

    # Step 8: Feature importance
    # Absolute coefficients indicate feature influence
    feature_importance = pd.DataFrame({"feature": X.columns, "importance": np.abs(final_model.coef_[0])}).sort_values("importance", ascending=False)

    print("Top 5 Most Important Features:")
    print(feature_importance.head())


if __name__ == "__main__":
    main()
