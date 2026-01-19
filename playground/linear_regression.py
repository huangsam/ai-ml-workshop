"""
Linear Regression Example using Boston Housing Dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def main():
    # Load the Boston Housing dataset
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.Series(boston.target, name="MEDV")

    print("Boston Housing Dataset:")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {y.name} (Median value of owner-occupied homes in $1000's)")
    print(f"Dataset shape: {X.shape}")
    print()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print()

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model trained successfully!")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print()

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print()

    # Feature importance (absolute coefficients)
    feature_importance = pd.DataFrame({"feature": X.columns, "importance": np.abs(model.coef_)}).sort_values("importance", ascending=False)

    print("Top 5 Most Important Features:")
    print(feature_importance.head())
    print()

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted House Prices")
    plt.grid(True)
    plt.savefig("linear_regression_results.png", dpi=300, bbox_inches="tight")
    print("Plot saved as 'linear_regression_results.png'")


if __name__ == "__main__":
    main()
