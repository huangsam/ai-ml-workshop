"""
Linear Regression Example using California Housing Dataset

This script demonstrates linear regression for predicting house prices.
Linear regression finds the best straight line to fit the data.

Key concepts:
- Pipeline: Encapsulates preprocessing and modeling to prevent data leakage.
- StandardScaler: Scales features to have mean=0 and variance=1.
- GridSearchCV: Hyperparameter tuning with cross-validation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # Step 1: Load and prepare the dataset
    # California Housing dataset contains features like median income, house age, etc.
    # Target is median house value in $100,000s
    california = fetch_california_housing()
    X = pd.DataFrame(california.data, columns=california.feature_names)  # Features as DataFrame
    y = pd.Series(california.target, name="MedHouseVal")  # Target variable

    print("California Housing Dataset:")
    print(f"Features: {list(X.columns)}")  # List of feature names
    print(f"Target: {y.name} (Median house value in $100,000's)")
    print(f"Dataset shape: {X.shape}")  # Number of samples and features
    print()

    # Step 2: Split the data into training and testing sets
    # Training set: used to train the model
    # Testing set: used to evaluate performance on unseen data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print()

    # Step 3: Define Pipeline and Hyperparameter Grid
    # Using a Pipeline is best practice to prevent data leakage during cross-validation.
    # The scaler is fitted ONLY on the training fold within the CV loop.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # Preprocessing step
            ("regressor", LinearRegression()),  # Model step
        ]
    )

    # Parameter grid for the pipeline steps
    # Note the double underscore notation: 'step_name__parameter_name'
    param_grid = {"regressor__fit_intercept": [True, False]}

    # GridSearchCV will now cross-validate the entire pipeline
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error")
    search.fit(X_train, y_train)

    # Best model (it's a fitted pipeline ready for prediction)
    best_pipeline = search.best_estimator_
    print("Best hyperparameters found:")
    print(search.best_params_)
    print()

    print("Model trained successfully!")
    # Access the linear regression model inside the pipeline to view coefficients
    final_model = best_pipeline.named_steps["regressor"]
    print(f"Coefficients: {final_model.coef_}")
    print(f"Intercept: {final_model.intercept_}")
    print()

    # Step 4: Make predictions on the test set
    # The pipeline automatically applies the scaler (fitted on train) to the test data
    y_pred = best_pipeline.predict(X_test)

    # Step 5: Evaluate the model
    # Mean Squared Error: average squared difference between predicted and actual
    mse = mean_squared_error(y_test, y_pred)
    # R² Score: proportion of variance explained (1.0 is perfect)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")  # Lower is better
    print(f"R² Score: {r2:.2f}")  # Higher is better (closer to 1.0)
    print()

    # Step 6: Analyze feature importance
    # Absolute coefficients show which features have the biggest impact
    # Use feature names from the original dataframe

    # Get feature names if available (sklearn < 1.0 doesn't support get_feature_names_out on scaler easily,
    # but here we know columns are unchanged by StandardScaler)
    feature_names = X.columns

    feature_importance = pd.DataFrame({"feature": feature_names, "importance": np.abs(final_model.coef_)}).sort_values("importance", ascending=False)

    print("Top 5 Most Important Features:")
    print(feature_importance.head())  # Most influential features
    print()

    # Step 7: Visualize results
    # Scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)  # Plot points
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)  # Perfect prediction line
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted House Prices (Pipeline Tuned)")
    plt.grid(True)
    plt.savefig("linear_regression_results.png", dpi=300, bbox_inches="tight")
    print("Plot saved as 'linear_regression_results.png'")


if __name__ == "__main__":
    main()
