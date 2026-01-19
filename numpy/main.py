
import numpy as np


def main():
    print("ML Fundamentals with NumPy")
    print("=" * 40)

    # 1. Vector Operations
    print("\n1. Vector Operations")
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"Dot product: {np.dot(v1, v2)}")
    print(f"Vector addition: {v1 + v2}")
    print(f"Scalar multiplication: {2 * v1}")

    # 2. Matrix Operations
    print("\n2. Matrix Operations")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print(f"A:\n{A}")
    print(f"B:\n{B}")
    print(f"Matrix multiplication A*B:\n{np.dot(A, B)}")
    print(f"Matrix transpose A^T:\n{A.T}")
    print(f"Matrix inverse A^-1:\n{np.linalg.inv(A)}")

    # 3. Linear Algebra for ML
    print("\n3. Linear Algebra for ML")

    # Eigenvalues and eigenvectors (used in PCA)
    print("Eigenvalues and eigenvectors:")
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")

    # 4. Linear Regression from Scratch
    print("\n4. Linear Regression from Scratch")

    # Generate sample data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    # Add bias term (x0 = 1)
    X_b = np.c_[np.ones((100, 1)), X]

    # Normal Equation: theta = (X^T * X)^-1 * X^T * y
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(f"Optimal parameters (Normal Equation): {theta_best.ravel()}")

    # Gradient Descent
    print("\nGradient Descent:")
    eta = 0.1  # learning rate
    n_iterations = 1000
    m = 100

    theta = np.random.randn(2, 1)  # random initialization

    for iteration in range(n_iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients

    print(f"Parameters after Gradient Descent: {theta.ravel()}")

    # 5. Cost Function Visualization
    print("\n5. Cost Function (Mean Squared Error)")

    def compute_cost(X, y, theta):
        m = len(y)
        predictions = X.dot(theta)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    # Test cost for different theta values
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            t = np.array([[theta0], [theta1]])
            J_vals[i, j] = compute_cost(X_b, y, t)

    print("Cost function computed for grid of theta values")
    print(f"Min cost: {np.min(J_vals):.4f}")

    # 6. Feature Scaling
    print("\n6. Feature Scaling")

    # Standardization (Z-score normalization)
    def standardize(X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Min-Max scaling
    def min_max_scale(X):
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    sample_data = np.array([[1, 2], [3, 4], [5, 6]])
    print(f"Original data:\n{sample_data}")
    print(f"Standardized:\n{standardize(sample_data)}")
    print(f"Min-Max scaled:\n{min_max_scale(sample_data)}")


if __name__ == "__main__":
    main()
