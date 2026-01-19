"""
ML Fundamentals with NumPy

This script demonstrates core mathematical concepts from machine learning
using NumPy, focusing on the foundations covered in Andrew Ng's Coursera course.

Topics covered:
- Vector and matrix operations
- Linear algebra (eigenvalues, eigenvectors)
- Linear regression (normal equation, gradient descent)
- Cost functions and optimization
- Feature scaling techniques

Each section includes mathematical explanations and practical examples.
"""

import numpy as np


def main():
    print("ML Fundamentals with NumPy")
    print("=" * 40)
    print("Demonstrating core mathematical concepts for machine learning")
    print()

    # ========================================================================
    # 1. VECTOR OPERATIONS
    # ========================================================================
    print("1. VECTOR OPERATIONS")
    print("-" * 30)
    print("Vectors are fundamental data structures in ML:")
    print("- Represent features, weights, and data points")
    print("- Enable efficient mathematical operations")
    print("- Form the basis for higher-dimensional operations")
    print()

    # Create example vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    print(f"v1: {v1} (shape: {v1.shape})")
    print(f"v2: {v2} (shape: {v2.shape})")
    print()

    # Dot product: measures similarity between vectors
    # Formula: v1·v2 = Σ(v1_i * v2_i)
    dot_product = np.dot(v1, v2)
    print(f"Dot product (v1·v2): {dot_product}")
    print("  - Measures vector similarity/alignment")
    print("  - Used in cosine similarity, neural network layers")
    print()

    # Vector addition: element-wise combination
    # Formula: (v1 + v2)_i = v1_i + v2_i
    vector_sum = v1 + v2
    print(f"Vector addition (v1 + v2): {vector_sum}")
    print("  - Combines vectors element-wise")
    print("  - Used in bias addition, residual connections")
    print()

    # Scalar multiplication: scales vector magnitude
    # Formula: (k*v)_i = k * v_i
    scalar_mult = 2 * v1
    print(f"Scalar multiplication (2*v1): {scalar_mult}")
    print("  - Scales vector magnitude uniformly")
    print("  - Used in learning rate application, normalization")
    print()

    # ========================================================================
    # 2. MATRIX OPERATIONS
    # ========================================================================
    print("2. MATRIX OPERATIONS")
    print("-" * 30)
    print("Matrices represent datasets and linear transformations:")
    print("- Rows: samples/instances, Columns: features/dimensions")
    print("- Enable batch operations on multiple vectors")
    print("- Foundation for neural networks and linear algebra")
    print()

    # Create example matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print(f"Matrix A ({A.shape}):\n{A}")
    print(f"Matrix B ({B.shape}):\n{B}")
    print()

    # Matrix multiplication: combines matrices
    # Formula: (A*B)[i,j] = Σ(A[i,k] * B[k,j])
    matrix_prod = np.dot(A, B)
    print(f"Matrix multiplication (A*B):\n{matrix_prod}")
    print("  - Must have compatible dimensions: A(m*n) * B(n*p) = C(m*p)")
    print("  - Used in neural network forward pass, linear transformations")
    print()

    # Matrix transpose: flips matrix along diagonal
    # Formula: (A^T)[i,j] = A[j,i]
    transpose_A = A.T
    print(f"Matrix transpose (A^T):\n{transpose_A}")
    print("  - Flips rows and columns")
    print("  - Used in normal equations, weight matrix operations")
    print()

    # Matrix inverse: finds matrix that undoes multiplication
    # Formula: A^-1 * A = I (identity matrix)
    inverse_A = np.linalg.inv(A)
    print(f"Matrix inverse (A^-1):\n{inverse_A}")
    print("  - Only exists for square, non-singular matrices")
    print("  - Used in normal equation for linear regression")
    print(f"  - Verification (A^-1 * A ≈ I):\n{np.dot(inverse_A, A)}")
    print()

    # ========================================================================
    # 3. LINEAR ALGEBRA FOR ML
    # ========================================================================
    print("3. LINEAR ALGEBRA FOR ML")
    print("-" * 30)
    print("Advanced linear algebra concepts used in ML:")
    print("- Eigenvalues/vectors: PCA, spectral clustering")
    print("- Matrix decompositions: SVD, QR factorization")
    print("- Norms and distances: regularization, similarity measures")
    print()

    # Eigenvalues and eigenvectors
    # For matrix A: A*v = λ*v, where λ is eigenvalue, v is eigenvector
    print("Eigenvalues and eigenvectors:")
    print("  - A*v = λ*v (eigenvalue equation)")
    print("  - Eigenvalues represent variance/scaling factors")
    print("  - Eigenvectors represent principal directions")
    print("  - Applications: PCA, stability analysis, vibration modes")

    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors (columns):\n{eigenvectors}")
    print()

    # Verify eigenvalue equation: A*v = λ*v
    print("Verification of eigenvalue equation A*v = λ*v:")
    for i, (eigenval, eigenvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        Av = np.dot(A, eigenvec)
        lambda_v = eigenval * eigenvec
        print(f"  λ_{i + 1} = {eigenval:.3f}")
        print(f"  A*v_{i + 1} = {Av}")
        print(f"  λ_{i + 1}*v_{i + 1} = {lambda_v}")
        print(f"  Difference: {np.abs(Av - lambda_v)}")
        print()

    # ========================================================================
    # 4. LINEAR REGRESSION FROM SCRATCH
    # ========================================================================
    print("4. LINEAR REGRESSION FROM SCRATCH")
    print("-" * 30)
    print("Linear regression finds the best straight line to fit data:")
    print("- Hypothesis: h_θ(x) = θ₀ + θ₁x₁ + ... + θₙxₙ")
    print("- Cost function: J(θ) = (1/(2m)) * Σ(h_θ(x^(i)) - y^(i))²")
    print("- Goal: minimize J(θ) to find optimal parameters θ")
    print()

    # Generate synthetic data: y = 4 + 3*x + noise
    print("Generating synthetic data: y = 4 + 3*x + noise")
    np.random.seed(42)  # For reproducible results
    X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
    y = 4 + 3 * X + np.random.randn(100, 1)  # True relationship + noise

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    print()

    # Add bias term (x₀ = 1) to feature matrix
    # This allows the intercept θ₀ to be treated as a regular parameter
    X_b = np.c_[np.ones((100, 1)), X]  # Add column of ones
    print(f"X_b shape (with bias): {X_b.shape}")
    print("First 5 rows of X_b (showing bias term):")
    print(X_b[:5])
    print()

    # ========================================================================
    # NORMAL EQUATION METHOD
    # ========================================================================
    print("NORMAL EQUATION METHOD")
    print("-" * 25)
    print("Closed-form solution: θ = (X^T * X)^-1 * X^T * y")
    print("- Advantages: No learning rate tuning, exact solution")
    print("- Disadvantages: O(n³) complexity, doesn't scale to large datasets")
    print("- Best for: Small to medium datasets (< 10,000 features)")
    print()

    # Normal Equation: θ = (X^T * X)^-1 * X^T * y
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print(f"Optimal parameters (Normal Equation): {theta_best.ravel()}")
    print(f"  - θ₀ (intercept): {theta_best[0, 0]:.3f} (true: 4.0)")
    print(f"  - θ₁ (slope): {theta_best[1, 0]:.3f} (true: 3.0)")
    print()

    # ========================================================================
    # GRADIENT DESCENT METHOD
    # ========================================================================
    print("GRADIENT DESCENT METHOD")
    print("-" * 25)
    print("Iterative optimization algorithm:")
    print("- Start with random parameters")
    print("- Compute gradients of cost function")
    print("- Update parameters: θ = θ - η * ∇J(θ)")
    print("- Repeat until convergence")
    print()

    eta = 0.1  # Learning rate (step size)
    n_iterations = 1000  # Maximum iterations
    m = 100  # Number of training examples

    print(f"Learning rate (η): {eta}")
    print(f"Maximum iterations: {n_iterations}")
    print(f"Training examples (m): {m}")
    print()

    # Initialize parameters randomly
    theta = np.random.randn(2, 1)  # 2 parameters: θ₀, θ₁
    print(f"Initial parameters: {theta.ravel()}")

    # Gradient descent loop
    for iteration in range(n_iterations):
        # Compute predictions: h_θ(x) = X_b * θ
        predictions = X_b.dot(theta)

        # Compute errors: h_θ(x^(i)) - y^(i)
        errors = predictions - y

        # Compute gradients: (1/m) * X^T * errors
        gradients = (1 / m) * X_b.T.dot(errors)

        # Update parameters: θ = θ - η * gradients
        theta = theta - eta * gradients

        # Optional: print progress every 100 iterations
        if (iteration + 1) % 200 == 0:
            cost = (1 / (2 * m)) * np.sum(errors**2)
            print(f"  Iteration {iteration + 1:4d}: Cost = {cost:.4f}, θ = {theta.ravel()}")

    print(f"\nFinal parameters after Gradient Descent: {theta.ravel()}")
    print(f"  - θ₀ (intercept): {theta[0, 0]:.3f} (true: 4.0)")
    print(f"  - θ₁ (slope): {theta[1, 0]:.3f} (true: 3.0)")
    print()

    # ========================================================================
    # 5. COST FUNCTION VISUALIZATION
    # ========================================================================
    print("5. COST FUNCTION VISUALIZATION")
    print("-" * 35)
    print("Mean Squared Error (MSE) cost function:")
    print("- J(θ) = (1/(2m)) * Σ(h_θ(x^(i)) - y^(i))²")
    print("- Measures average squared difference between predictions and actual values")
    print("- Convex function: single global minimum")
    print("- Gradient points toward the minimum")
    print()

    def compute_cost(X, y, theta):
        """
        Compute Mean Squared Error cost function.

        Parameters:
        X: Feature matrix (m * n)
        y: Target values (m * 1)
        theta: Parameters (n * 1)

        Returns:
        cost: Scalar cost value
        """
        m = len(y)
        predictions = X.dot(theta)
        errors = predictions - y
        cost = (1 / (2 * m)) * np.sum(errors**2)
        return cost

    # Create grid of theta values to visualize cost function
    theta0_vals = np.linspace(-10, 10, 100)  # θ₀ values
    theta1_vals = np.linspace(-1, 4, 100)  # θ₁ values
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    # Compute cost for each combination of θ₀, θ₁
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            t = np.array([[theta0], [theta1]])
            J_vals[i, j] = compute_cost(X_b, y, t)

    print("Cost function analysis:")
    print(f"  - Grid size: {len(theta0_vals)} * {len(theta1_vals)} = {J_vals.size} points")
    print(f"  - Minimum cost: {np.min(J_vals):.4f}")
    print(f"  - Maximum cost: {np.max(J_vals):.4f}")
    print(f"  - Cost range: {np.max(J_vals) - np.min(J_vals):.4f}")
    print()

    # Find optimal parameters from grid search
    min_idx = np.unravel_index(np.argmin(J_vals), J_vals.shape)
    optimal_theta0 = theta0_vals[min_idx[0]]
    optimal_theta1 = theta1_vals[min_idx[1]]
    print("Grid search optimal parameters:")
    print(f"  - θ₀: {optimal_theta0:.3f}")
    print(f"  - θ₁: {optimal_theta1:.3f}")
    print(f"  - Cost: {np.min(J_vals):.4f}")
    print()

    # ========================================================================
    # 6. FEATURE SCALING
    # ========================================================================
    print("6. FEATURE SCALING")
    print("-" * 20)
    print("Feature scaling ensures all features contribute equally to the model:")
    print("- Prevents features with larger ranges from dominating")
    print("- Critical for gradient descent convergence")
    print("- Required for distance-based algorithms (KNN, K-means)")
    print("- Two main techniques: standardization and min-max scaling")
    print()

    def standardize(X):
        """
        Standardization (Z-score normalization).
        Centers data around mean with unit variance.

        Formula: x_scaled = (x - μ) / σ
        - μ: mean of feature
        - σ: standard deviation of feature

        Result: mean = 0, std = 1
        """
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def min_max_scale(X):
        """
        Min-Max scaling (Normalization).
        Scales to fixed range [0, 1].

        Formula: x_scaled = (x - min) / (max - min)

        Result: values in range [0, 1]
        """
        return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

    # Create sample data to demonstrate scaling
    sample_data = np.array([[1, 2], [3, 4], [5, 6]])
    print("Sample data (3 samples, 2 features):")
    print(sample_data)
    print()

    print("Feature statistics (before scaling):")
    print(
        f"  - Feature 1: mean={np.mean(sample_data[:, 0]):.2f}, std={np.std(sample_data[:, 0]):.2f}, range=[{np.min(sample_data[:, 0])}, {np.max(sample_data[:, 0])}]"
    )
    print(
        f"  - Feature 2: mean={np.mean(sample_data[:, 1]):.2f}, std={np.std(sample_data[:, 1]):.2f}, range=[{np.min(sample_data[:, 1])}, {np.max(sample_data[:, 1])}]"
    )
    print()

    # Apply standardization
    standardized = standardize(sample_data)
    print("After STANDARDIZATION (Z-score):")
    print(standardized)
    print(f"  - Feature 1: mean={np.mean(standardized[:, 0]):.2f}, std={np.std(standardized[:, 0]):.2f}")
    print(f"  - Feature 2: mean={np.mean(standardized[:, 1]):.2f}, std={np.std(standardized[:, 1]):.2f}")
    print("  - Preserves shape, centers at 0, unit variance")
    print()

    # Apply min-max scaling
    minmax_scaled = min_max_scale(sample_data)
    print("After MIN-MAX SCALING:")
    print(minmax_scaled)
    print(f"  - Feature 1: range=[{np.min(minmax_scaled[:, 0]):.2f}, {np.max(minmax_scaled[:, 0]):.2f}]")
    print(f"  - Feature 2: range=[{np.min(minmax_scaled[:, 1]):.2f}, {np.max(minmax_scaled[:, 1]):.2f}]")
    print("  - Scales to [0, 1] range")
    print()

    print("WHEN TO USE EACH METHOD:")
    print("- Standardization: When data follows normal distribution, for gradient descent")
    print("- Min-Max Scaling: When you need bounded ranges, for neural networks")
    print("- Both preserve relationships between data points")
    print("- Always fit scaler on training data, apply to test data")


if __name__ == "__main__":
    main()
