# NumPy ML Fundamentals

This document covers the mathematical foundations of machine learning implemented using NumPy, focusing on concepts from Andrew Ng's Coursera Machine Learning course. These implementations demonstrate the core math behind ML algorithms without relying on high-level libraries.

## Linear Algebra Operations

### Vector Operations
Vectors are fundamental to ML for representing features, weights, and data points.

**Key Operations**:
- **Dot Product**: `np.dot(v1, v2)` - Measures similarity between vectors
- **Vector Addition**: `v1 + v2` - Combines vectors element-wise
- **Scalar Multiplication**: `k * v` - Scales vector magnitude

**ML Applications**:
- Feature vectors in datasets
- Weight vectors in neural networks
- Similarity calculations in KNN

### Matrix Operations
Matrices represent datasets and transformations in ML.

**Key Operations**:
- **Matrix Multiplication**: `np.dot(A, B)` - Combines matrices (must be compatible dimensions)
- **Transpose**: `A.T` - Flips matrix along diagonal
- **Inverse**: `np.linalg.inv(A)` - Finds matrix that undoes multiplication (for square matrices)

**ML Applications**:
- Dataset representation (rows = samples, columns = features)
- Linear transformations
- Normal equation in linear regression

### Eigenvalues and Eigenvectors
Critical for dimensionality reduction techniques like PCA.

**Computation**: `eigenvalues, eigenvectors = np.linalg.eig(A)`
- Eigenvalues represent variance explained
- Eigenvectors represent principal directions

**ML Applications**:
- Principal Component Analysis (PCA)
- Understanding data variance
- Spectral clustering

## Optimization Algorithms

### Gradient Descent
Iterative optimization algorithm that finds minimum of cost function.

**Algorithm**:
```python
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
```

**Key Parameters**:
- **Learning Rate (η)**: Step size (too small = slow, too large = unstable)
- **Iterations**: Number of updates
- **Convergence**: When gradients become very small

**ML Applications**:
- Training neural networks
- Optimizing linear/logistic regression
- Any differentiable loss function

### Normal Equation
Closed-form solution for linear regression (no iteration needed).

**Formula**: `θ = (X^T * X)^-1 * X^T * y`
- Computationally expensive for large datasets (O(n³))
- No learning rate tuning required
- Always finds optimal solution (for linear regression)

## Cost Functions

### Mean Squared Error (MSE)
Most common cost function for regression problems.

**Formula**: `J(θ) = (1/(2m)) * Σ(h_θ(x^(i)) - y^(i))²`
- **h_θ(x)**: Hypothesis/prediction function
- **m**: Number of training examples
- **1/2m**: Scaling factor (makes derivative cleaner)

**Properties**:
- Always non-negative
- Convex function (single global minimum)
- Penalizes large errors more (squared term)

**ML Applications**:
- Linear regression training
- Neural network regression layers
- Evaluating regression model performance

## Feature Scaling

### Standardization (Z-score Normalization)
Centers data around mean with unit variance.

**Formula**: `x_scaled = (x - μ) / σ`
- **μ**: Mean of feature
- **σ**: Standard deviation of feature

**Properties**:
- Mean = 0, Standard deviation = 1
- Preserves shape of distribution
- Sensitive to outliers

### Min-Max Scaling
Scales to fixed range (usually 0-1).

**Formula**: `x_scaled = (x - min) / (max - min)`

**Properties**:
- Bounded range [0,1]
- Preserves relationships between data points
- Very sensitive to outliers

**ML Applications**:
- Required for gradient descent convergence
- Distance-based algorithms (KNN, K-means)
- Neural network inputs
- PCA and other matrix factorization methods

## Linear Regression from Scratch

### Mathematical Foundation
**Hypothesis**: `h_θ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ`
**Cost Function**: `J(θ) = (1/(2m)) * Σ(h_θ(x^(i)) - y^(i))²`
**Gradient**: `∂J/∂θⱼ = (1/m) * Σ(h_θ(x^(i)) - y^(i)) * xⱼ^(i)`

### Implementation Steps
1. **Data Preparation**: Add bias term (x₀ = 1) to feature matrix
2. **Initialize Parameters**: Random or zeros
3. **Compute Gradients**: Vectorized calculation using matrix operations
4. **Update Parameters**: `θ = θ - η * gradients`
5. **Convergence Check**: Monitor cost function decrease

### Key Insights
- **Vectorization**: Matrix operations handle all training examples simultaneously
- **Bias Term**: Allows model to fit non-zero intercepts
- **Feature Scaling**: Critical for gradient descent convergence
- **Learning Rate**: Must be tuned for stable convergence

## Practical Considerations

### Computational Complexity
- **Normal Equation**: O(n³) - Suitable for small datasets (< 10,000 features)
- **Gradient Descent**: O(m*n) per iteration - Scales to large datasets
- **Matrix Inversion**: Numerically unstable for singular matrices

### Numerical Stability
- **Feature Scaling**: Prevents overflow/underflow in computations
- **Regularization**: Adds stability to matrix inversion
- **Convergence Monitoring**: Track cost function to detect issues

### Memory Efficiency
- **Vectorization**: Single matrix operations vs element-wise loops
- **In-place Operations**: Modify arrays without creating copies
- **Data Types**: Use appropriate precision (float32 vs float64)

## Connection to Modern ML

### Deep Learning Bridge
- **Backpropagation**: Extends gradient descent to multi-layer networks
- **Activation Functions**: Non-linear transformations between layers
- **Chain Rule**: Computes gradients through complex function compositions

### Advanced Topics
- **Regularization**: L1/L2 penalties prevent overfitting
- **Stochastic Gradient Descent**: Updates using mini-batches
- **Adam/Momentum**: Advanced optimization algorithms
- **Automatic Differentiation**: Frameworks like PyTorch handle gradient computation

This foundation provides the mathematical intuition needed to understand and debug modern ML algorithms, even when using high-level libraries like scikit-learn or deep learning frameworks.
