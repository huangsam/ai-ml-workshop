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

## Calculus Foundations for Machine Learning

### Partial Derivatives
In ML, cost functions depend on multiple parameters. Partial derivatives measure how cost changes with respect to each parameter independently.

**Definition**: `∂J/∂θⱼ` = rate of change of cost J with respect to parameter θⱼ

**Example in Linear Regression**:
```
J(θ₀, θ₁) = (1/(2m)) * Σ(θ₀ + θ₁x^(i) - y^(i))²

∂J/∂θ₀ = (1/m) * Σ(θ₀ + θ₁x^(i) - y^(i))
∂J/∂θ₁ = (1/m) * Σ(θ₀ + θ₁x^(i) - y^(i)) * x^(i)
```

**Intuition**:
- Positive gradient: Increase parameters to decrease cost
- Negative gradient: Decrease parameters to decrease cost
- Zero gradient: At critical point (local minimum/maximum)

### The Chain Rule
Enables gradient computation through nested function compositions. **Critical for backpropagation in neural networks.**

**Simple Form**: If `y = f(g(x))`, then `dy/dx = (dy/dg) * (dg/dx)`

**Example**: Sigmoid composed with linear function
```
z = θ^T * x           (linear layer)
a = sigmoid(z)        (activation)
J = loss(a, y)        (cost function)

dJ/dθ = (dJ/da) * (da/dz) * (dz/dθ)
      = (dJ/da) * sigmoid'(z) * x
```

**Multi-layer Case**: For deep networks, chain rule applies recursively
```
dJ/dθ₁ = (dJ/da_n) * (da_n/dz_n) * ... * (da₂/dz₂) * (dz₂/dθ₁)
```

This is **backpropagation**: computing gradients from output to input.

### Gradient Verification
Before training on large datasets, verify gradients are correct using **numerical gradient checking**.

**Finite Differences Approximation**:
```
∂J/∂θ ≈ (J(θ + ε) - J(θ - ε)) / (2ε)    [where ε ≈ 1e-5]
```

**Verification Process**:
1. Compute analytical gradient: `dθ_analytical = backprop()`
2. Compute numerical gradient: `dθ_numerical = finite_differences()`
3. Check relative error: `|dθ_analytical - dθ_numerical| / (|dθ_analytical| + |dθ_numerical|) < 1e-4`
4. If passes, implementation is correct; if fails, debug backprop code

**Trade-offs**:
- Analytical gradients: Fast, scalable, but complex to implement
- Numerical gradients: Slow (requires 2m forward passes), but simple and reliable for verification

## Gradient Descent Variants

### Batch Gradient Descent (BGD)
Uses entire training set for each update.

**Update Rule**: `θ = θ - η * ∇J(θ)`

**Pros**:
- Smooth convergence (each step uses all data)
- Can use larger learning rates
- Better for parallel computation

**Cons**:
- Slow: Must process all m examples before each update
- Memory intensive for large datasets
- May get stuck in local minima

### Stochastic Gradient Descent (SGD)
Updates after each single example.

**Update Rule**: `θ = θ - η * ∇J(θ; x^(i), y^(i))`

**Pros**:
- Fast: Updates immediately after each example
- Can escape local minima (noise helps)
- Online learning possible (streaming data)

**Cons**:
- Noisy convergence (oscillates around minimum)
- Harder to parallelize
- Requires learning rate scheduling

### Mini-Batch Gradient Descent
Compromise: Update after b training examples (batch size).

**Update Rule**: `θ = θ - η * (1/b) * Σ∇J(θ; x^(i), y^(i))` for i ∈ batch

**Pros**:
- Best of both worlds: less noisy than SGD, faster than BGD
- Parallelizable (process batch on GPU)
- Practical default for deep learning

**Cons**:
- Introduces batch size as hyperparameter
- Still requires learning rate tuning

**Modern Practice**: Mini-batch is standard in frameworks like PyTorch/TensorFlow

### Advanced Optimizers
**Momentum**:
- Accumulates gradients over time: `v = βv + ∇J`
- Accelerates in consistent directions, dampens oscillations
- Default β ≈ 0.9

**RMSprop**:
- Adapts learning rate per parameter: `θ = θ - η/(sqrt(v) + ε) * ∇J`
- Helps with features at different scales

**Adam (Adaptive Moment Estimation)**:
- Combines momentum and RMSprop
- Maintains both first moment (mean) and second moment (variance) of gradients
- Most popular optimizer in deep learning

## Backpropagation Deep Dive

### Computational Graph
Visual representation of nested function composition.

**Example**: `L = loss(sigmoid(linear(x)))`
```
x → [Linear: z = Wx + b] → z → [Sigmoid: a = 1/(1+e^-z)] → a → [Loss: J] → L
                                 ↓                           ↓
                         (da/dz = a(1-a))        (dJ/da = (a-y)/m)
```

**Forward Pass**: Compute each intermediate value (z, a, L)
**Backward Pass**: Compute gradients using chain rule in reverse order

### Backpropagation Algorithm

**Forward Pass**: Store all intermediate values
```python
z1 = W1 @ x + b1
a1 = relu(z1)
z2 = W2 @ a1 + b2
a2 = sigmoid(z2)
L = binary_cross_entropy(a2, y)
```

**Backward Pass**: Compute gradients from output to input using chain rule
```python
# Output layer gradient
dL/da2 = (a2 - y) / m                    # Loss derivative

# Sigmoid gradient
da2/dz2 = a2 * (1 - a2)                  # Activation derivative
dL/dz2 = dL/da2 * da2/dz2

# Weight gradients
dL/dW2 = dL/dz2 @ a1.T                   # Chain rule: dz2/dW2 = a1^T
dL/db2 = sum(dL/dz2)

# Backprop to hidden layer
dL/da1 = dL/dz2 @ W2.T                   # Chain rule: dz2/da1 = W2^T

# ReLU gradient
da1/dz1 = (z1 > 0)                       # Step function
dL/dz1 = dL/da1 * da1/dz1

# Hidden layer weight gradients
dL/dW1 = x.T @ dL/dz1
dL/db1 = sum(dL/dz1)
```

**Parameter Update**:
```python
W1 = W1 - learning_rate * dL/dW1
b1 = b1 - learning_rate * dL/db1
W2 = W2 - learning_rate * dL/dW2
b2 = b2 - learning_rate * dL/db2
```

### Why Backpropagation Works

**Efficiency**: Computes all gradients in O(1) backward pass vs O(n_params) forward passes
**Correctness**: Chain rule guarantees mathematically correct gradients
**Generality**: Works for any differentiable function composition

**Key Insight**: Gradients propagate information about how to improve predictions through all layers simultaneously.

### Why This Matters Today

- **PyTorch/TensorFlow**: Autograd automates backprop; understanding it helps debug training
- **Custom Layers**: Implementing gradients for novel architectures requires backprop knowledge
- **Numerical Issues**: Understanding vanishing/exploding gradients helps with network design
- **Transfer Learning**: Fine-tuning involves backprop with frozen layers

## Regularization Theory

### Why Regularization?

**Overfitting Problem**: Model memorizes training data, poor generalization

**Example**:
- Training error: 1%
- Test error: 20% ← Problem!

**Solution**: Add penalty term to cost function that discourages large weights

### L1 Regularization (Lasso)
**Cost Function**: `J(θ) = (1/(2m)) * Σ(h_θ(x^(i)) - y^(i))² + (λ/(2m)) * Σ|θ|`

**Properties**:
- Penalty proportional to absolute value of weights
- Encourages **sparsity**: many weights become exactly zero
- Feature selection effect

**Gradient**: `∂J/∂θⱼ = ... + (λ/m) * sign(θⱼ)`

### L2 Regularization (Ridge)
**Cost Function**: `J(θ) = (1/(2m)) * Σ(h_θ(x^(i)) - y^(i))² + (λ/(2m)) * Σ θⱼ²`

**Properties**:
- Penalty proportional to squared weights
- Encourages small weights, but rarely zero
- More stable than L1

**Gradient**: `∂J/∂θⱼ = ... + (λ/m) * θⱼ`

### Hyperparameter: Regularization Strength (λ)
- **λ = 0**: No regularization (may overfit)
- **λ = optimal**: Best generalization
- **λ = very large**: Underfitting (all weights → 0)

**Selection**: Use cross-validation to find best λ

### Dropout (Deep Learning)
Random deactivation of neurons during training.

**Benefits**:
- Prevents co-adaptation of neurons
- Acts as ensemble of models
- Effective regularization for deep networks

**Implementation**: During training, drop each neuron with probability p; scale activations by 1/(1-p) during inference

## Summary: From Theory to Practice

| Concept | Theory | Implementation |
|---------|--------|-----------------|
| **Optimization** | Gradient descent minimizes cost | SGD/Adam in PyTorch |
| **Gradients** | Partial derivatives via chain rule | Backpropagation/Autograd |
| **Feature Scaling** | Normalization for stability | StandardScaler/MinMaxScaler |
| **Regularization** | Penalize complexity | L1/L2 loss terms, Dropout |
| **Evaluation** | Cost functions measure fit | Loss/accuracy metrics |

See [workshop/core/numpy/backpropagation.py](../workshop/core/numpy/backpropagation.py) for a complete implementation demonstrating these concepts.

---

## Project 5: ML Fundamentals with NumPy

This project implements core ML concepts from scratch using NumPy:

### Linear Algebra Foundations
- Vector and matrix operations (dot products, transpose, inverse)
- Eigenvalue decomposition for PCA
- Matrix multiplication and reshaping for data manipulation
- Broadcasting for efficient vectorized operations

### Cost Functions & Optimization
- **Mean Squared Error (MSE)**: Regression loss function
- **Gradient computation**: Partial derivatives for each parameter
- **Gradient descent**: Iterative optimization using learning rates
- **Normal equation**: Closed-form solution for linear regression

### Feature Scaling
- **Standardization**: Zero mean, unit variance (z-score normalization)
- **Min-Max scaling**: Scale to [0, 1] range
- **Importance**: Prevents features with large scales from dominating

### Linear Regression Implementation
- Normal equation approach: θ = (X^T X)^(-1) X^T y
- Gradient descent approach: θ := θ - α∇J(θ)
- Comparison of convergence and computational complexity
- Feature engineering and polynomial features

### Principal Component Analysis (PCA)
- Eigenvalue/eigenvector computation from covariance matrix
- Variance explained by each principal component
- Dimensionality reduction while preserving information
- Applications: visualization, feature extraction, noise reduction

---

## Bonus: Backpropagation from Scratch

This bonus project implements a neural network with manual backpropagation to understand gradient computation:

### Neural Network Architecture
- 2-layer neural network: Input → Hidden (sigmoid) → Output (linear)
- Trained on XOR problem (non-linearly separable)
- Manual weight initialization and forward propagation

### Backpropagation Algorithm
- Forward pass: Compute layer-by-layer activations
- Loss computation: MSE loss for the problem
- Backward pass: Compute gradients using chain rule
  - Output layer: ∂L/∂z = (ŷ - y) ⊙ g'(z)
  - Hidden layer: ∂L/∂z = (W^T δ) ⊙ g'(z)
  - Parameter gradients: ∂L/∂W = δ a^T

### Gradient Verification
- **Numerical gradient** using finite differences: (f(θ+ε) - f(θ-ε)) / (2ε)
- **Analytical gradient** from backpropagation
- Verification with ε ≈ 1e-5 to catch implementation bugs
- Essential for debugging custom neural network implementations

### Key Insights
- Gradient computation through composition of functions (chain rule)
- ReLU activations prevent vanishing gradient problems
- Proper weight initialization crucial for training stability
- XOR problem requires hidden layer (linear separator insufficient)
- Learning curves show convergence behavior and potential overfitting

### Files
- `workshop/core/numpy/main.py`: Linear algebra operations and concepts
- `workshop/core/numpy/backpropagation.py`: Complete neural network with manual backprop
