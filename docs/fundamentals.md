# ML Fundamentals: Theory & Math from Scratch

This document covers the mathematical foundations of machine learning and their implementation using NumPy. These concepts are essential for understanding how models learn and for debugging training issues.

---

## Calculus & Optimization

### Derivatives and Gradients
- **Derivative**: Rate of change of a function with respect to one variable.
- **Partial derivative**: Rate of change with respect to one variable in a multivariate function ($\partial f/\partial x$).
- **Gradient**: Vector of all partial derivatives $\nabla f = [\partial f/\partial x_1, \partial f/\partial x_2, ..., \partial f/\partial x_n]$.
- **Gradient direction**: Always points toward steepest increase; negative gradient points toward steepest decrease.

### The Chain Rule
Enables gradient computation through nested function compositions. **Critical for backpropagation in neural networks.**
- **Simple Form**: If $y = f(g(x))$, then $dy/dx = (dy/dg) \cdot (dg/dx)$.
- **Multi-layer Case**: $dJ/d\theta_1 = (dJ/da_n) \cdot (da_n/dz_n) \cdot ... \cdot (da_2/dz_2) \cdot (dz_2/d\theta_1)$.

### Optimization Algorithms
- **Gradient Descent**: Iterative algorithm to find the minimum of a cost function by moving in the direction of the negative gradient.
- **Normal Equation**: Closed-form solution for linear regression: $\theta = (X^T X)^{-1} X^T y$. Efficient for small datasets but scales poorly ($O(n^3)$).

---

## Numerical Foundations (NumPy)

### Linear Algebra
- **Dot Product**: Measures similarity between vectors and performs weighted sums.
- **Matrix Multiplication**: Transforms data through layers of a network.
- **Eigenvalues/vectors**: Critical for dimensionality reduction (PCA). Eigenvalues represent variance explained.

### Feature Scaling
- **Standardization (Z-score)**: Centers data around mean=0 with std=1.
- **Min-Max Scaling**: Scales to a fixed range (usually [0, 1]).
- **Why?**: Prevents features with large scales from dominating the gradient updates and speeds up convergence.

---

## Loss Functions & Regularization

### Common Loss Functions
- **Mean Squared Error (MSE)**: Standard for regression. Penalizes large errors quadratically.
- **Cross-Entropy Loss**: Standard for classification. Measures divergence between true and predicted distributions.

### Regularization
- **L2 Regularization (Ridge)**: Adds penalty $\lambda \sum w_j^2$. Encourages small, distributed weights to reduce overfitting.
- **L1 Regularization (Lasso)**: Adds penalty $\lambda \sum |w_j|$. Promotes sparsity, acting as automated feature selection.
- **Dropout**: Randomly deactivates neurons during training to prevent co-adaptation.
- **Batch Normalization**: Normalizes layer inputs to stabilize training and allow higher learning rates.

---

## Backpropagation Deep Dive

Backpropagation computes gradients of the loss function with respect to model parameters using the chain rule, moving backward from output to input.

### Implementation Pattern (From Scratch)
1. **Forward Pass**: Compute and store intermediate linear values ($z$) and activations ($a$).
2. **Output Gradient**: Calculate error relative to the prediction ($\delta^{[L]}$).
3. **Backward Prop**: Recursively compute $\delta^{[l]}$ for hidden layers using weights from the next layer.
4. **Parameter Updates**: Use gradients to adjust weights: $W = W - \eta \cdot \nabla W$.

### Vanishing & Exploding Gradients
- **Vanishing**: Gradients become too small in deep networks (mitigated by ReLU and skip connections).
- **Exploding**: Gradients become too large (mitigated by gradient clipping).

---

## Core Implementations

### Linear Regression
Implemented using both the **Normal Equation** for small-scale accuracy and **Gradient Descent** for scalability. Demonstrates the importance of vectorization handle all training examples simultaneously.

### Neural Network (XOR Problem)
A 2-layer network implementation demonstrating manual backpropagation. Solving XOR highlights why non-linear activation functions (Sigmoid/ReLU) and hidden layers are necessary for non-linearly separable data.

### Principal Component Analysis (PCA)
Uses eigenvalue decomposition to project data onto principal components. Focuses on variance preservation and dimensionality reduction for visualization and noise reduction.

### Gradient Verification
Uses finite differences: $\partial J/\partial \theta \approx \frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2\epsilon}$.
Essential for debugging custom gradient implementations by comparing analytical results with numerical approximations.

---

## Connection to Modern Frameworks
While tools like PyTorch handle Autograd, understanding these foundations is critical for:
- Debugging training instability (oscillations, divergence).
- Designing custom loss functions or layers.
- Optimizing inference performance and memory layout.
