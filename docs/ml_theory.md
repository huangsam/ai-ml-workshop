# ML Theory: Mathematical Foundations

This document covers the mathematical theory underlying machine learning, essential for understanding model behavior and debugging training issues.

## Table of Contents
1. [Calculus & Optimization](#calculus--optimization)
2. [Loss Functions & Regularization](#loss-functions--regularization)
3. [Backpropagation](#backpropagation)
4. [Convergence & Stability](#convergence--stability)
5. [Gradient-Based Optimization](#gradient-based-optimization)

---

## Calculus & Optimization

### Derivatives and Gradients
- **Derivative**: Rate of change of a function with respect to one variable
- **Partial derivative**: Rate of change with respect to one variable in a multivariate function (∂f/∂x)
- **Gradient**: Vector of all partial derivatives ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
- **Gradient direction**: Always points toward steepest increase; negative gradient points toward steepest decrease

### Chain Rule
Fundamental for backpropagation:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

In multivariate form:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

This allows computing gradients through multiple layers by composing local derivatives.

### Optimization Problem
Minimize a loss function:
$$\theta^* = \arg\min_{\theta} L(\theta)$$

Where:
- θ: model parameters
- L(θ): loss function measuring prediction error

---

## Loss Functions & Regularization

### Common Loss Functions

**Mean Squared Error (MSE)** - Regression
$$L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
- Penalizes large errors quadratically
- Sensitive to outliers
- Smooth gradients

**Cross-Entropy Loss** - Classification
$$L_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
- For multiclass: y_i is true label indicator, ŷ_i is predicted probability
- Measures divergence between true and predicted distributions
- Commonly paired with softmax for classification

**Binary Cross-Entropy** - Binary Classification
$$L_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$
- Measures difference for two-class problems
- Well-calibrated probability estimates

### Regularization

**L2 Regularization (Ridge)**
$$L_{total} = L_{data} + \lambda \sum_{j} w_j^2$$
- Penalizes large weights (weight decay)
- Encourages smaller, distributed weights
- More stable gradients, reduces overfitting

**L1 Regularization (Lasso)**
$$L_{total} = L_{data} + \lambda \sum_{j} |w_j|$$
- Promotes sparsity (some weights → 0)
- Feature selection side effect
- Non-differentiable at 0 (subgradient)

**Dropout**
- During training: randomly zero out activations with probability p
- Prevents co-adaptation of neurons
- Acts as ensemble of sub-networks
- Reduce capacity without changing architecture

**Batch Normalization**
- Normalize layer inputs to mean=0, std=1
- Stabilizes training, allows higher learning rates
- Reduces internal covariate shift
- Slight regularization effect

---

## Backpropagation

### Overview
Backpropagation is an efficient algorithm for computing gradients of a loss function with respect to model parameters using the chain rule.

### Forward Pass
Compute predictions through the network:
$$a^{[l]} = g^{[l]}(z^{[l]}) = g^{[l]}(W^{[l]} a^{[l-1]} + b^{[l]})$$

Where:
- z^[l]: pre-activation (linear combination)
- a^[l]: activation (post-nonlinearity)
- g^[l]: activation function

### Backward Pass
Compute gradients layer-by-layer:

**Output layer**:
$$\delta^{[L]} = \nabla_a L \odot g'^{[L]}(z^{[L]})$$

**Hidden layers** (l = L-1, ..., 1):
$$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot g'^{[l]}(z^{[l]})$$

Where ⊙ denotes element-wise multiplication, g' is derivative of activation function.

**Gradient w.r.t. parameters**:
$$\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T$$
$$\frac{\partial L}{\partial b^{[l]}} = \delta^{[l]}$$

### Implementation Considerations

**Gradient verification** (numerical gradient checking):
$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$
- Use ε ≈ 1e-5 for verification
- Compare with analytical gradients
- Catches implementation bugs before training

**Vanishing gradients**: In deep networks, gradients of early layers become very small
- Solution 1: Use ReLU instead of sigmoid/tanh
- Solution 2: Residual connections (skip connections)
- Solution 3: Gradient clipping

**Exploding gradients**: Gradients become very large
- Solution: Gradient clipping by norm
- Solution: Normalize by weight matrices

---

## Convergence & Stability

### Stationary Points
- **Minimum**: Local lowest point, ∇L = 0 and Hessian positive definite
- **Maximum**: Local highest point, ∇L = 0 and Hessian negative definite
- **Saddle point**: ∇L = 0 but neither min nor max

In high dimensions, saddle points are common but SGD typically escapes them.

### Convergence Rate
- **Divergent**: Loss increases, learning rate too high
- **Slow convergence**: Learning rate too low or poor initialization
- **Oscillatory**: Learning rate makes parameter updates overshoot optimum
- **Ideal**: Smooth, steady decrease toward minimum

### Condition Number
For quadratic loss: κ = λ_max / λ_min (ratio of largest to smallest eigenvalues)
- High condition number: slow convergence (ill-conditioned)
- Solution: Feature scaling, preconditioning, adaptive learning rates

---

## Gradient-Based Optimization

### Vanilla Gradient Descent (GD)
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

- η: learning rate (hyperparameter)
- Update based on full dataset gradient
- Stable but slow for large datasets

### Stochastic Gradient Descent (SGD)
$$\theta_{t+1} = \theta_t - \eta \nabla L_i(\theta_t)$$

Where L_i is loss on single sample:
- Fast, noisy gradients provide implicit regularization
- Can escape sharp minima (sharper minima generalize worse)
- Non-uniform learning rate scheduling beneficial

### Mini-Batch SGD
$$\theta_{t+1} = \theta_t - \eta \nabla \bar{L}_{B}(\theta_t)$$

- Gradient over batch B (typically 32-256 samples)
- **Standard practice**: Balances GD stability with computational efficiency
- Good GPU utilization

### Momentum
$$v_t = \beta v_{t-1} + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

- Accelerates gradients in consistent directions
- β typically 0.9 (9:1 weighting of previous vs. current)
- Helps escape shallow local minima
- Momentum builds up in direction of steepest descent

### Nesterov Acceleration
Look ahead before computing gradient:
$$\theta_{t+1} = \theta_t - \eta v_t$$
$$v_t = \beta v_{t-1} + \nabla L(\theta_t - \eta \beta v_{t-1})$$

- Slightly faster convergence than standard momentum
- "Lookahead" effect prevents overshooting

### Adaptive Learning Rates (Adam)
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L)^2$$
$$\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

- m_t: first moment estimate (mean)
- v_t: second moment estimate (variance)
- β₁ = 0.9, β₂ = 0.999, ε = 1e-8 (defaults)
- Per-parameter adaptive learning rates
- Works well across diverse problems with limited tuning

### Learning Rate Scheduling
Reduce learning rate over time:

**Step decay**: Reduce by factor every N epochs
**Cosine annealing**: η(t) = η_min + (η_max - η_min) * (1 + cos(πt/T)) / 2
**Warmup**: Gradually increase learning rate first N epochs (helps stability)

---

## Key Takeaways

1. **Gradient descent finds stationary points**, not guaranteed global minima (NP-hard in general)
2. **Momentum & adaptive methods** reduce manual tuning burden
3. **Feature scaling & initialization** critically affect convergence
4. **Regularization prevents overfitting** by constraining model capacity
5. **Backpropagation** enables efficient gradient computation through chain rule
6. **Batch size** affects both final accuracy (larger = more stable estimates) and training speed
7. **Modern practice**: Adam optimizer, batch normalization, dropout, learning rate scheduling

---

## Further Reading

- Goodfellow et al. "Deep Learning" (2016) - Comprehensive textbook
- Kingma & Ba "Adam: A Method for Stochastic Optimization" (2014)
- Nesterov "Introductory Lectures on Convex Optimization" (2004)
