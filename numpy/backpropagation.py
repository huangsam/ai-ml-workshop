"""
Backpropagation from Scratch with NumPy

This script demonstrates the mathematics of backpropagation step-by-step.
It shows how gradients flow backward through a neural network using the chain rule.

Key concepts:
- Computational graphs and forward/backward passes
- Chain rule for computing gradients
- Gradient verification (numerical vs analytical)
- How PyTorch/TensorFlow automates this process
"""

from typing import Tuple

import matplotlib.pyplot as plt

import numpy as np

# --- 1. SIMPLE NEURAL NETWORK ---


class SimpleNeuralNetwork:
    """
    A minimal 2-layer neural network for teaching backpropagation.

    Architecture:
        Input (2) -> Hidden (4) -> Output (1)
        y_pred = sigmoid(W2 @ relu(W1 @ x + b1) + b2)
    """

    def __init__(self, input_size: int = 2, hidden_size: int = 4, output_size: int = 1, seed: int = 42):
        """
        Initialize network with random weights.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        # Xavier initialization for better convergence
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # Store intermediate values for backward pass
        self.cache = {}

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU: 1 if x > 0, else 0."""
        return (x > 0).astype(float)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip to avoid overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))."""
        return x * (1.0 - x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input data of shape (batch_size, input_size)

        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # Hidden layer: z1 = x @ W1 + b1
        self.cache["z1"] = np.dot(x, self.W1) + self.b1
        # Hidden activation: a1 = relu(z1)
        self.cache["a1"] = self.relu(self.cache["z1"])

        # Output layer: z2 = a1 @ W2 + b2
        self.cache["z2"] = np.dot(self.cache["a1"], self.W2) + self.b2
        # Output activation: a2 = sigmoid(z2)
        self.cache["a2"] = self.sigmoid(self.cache["z2"])

        # Store inputs for backward pass
        self.cache["x"] = x

        return self.cache["a2"]

    def backward(self, y_true: np.ndarray, learning_rate: float = 0.01) -> Tuple[dict, float]:
        """
        Backward pass using backpropagation algorithm.

        Applies chain rule: dL/dW = dL/da * da/dz * dz/dW

        Args:
            y_true: True labels of shape (batch_size, 1)
            learning_rate: Learning rate for gradient descent

        Returns:
            Tuple of (gradients_dict, loss)
        """
        batch_size = self.cache["x"].shape[0]

        # --- STEP 1: Compute loss and output layer gradient ---
        # Binary Cross-Entropy Loss: L = -y*log(y_pred) - (1-y)*log(1-y_pred)
        # Gradient w.r.t output: dL/da2 = (y_pred - y_true) / batch_size
        y_pred = self.cache["a2"]
        loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

        # dL/da2: gradient of loss w.r.t output activation
        dL_da2 = (y_pred - y_true) / batch_size

        # --- STEP 2: Backprop through sigmoid ---
        # da2/dz2 = sigmoid'(z2) = a2 * (1 - a2)
        # dL/dz2 = dL/da2 * da2/dz2
        da2_dz2 = self.sigmoid_derivative(y_pred)
        dL_dz2 = dL_da2 * da2_dz2

        # --- STEP 3: Compute W2 and b2 gradients ---
        # dL/dW2 = dL/dz2 @ a1.T (using matrix multiplication)
        dL_dW2 = np.dot(self.cache["a1"].T, dL_dz2)
        # dL/db2 = sum(dL/dz2) along batch dimension
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        # --- STEP 4: Backprop to hidden layer ---
        # dL/da1 = dL/dz2 @ W2.T (chain rule through W2)
        dL_da1 = np.dot(dL_dz2, self.W2.T)

        # --- STEP 5: Backprop through ReLU ---
        # da1/dz1 = relu'(z1) = 1 if z1 > 0 else 0
        # dL/dz1 = dL/da1 * da1/dz1
        da1_dz1 = self.relu_derivative(self.cache["z1"])
        dL_dz1 = dL_da1 * da1_dz1

        # --- STEP 6: Compute W1 and b1 gradients ---
        # dL/dW1 = x.T @ dL/dz1
        dL_dW1 = np.dot(self.cache["x"].T, dL_dz1)
        # dL/db1 = sum(dL/dz1) along batch dimension
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Store gradients
        gradients = {"dW1": dL_dW1, "db1": dL_db1, "dW2": dL_dW2, "db2": dL_db2}

        # --- STEP 7: Update weights using gradient descent ---
        # W = W - learning_rate * dL/dW
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2

        return gradients, loss

    def numerical_gradient(self, x: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-5) -> dict:
        """
        Compute numerical gradients for verification (finite differences).

        This is slower but useful to verify analytical gradients are correct.

        Args:
            x: Input data
            y_true: True labels
            epsilon: Small perturbation for finite differences

        Returns:
            Dictionary of numerical gradients
        """
        numerical_grads = {}

        # Check W1
        numerical_grads["dW1"] = np.zeros_like(self.W1)
        for i in range(self.W1.shape[0]):
            for j in range(self.W1.shape[1]):
                # f(W + eps)
                self.W1[i, j] += epsilon
                y_plus = self.forward(x)
                loss_plus = -np.mean(y_true * np.log(y_plus + 1e-8) + (1 - y_true) * np.log(1 - y_plus + 1e-8))

                # f(W - eps)
                self.W1[i, j] -= 2 * epsilon
                y_minus = self.forward(x)
                loss_minus = -np.mean(y_true * np.log(y_minus + 1e-8) + (1 - y_true) * np.log(1 - y_minus + 1e-8))

                # Gradient = (f(x+eps) - f(x-eps)) / (2*eps)
                numerical_grads["dW1"][i, j] = (loss_plus - loss_minus) / (2 * epsilon)

                # Restore original value
                self.W1[i, j] += epsilon

        # For brevity, only check W1 (same process applies to other parameters)
        return numerical_grads


def synthetic_xor_data(num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate XOR dataset (classic non-linear problem).

    Args:
        num_samples: Number of samples per class

    Returns:
        Tuple of (X, y) where X is input and y is binary labels
    """
    np.random.seed(42)

    # Class 1: top-left and bottom-right (XOR positive)
    class1_a = np.random.randn(num_samples, 2) * 0.3 + np.array([-1, 1])
    class1_b = np.random.randn(num_samples, 2) * 0.3 + np.array([1, -1])
    class1 = np.vstack([class1_a, class1_b])

    # Class 0: top-right and bottom-left (XOR negative)
    class0_a = np.random.randn(num_samples, 2) * 0.3 + np.array([1, 1])
    class0_b = np.random.randn(num_samples, 2) * 0.3 + np.array([-1, -1])
    class0 = np.vstack([class0_a, class0_b])

    X = np.vstack([class1, class0])
    y = np.vstack([np.ones((len(class1), 1)), np.zeros((len(class0), 1))])

    # Shuffle
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def main() -> None:
    """
    Main demonstration of backpropagation.
    """
    print("=" * 70)
    print("Backpropagation from Scratch")
    print("=" * 70)

    # 1. Generate data
    print("\n1. Generating XOR dataset...")
    X, y = synthetic_xor_data(num_samples=50)
    print(f"   Dataset shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")

    # 2. Create network
    print("\n2. Creating neural network...")
    net = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    print(f"   W1 shape: {net.W1.shape}")
    print(f"   W2 shape: {net.W2.shape}")

    # 3. Training
    print("\n3. Training with backpropagation...")
    num_epochs = 100
    losses = []
    learning_rate = 0.1

    for epoch in range(num_epochs):
        # Forward pass
        net.forward(X)

        # Backward pass (computes analytical gradients)
        gradients, loss = net.backward(y, learning_rate=learning_rate)
        losses.append(loss)

        if (epoch + 1) % 20 == 0:
            # Compute numerical gradients for first batch (verification)
            if epoch == 19:
                print(f"   Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.6f}")
                print("\n4. Gradient Verification (Analytical vs Numerical)...")

                X_batch = X[:10]
                y_batch = y[:10]

                # Reset network state for numerical gradient
                net.forward(X_batch)
                numerical_grads = net.numerical_gradient(X_batch, y_batch)
                net.forward(X_batch)
                gradients, _ = net.backward(y_batch, learning_rate=0.0)  # No update

                # Compare
                analytical_dW1 = gradients["dW1"]
                numerical_dW1 = numerical_grads["dW1"]

                diff = np.abs(analytical_dW1 - numerical_dW1)
                rel_error = diff / (np.abs(analytical_dW1) + np.abs(numerical_dW1) + 1e-8)

                print(f"   Max absolute difference: {np.max(diff):.2e}")
                print(f"   Mean relative error: {np.mean(rel_error):.2e}")
                print("   ✓ Gradients match! (error < 1e-4)" if np.mean(rel_error) < 1e-4 else "   ✗ Gradient mismatch detected")
            else:
                print(f"   Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.6f}")

    # 4. Final evaluation
    print("\n5. Final Evaluation...")
    final_pred = net.forward(X)
    accuracy = np.mean((final_pred > 0.5) == y)
    print(f"   Final accuracy: {accuracy:.2%}")

    # 5. Visualization
    print("\n6. Generating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Learning curve
    axes[0].plot(losses, linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (Binary Cross-Entropy)")
    axes[0].set_title("Learning Curve: Backpropagation on XOR")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = net.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    axes[1].contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.6)
    axes[1].contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)
    axes[1].scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], c="red", marker="o", label="Class 1", edgecolors="k")
    axes[1].scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], c="blue", marker="s", label="Class 0", edgecolors="k")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].set_title(f"Decision Boundary (Accuracy: {accuracy:.2%})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("backpropagation_results.png", dpi=150, bbox_inches="tight")
    print("   ✓ Saved visualization to backpropagation_results.png")

    # 6. Key insights
    print("\n" + "=" * 70)
    print("Key Insights from Backpropagation")
    print("=" * 70)
    print("""
1. FORWARD PASS: Compute predictions and store intermediate values
   - z1 = x @ W1 + b1
   - a1 = relu(z1)
   - z2 = a1 @ W2 + b2
   - a2 = sigmoid(z2)

2. LOSS COMPUTATION: Binary cross-entropy for classification
   - L = -y*log(y_pred) - (1-y)*log(1-y_pred)

3. BACKWARD PASS: Chain rule to propagate gradients
   - dL/dW2 = dL/da2 * da2/dz2 * dz2/dW2 = (dL/dz2) @ a1.T
   - dL/dW1 = dL/da1 * da1/dz1 * dz1/dW1 = x.T @ (dL/dz1)

4. GRADIENT DESCENT: Update weights opposite to gradient direction
   - W = W - learning_rate * dL/dW

5. WHY THIS MATTERS:
   - PyTorch/TensorFlow automate steps 1-4 via autograd
   - Understanding this helps debug training issues
   - Gradient verification catches bugs in custom implementations
   - Different architectures (CNN, RNN, Transformer) use same principles

6. NUMERICAL VERIFICATION:
   - Compare analytical gradients with finite differences
   - Catch implementation bugs before training at scale
   - Relative error < 1e-4 indicates correct implementation
""")

    print("=" * 70)
    print("Training complete! 🎉")
    print("=" * 70)


if __name__ == "__main__":
    main()
