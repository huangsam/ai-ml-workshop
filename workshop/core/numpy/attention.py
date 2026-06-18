"""Single-Head Self-Attention from scratch in NumPy with exact backpropagation."""

import matplotlib.pyplot as plt
import numpy as np


def softmax(x):
    """Numerically stable row-wise softmax."""
    # Assume x is 2D: (L, L)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def softmax_backward(S, dS):
    """Backpropagation through row-wise softmax.

    S: Softmax output (L, L)
    dS: Gradients w.r.t S (L, L)
    """
    L = S.shape[0]
    dA = np.zeros_like(S)
    for i in range(L):
        s = S[i].reshape(-1, 1)
        ds = dS[i].reshape(-1, 1)
        # Jacobian matrix for softmax: diag(s) - s s^T
        J = np.diag(S[i]) - np.dot(s, s.T)
        dA[i] = np.dot(J, ds).ravel()
    return dA


def main(hook=None, config=None):
    from workshop.utils.hooks import NoOpProgressHook

    config = config or {}
    hook = hook or NoOpProgressHook()

    epochs = int(config.get("epochs", 150))
    learning_rate = float(config.get("learning_rate", 0.05))
    embedding_dim = int(config.get("embedding_dim", 16))
    sequence_length = int(config.get("sequence_length", 6))

    print("Single-Head Self-Attention from Scratch")
    print("=" * 45)
    print(f"Sequence Length (L): {sequence_length}")
    print(f"Embedding Dimension (D): {embedding_dim}")
    print(f"Epochs: {epochs}, Learning Rate: {learning_rate}")
    print()

    # Generate synthetic input sequence X (L, D) and target Y (L, D)
    np.random.seed(42)
    X = np.random.randn(sequence_length, embedding_dim)

    # Target task: Output should be a reversed/shifted representation of input
    # (Forces the model to pay attention to different positions in the sequence)
    Y = np.roll(X, shift=1, axis=0)

    # Initialize weight matrices W_Q, W_K, W_V (D, D) using Xavier/Glorot initialization
    limit = np.sqrt(6.0 / (2 * embedding_dim))
    W_Q = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim))
    W_K = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim))
    W_V = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim))

    if hook.is_cancelled():
        return
    hook.update_stage("Vector Embeddings", 10)
    print("Embeddings and target labels initialized.")

    if hook.is_cancelled():
        return
    hook.update_stage("Attention Computation", 20)

    reporting_interval = max(1, epochs // 10)

    # Training loop
    for epoch in range(epochs):
        if hook.is_cancelled():
            return

        # --- Forward Pass ---
        # 1. Linear projections to Query, Key, Value
        Q = np.dot(X, W_Q)  # (L, D)
        K = np.dot(X, W_K)  # (L, D)
        V = np.dot(X, W_V)  # (L, D)

        # 2. Scaled Dot-Product Attention Scores
        # Scores = Q K^T / sqrt(D)
        scale = np.sqrt(embedding_dim)
        scores = np.dot(Q, K.T) / scale  # (L, L)

        # 3. Softmax to get Attention weights matrix
        A = softmax(scores)  # (L, L)

        # 4. Multiply by Value matrix to get output
        attn_out = np.dot(A, V)  # (L, D)

        # 5. Calculate MSE Loss
        loss = 0.5 * np.sum((attn_out - Y) ** 2)

        # Calculate a simple sequence accuracy (where cosine similarity to target > 0.8)
        similarity = np.sum(attn_out * Y, axis=1) / (np.linalg.norm(attn_out, axis=1) * np.linalg.norm(Y, axis=1) + 1e-8)
        accuracy = np.mean(similarity > 0.8)

        # --- Backward Pass ---
        # 1. Gradient of Loss w.r.t Output attn_out
        d_attn_out = attn_out - Y  # (L, D)

        # 2. Gradients w.r.t V and A
        # attn_out = A V  =>  dV = A^T d_attn_out, dA = d_attn_out V^T
        dV = np.dot(A.T, d_attn_out)  # (L, D)
        dA = np.dot(d_attn_out, V.T)  # (L, L)

        # 3. Gradient through Softmax
        dScores_scaled = softmax_backward(A, dA)  # (L, L)
        dScores = dScores_scaled / scale  # (L, L)

        # 4. Gradients w.r.t Q and K
        # scores = Q K^T  =>  dQ = dScores K,  dK = dScores^T Q
        dQ = np.dot(dScores, K)  # (L, D)
        dK = np.dot(dScores.T, Q)  # (L, D)

        # 5. Gradients w.r.t Weights
        # Q = X W_Q  =>  dW_Q = X^T dQ
        dW_Q = np.dot(X.T, dQ)
        dW_K = np.dot(X.T, dK)
        dW_V = np.dot(X.T, dV)

        # Update weights using SGD
        W_Q -= learning_rate * dW_Q
        W_K -= learning_rate * dW_K
        W_V -= learning_rate * dW_V

        # Telemetry updates
        if epoch % reporting_interval == 0 or epoch == epochs - 1:
            progress = 20 + int(60 * (epoch / epochs))
            hook.update_stage("Optimization", progress)
            hook.update_metrics({"epoch": epoch + 1, "loss": float(loss), "accuracy": float(accuracy)})
            print(f"Epoch {epoch + 1:3d}/{epochs}: Loss={loss:.6f}, CosSim Accuracy={accuracy:.2%}")

    if hook.is_cancelled():
        return
    hook.update_stage("Weights Extraction", 85)
    print("\nAttention matrix learned. Extracting final weights...")

    # Calculate final attention weights
    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    A_final = softmax(np.dot(Q, K.T) / np.sqrt(embedding_dim))

    if hook.is_cancelled():
        return
    hook.update_stage("Visualization", 90)

    # Plot Attention Weight Matrix Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(A_final, cmap="plasma", vmin=0, vmax=1)
    plt.colorbar(label="Attention Weight")

    # Add text labels on cell items
    for i in range(sequence_length):
        for j in range(sequence_length):
            plt.text(j, i, f"{A_final[i, j]:.2f}", ha="center", va="center", color="white" if A_final[i, j] < 0.6 else "black", fontweight="bold")

    plt.xlabel("Key Tokens (Columns)")
    plt.ylabel("Query Tokens (Rows)")
    plt.title("NumPy Self-Attention Alignment Matrix")
    plt.xticks(range(sequence_length), [f"Token {i}" for i in range(sequence_length)])
    plt.yticks(range(sequence_length), [f"Token {i}" for i in range(sequence_length)])
    plt.tight_layout()
    hook.save_plot("attention_weights_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    hook.update_stage("Complete", 100)
