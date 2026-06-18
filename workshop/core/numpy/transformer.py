"""Transformer block from scratch in NumPy with Multi-Head Attention, LayerNorm, and FFN."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def softmax(x):
    """Numerically stable row-wise softmax."""
    # Assume x is 2D: (L, L) or (L, V)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class CharTokenizer:
    """Simple character-level tokenizer for toy text datasets."""

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_id = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.id_to_char[i] for i in ids])


class NumpyTransformerBlock:
    """A single-layer Causal Transformer Block implemented in pure NumPy."""

    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, seed: int = 42):
        np.random.seed(seed)
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.d_k = d_model // num_heads

        # Xavier limits for weight initialization
        limit_attn = np.sqrt(6.0 / (d_model + self.d_k))
        limit_ffn1 = np.sqrt(6.0 / (d_model + hidden_dim))
        limit_ffn2 = np.sqrt(6.0 / (hidden_dim + d_model))

        # Multi-head attention projection weights
        self.W_Q = [np.random.uniform(-limit_attn, limit_attn, (d_model, self.d_k)) for _ in range(num_heads)]
        self.W_K = [np.random.uniform(-limit_attn, limit_attn, (d_model, self.d_k)) for _ in range(num_heads)]
        self.W_V = [np.random.uniform(-limit_attn, limit_attn, (d_model, self.d_k)) for _ in range(num_heads)]
        self.W_O = np.random.uniform(-limit_attn, limit_attn, (d_model, d_model))

        # FeedForward Network weights and biases
        self.W1 = np.random.uniform(-limit_ffn1, limit_ffn1, (d_model, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.uniform(-limit_ffn2, limit_ffn2, (hidden_dim, d_model))
        self.b2 = np.zeros((1, d_model))

        # Layer Normalization scale (gamma) and shift (beta) parameters
        self.gamma1 = np.ones((1, d_model))
        self.beta1 = np.zeros((1, d_model))
        self.gamma2 = np.ones((1, d_model))
        self.beta2 = np.zeros((1, d_model))

        # Cache to store intermediate activations for backprop
        self.cache: dict[str, Any] = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the transformer block.

        Args:
            X: Input matrix of shape (seq_len, d_model)

        Returns:
            Output matrix of shape (seq_len, d_model)
        """
        # --- LayerNorm 1 ---
        mean1 = np.mean(X, axis=-1, keepdims=True)
        var1 = np.var(X, axis=-1, keepdims=True)
        X_norm1 = (X - mean1) / np.sqrt(var1 + 1e-5)
        LN1_out = self.gamma1 * X_norm1 + self.beta1

        self.cache["X"] = X
        self.cache["mean1"] = mean1
        self.cache["var1"] = var1
        self.cache["X_norm1"] = X_norm1
        self.cache["LN1_out"] = LN1_out

        # --- Multi-Head Causal Attention ---
        L = X.shape[0]
        heads_O = []
        heads_A = []
        heads_Q = []
        heads_K = []
        heads_V = []

        # Causal mask: set upper triangle values above the diagonal to 1
        mask = np.triu(np.ones((L, L)), k=1)

        for h in range(self.num_heads):
            Q_h = np.dot(LN1_out, self.W_Q[h])  # (L, d_k)
            K_h = np.dot(LN1_out, self.W_K[h])  # (L, d_k)
            V_h = np.dot(LN1_out, self.W_V[h])  # (L, d_k)

            # Scaled dot-product attention
            scores_h = np.dot(Q_h, K_h.T) / np.sqrt(self.d_k)  # (L, L)
            scores_h = np.where(mask == 1, -1e9, scores_h)  # Apply causal masking
            A_h = softmax(scores_h)  # (L, L)
            O_h = np.dot(A_h, V_h)  # (L, d_k)

            heads_O.append(O_h)
            heads_A.append(A_h)
            heads_Q.append(Q_h)
            heads_K.append(K_h)
            heads_V.append(V_h)

        heads_concat = np.hstack(heads_O)  # (L, d_model)
        attn_out = np.dot(heads_concat, self.W_O)  # (L, d_model)

        self.cache["heads_Q"] = heads_Q
        self.cache["heads_K"] = heads_K
        self.cache["heads_V"] = heads_V
        self.cache["heads_A"] = heads_A
        self.cache["heads_O"] = heads_O
        self.cache["O"] = heads_concat
        self.cache["attn_out"] = attn_out

        # Residual connection 1
        X_res1 = X + attn_out
        self.cache["X_res1"] = X_res1

        # --- LayerNorm 2 ---
        mean2 = np.mean(X_res1, axis=-1, keepdims=True)
        var2 = np.var(X_res1, axis=-1, keepdims=True)
        X_norm2 = (X_res1 - mean2) / np.sqrt(var2 + 1e-5)
        LN2_out = self.gamma2 * X_norm2 + self.beta2

        self.cache["mean2"] = mean2
        self.cache["var2"] = var2
        self.cache["X_norm2"] = X_norm2
        self.cache["LN2_out"] = LN2_out

        # --- FeedForward Network ---
        H = np.dot(LN2_out, self.W1) + self.b1  # (L, hidden_dim)
        A_relu = np.maximum(0, H)  # ReLU activation, (L, hidden_dim)
        ffn_out = np.dot(A_relu, self.W2) + self.b2  # (L, d_model)

        self.cache["H"] = H
        self.cache["A_relu"] = A_relu
        self.cache["ffn_out"] = ffn_out

        # Residual connection 2
        return X_res1 + ffn_out

    def backward(self, dout: np.ndarray, lr: float) -> np.ndarray:
        """Backward pass through the transformer block and updates parameter weights.

        Args:
            dout: Gradient w.r.t the block output, shape (seq_len, d_model)
            lr: Learning rate for SGD updates

        Returns:
            Gradient w.r.t the block input, shape (seq_len, d_model)
        """
        # --- FFN Residual Path ---
        d_ffn_out = dout

        # FFN backpropagation
        dW2 = np.dot(self.cache["A_relu"].T, d_ffn_out)
        db2 = np.sum(d_ffn_out, axis=0, keepdims=True)
        d_A_relu = np.dot(d_ffn_out, self.W2.T)

        dH = d_A_relu * (self.cache["H"] > 0)
        dW1 = np.dot(self.cache["LN2_out"].T, dH)
        db1 = np.sum(dH, axis=0, keepdims=True)
        d_LN2_out = np.dot(dH, self.W1.T)

        # LayerNorm 2 backpropagation
        dgamma2 = np.sum(d_LN2_out * self.cache["X_norm2"], axis=0, keepdims=True)
        dbeta2 = np.sum(d_LN2_out, axis=0, keepdims=True)

        D = self.d_model
        var2_eps = self.cache["var2"] + 1e-5
        d_X_res1_from_ln = (
            (1.0 / D)
            * self.gamma2
            / np.sqrt(var2_eps)
            * (
                D * d_LN2_out
                - np.sum(d_LN2_out, axis=-1, keepdims=True)
                - self.cache["X_norm2"] * np.sum(d_LN2_out * self.cache["X_norm2"], axis=-1, keepdims=True)
            )
        )

        # Combine gradients at Residual 2 node
        d_X_res1 = dout + d_X_res1_from_ln

        # --- Attention Residual Path ---
        d_attn_out = d_X_res1

        # Attention Output Projection backpropagation
        dW_O = np.dot(self.cache["O"].T, d_attn_out)
        d_heads_concat = np.dot(d_attn_out, self.W_O.T)

        # Split projection gradients among heads
        d_heads_O = np.hsplit(d_heads_concat, self.num_heads)
        d_LN1_out = np.zeros_like(self.cache["LN1_out"])

        dW_Q_heads = []
        dW_K_heads = []
        dW_V_heads = []

        for h in range(self.num_heads):
            d_O_h = d_heads_O[h]
            A_h = self.cache["heads_A"][h]
            V_h = self.cache["heads_V"][h]
            Q_h = self.cache["heads_Q"][h]
            K_h = self.cache["heads_K"][h]

            # Context matrix backprop
            dV_h = np.dot(A_h.T, d_O_h)
            dA_h = np.dot(d_O_h, V_h.T)

            # Causal-masked Softmax backprop
            d_scores_h = A_h * (dA_h - np.sum(dA_h * A_h, axis=-1, keepdims=True))
            d_scores_scaled_h = d_scores_h / np.sqrt(self.d_k)

            # Q/K projections backprop
            dQ_h = np.dot(d_scores_scaled_h, K_h)
            dK_h = np.dot(d_scores_scaled_h.T, Q_h)

            dW_Q_h = np.dot(self.cache["LN1_out"].T, dQ_h)
            dW_K_h = np.dot(self.cache["LN1_out"].T, dK_h)
            dW_V_h = np.dot(self.cache["LN1_out"].T, dV_h)

            dW_Q_heads.append(dW_Q_h)
            dW_K_heads.append(dW_K_h)
            dW_V_heads.append(dW_V_h)

            # Accumulate gradients w.r.t LN1_out
            d_LN1_out += np.dot(dQ_h, self.W_Q[h].T)
            d_LN1_out += np.dot(dK_h, self.W_K[h].T)
            d_LN1_out += np.dot(dV_h, self.W_V[h].T)

        # LayerNorm 1 backpropagation
        dgamma1 = np.sum(d_LN1_out * self.cache["X_norm1"], axis=0, keepdims=True)
        dbeta1 = np.sum(d_LN1_out, axis=0, keepdims=True)

        var1_eps = self.cache["var1"] + 1e-5
        d_X_from_ln1 = (
            (1.0 / D)
            * self.gamma1
            / np.sqrt(var1_eps)
            * (
                D * d_LN1_out
                - np.sum(d_LN1_out, axis=-1, keepdims=True)
                - self.cache["X_norm1"] * np.sum(d_LN1_out * self.cache["X_norm1"], axis=-1, keepdims=True)
            )
        )

        # Total input gradient: residual path + LayerNorm 1 path
        dX = d_X_res1 + d_X_from_ln1

        # Apply SGD Updates to all parameters
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        self.W_O -= lr * dW_O
        for h in range(self.num_heads):
            self.W_Q[h] -= lr * dW_Q_heads[h]
            self.W_K[h] -= lr * dW_K_heads[h]
            self.W_V[h] -= lr * dW_V_heads[h]

        self.gamma1 -= lr * dgamma1
        self.beta1 -= lr * dbeta1
        self.gamma2 -= lr * dgamma2
        self.beta2 -= lr * dbeta2

        return dX


def get_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Generates sine/cosine positional encoding vectors."""
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000.0 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000.0 ** (i / d_model)))
    return pe


def main(hook=None, config=None):
    """Entry point for the Transformer from scratch lesson."""
    from workshop.utils.hooks import NoOpProgressHook

    config = config or {}
    hook = hook or NoOpProgressHook()

    epochs = int(config.get("epochs", 120))
    learning_rate = float(config.get("learning_rate", 0.01))
    embedding_dim = int(config.get("embedding_dim", 16))
    num_heads = int(config.get("num_heads", 2))
    hidden_dim = int(config.get("hidden_dim", 32))
    seq_len = 8

    print("Causal Transformer Block from Scratch (NumPy)")
    print("=" * 50)
    print(f"Embedding Dim (D): {embedding_dim}")
    print(f"Attention Heads: {num_heads}")
    print(f"FFN Hidden Dim: {hidden_dim}")
    print(f"Epochs: {epochs}, Learning Rate: {learning_rate}")
    print()

    if embedding_dim % num_heads != 0:
        raise ValueError(f"Embedding dimension ({embedding_dim}) must be divisible by number of heads ({num_heads}).")

    if hook.is_cancelled():
        return
    hook.update_stage("Tokenization & Setup", 5)

    # Dataset: A short quote trained on character level
    text = "attention is all you need"
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size

    # Prepare datasets using a sliding window of length seq_len
    X_raw = []
    Y_raw = []
    for i in range(len(text) - seq_len):
        X_raw.append(tokenizer.encode(text[i : i + seq_len]))
        Y_raw.append(tokenizer.encode(text[i + 1 : i + seq_len + 1]))

    X_train = np.array(X_raw)  # (N, seq_len)
    Y_train = np.array(Y_raw)  # (N, seq_len)
    num_samples = X_train.shape[0]

    if hook.is_cancelled():
        return
    hook.update_stage("Embeddings & PE", 15)

    # Initialize vocabulary embeddings E (vocab_size, d_model)
    limit_emb = np.sqrt(6.0 / (vocab_size + embedding_dim))
    E = np.random.uniform(-limit_emb, limit_emb, (vocab_size, embedding_dim))

    # Initialize output vocabulary projection weights (d_model, vocab_size) and biases
    limit_proj = np.sqrt(6.0 / (embedding_dim + vocab_size))
    W_vocab = np.random.uniform(-limit_proj, limit_proj, (embedding_dim, vocab_size))
    b_vocab = np.zeros((1, vocab_size))

    # Compute static positional encoding PE
    PE = get_positional_encoding(seq_len, embedding_dim)

    # Build Transformer Block
    block = NumpyTransformerBlock(d_model=embedding_dim, num_heads=num_heads, hidden_dim=hidden_dim, seed=42)

    reporting_interval = max(1, epochs // 10)

    if hook.is_cancelled():
        return
    hook.update_stage("Training Loop", 25)

    # Training Loop (epoch based)
    for epoch in range(epochs):
        if hook.is_cancelled():
            return

        epoch_loss = 0.0
        epoch_correct = 0
        total_tokens = 0

        # Loop through each sequence sample sequentially (online stochastic gradient descent)
        for i in range(num_samples):
            x_seq = X_train[i]  # (seq_len,)
            y_seq = Y_train[i]  # (seq_len,)

            # --- Forward Pass ---
            # 1. Embedding lookup
            X_embed = E[x_seq]  # (seq_len, d_model)

            # 2. Add Positional Encoding
            X_pe = X_embed + PE

            # 3. Pass through Transformer Block
            X_block = block.forward(X_pe)

            # 4. Vocabulary Logits
            logits = np.dot(X_block, W_vocab) + b_vocab  # (seq_len, vocab_size)

            # 5. Softmax cross entropy loss
            probs = softmax(logits)  # (seq_len, vocab_size)

            # Compute CE loss
            sample_loss = -np.sum(np.log(probs[np.arange(seq_len), y_seq] + 1e-9))
            epoch_loss += sample_loss

            # Accuracy calculations
            predictions = np.argmax(probs, axis=-1)
            epoch_correct += np.sum(predictions == y_seq)
            total_tokens += seq_len

            # --- Backward Pass ---
            # d_logits w.r.t loss
            d_logits = probs.copy()
            d_logits[np.arange(seq_len), y_seq] -= 1.0
            # Scale gradients by sequence length to stabilize updates
            d_logits /= seq_len

            # Projections backpropagation
            dW_vocab = np.dot(X_block.T, d_logits)
            db_vocab = np.sum(d_logits, axis=0, keepdims=True)
            d_X_block = np.dot(d_logits, W_vocab.T)

            # Transformer block backpropagation
            d_X_pe = block.backward(d_X_block, learning_rate)

            # Embedding layer backpropagation (accumulate gradients on indices)
            dE = np.zeros_like(E)
            for pos, token in enumerate(x_seq):
                dE[token] += d_X_pe[pos]

            # Update weights using SGD
            W_vocab -= learning_rate * dW_vocab
            b_vocab -= learning_rate * db_vocab
            E -= learning_rate * dE

        # Epoch metrics
        mean_loss = epoch_loss / (num_samples * seq_len)
        acc = epoch_correct / total_tokens

        # Telemetry updates
        if epoch % reporting_interval == 0 or epoch == epochs - 1:
            progress = 25 + int(55 * (epoch / epochs))
            hook.update_stage("Training Loop", progress)
            hook.update_metrics({"epoch": epoch + 1, "loss": float(mean_loss), "accuracy": float(acc)})
            print(f"Epoch {epoch + 1:3d}/{epochs}: Loss={mean_loss:.4f}, Accuracy={acc:.2%}")

    if hook.is_cancelled():
        return
    hook.update_stage("Text Generation", 80)
    print("\nTraining completed. Autoregressively generating text starting with prompt 'atte'...")

    # Autoregressive text generation
    prompt = "atte"
    generated_ids = tokenizer.encode(prompt)

    # Use seq_len of positional encoding, dynamically padding/truncating active context
    for _ in range(15):
        # We need the last seq_len tokens to feed into the transformer
        context_ids = generated_ids[-seq_len:]
        context_len = len(context_ids)

        # Pad with zeros if prompt is too short (shouldn't happen here since prompt is length 4, seq_len is 8, but just in case)
        if context_len < seq_len:
            context_ids = [0] * (seq_len - context_len) + context_ids

        # Forward pass for generation
        X_embed = E[context_ids]
        X_pe = X_embed + PE

        X_block = block.forward(X_pe)
        logits = np.dot(X_block, W_vocab) + b_vocab  # (seq_len, vocab_size)
        probs = softmax(logits)

        # Predict next token (based on the last token's probability output)
        next_token_id = np.argmax(probs[-1])
        generated_ids.append(next_token_id)

    generated_text = tokenizer.decode(generated_ids)
    print(f"Prompt: '{prompt}'")
    print(f"Generated text: '{generated_text}'")
    print()

    if hook.is_cancelled():
        return
    hook.update_stage("Visualization", 90)

    # Retrieve final attention weights of each head for the last sample
    X_embed = E[X_train[-1]]
    X_pe = X_embed + PE
    _ = block.forward(X_pe)
    final_A = block.cache["heads_A"]  # List of attention weight matrices (L, L)

    # Plot Multi-Head Attention Alignments
    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 5))
    if num_heads == 1:
        axes = [axes]

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(final_A[h], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Head {h + 1} Attention Alignment", fontweight="bold")
        ax.set_xlabel("Key Tokens (Columns)")
        ax.set_ylabel("Query Tokens (Rows)")

        input_tokens = [tokenizer.id_to_char[c] for c in X_train[-1]]
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(input_tokens)
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels(input_tokens)

        # Display numeric values inside cells
        for r in range(seq_len):
            for c in range(seq_len):
                val = final_A[h][r, c]
                color = "white" if val < 0.5 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    fig.colorbar(im, ax=axes.ravel().tolist(), label="Attention Coefficient")
    plt.suptitle("Causal Self-Attention Heatmap (Note Causal Masking Upper Triangle)", fontsize=14, y=0.98, fontweight="bold")
    hook.save_plot("transformer_attention_heads.png", dpi=150, bbox_inches="tight")
    plt.close()

    hook.update_stage("Complete", 100)
