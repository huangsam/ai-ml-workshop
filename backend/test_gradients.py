"""Unit tests verifying NumPy analytical gradients against PyTorch autograd using pytest."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from workshop.core.numpy.attention import softmax as attention_softmax
from workshop.core.numpy.attention import softmax_backward as attention_softmax_backward
from workshop.core.numpy.backpropagation import SimpleNeuralNetwork
from workshop.core.numpy.transformer import NumpyTransformerBlock


@pytest.fixture(autouse=True)
def set_seeds():
    """Set random seeds before every test to ensure reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)


def test_backpropagation_gradients():
    """Verify SimpleNeuralNetwork gradients w.r.t PyTorch Autograd using MSE loss gradient mapping."""
    # Initialize NumPy SNN
    net = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, seed=42)
    X = np.random.randn(5, 2)
    Y = np.random.randint(0, 2, (5, 1)).astype(float)

    # NumPy Forward and Backward pass
    _ = net.forward(X)
    grads_np, _ = net.backward(Y, learning_rate=0.0)  # lr=0 to prevent updating weights before checking

    # Matching PyTorch Model
    class PyTorchNet(nn.Module):
        def __init__(self, W1, b1, W2, b2):
            super().__init__()
            self.W1 = nn.Parameter(torch.tensor(W1, dtype=torch.float64))
            self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float64))
            self.W2 = nn.Parameter(torch.tensor(W2, dtype=torch.float64))
            self.b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float64))

        def forward(self, x):
            z1 = torch.matmul(x, self.W1) + self.b1
            a1 = torch.relu(z1)
            z2 = torch.matmul(a1, self.W2) + self.b2
            a2 = torch.sigmoid(z2)
            return a2

    pt_net = PyTorchNet(net.W1, net.b1, net.W2, net.b2)

    # PyTorch Forward and Backward pass
    X_pt = torch.tensor(X, dtype=torch.float64)
    Y_pt = torch.tensor(Y, dtype=torch.float64)
    y_pred_pt = pt_net(X_pt)

    # SNN uses Mean Squared Error gradient scaling for backprop
    loss_pt = 0.5 * torch.mean((y_pred_pt - Y_pt) ** 2)
    loss_pt.backward()

    # Compare analytical weights/biases gradients
    np.testing.assert_allclose(grads_np["dW1"], pt_net.W1.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(grads_np["db1"], pt_net.b1.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(grads_np["dW2"], pt_net.W2.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(grads_np["db2"], pt_net.b2.grad.numpy(), rtol=1e-7, atol=1e-7)


def test_attention_gradients():
    """Verify NumPy Self-Attention gradients against PyTorch Autograd."""
    sequence_length = 6
    embedding_dim = 16
    scale = np.sqrt(embedding_dim)

    X = np.random.randn(sequence_length, embedding_dim)
    Y = np.random.randn(sequence_length, embedding_dim)

    # Initialize weights
    limit = np.sqrt(6.0 / (2 * embedding_dim))
    W_Q = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim))
    W_K = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim))
    W_V = np.random.uniform(-limit, limit, (embedding_dim, embedding_dim))

    # --- NumPy attention logic ---
    Q = np.dot(X, W_Q)
    K = np.dot(X, W_K)
    V = np.dot(X, W_V)
    scores = np.dot(Q, K.T) / scale
    A = attention_softmax(scores)
    attn_out = np.dot(A, V)

    # Backward
    d_attn_out = attn_out - Y
    dV = np.dot(A.T, d_attn_out)
    dA = np.dot(d_attn_out, V.T)
    dScores_scaled = attention_softmax_backward(A, dA)
    dScores = dScores_scaled / scale
    dQ = np.dot(dScores, K)
    dK = np.dot(dScores.T, Q)
    dW_Q = np.dot(X.T, dQ)
    dW_K = np.dot(X.T, dK)
    dW_V = np.dot(X.T, dV)

    # --- PyTorch autograd logic ---
    X_pt = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    Y_pt = torch.tensor(Y, dtype=torch.float64)
    W_Q_pt = torch.tensor(W_Q, dtype=torch.float64, requires_grad=True)
    W_K_pt = torch.tensor(W_K, dtype=torch.float64, requires_grad=True)
    W_V_pt = torch.tensor(W_V, dtype=torch.float64, requires_grad=True)

    Q_pt = torch.matmul(X_pt, W_Q_pt)
    K_pt = torch.matmul(X_pt, W_K_pt)
    V_pt = torch.matmul(X_pt, W_V_pt)
    scores_pt = torch.matmul(Q_pt, K_pt.t()) / scale
    A_pt = torch.softmax(scores_pt, dim=-1)
    out_pt = torch.matmul(A_pt, V_pt)

    loss_pt = 0.5 * torch.sum((out_pt - Y_pt) ** 2)
    loss_pt.backward()

    # Compare gradients
    np.testing.assert_allclose(dW_Q, W_Q_pt.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(dW_K, W_K_pt.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(dW_V, W_V_pt.grad.numpy(), rtol=1e-7, atol=1e-7)


def test_transformer_gradients():
    """Verify NumpyTransformerBlock gradients against PyTorch Autograd."""
    seq_len = 8
    d_model = 16
    num_heads = 2
    hidden_dim = 32
    d_k = d_model // num_heads

    block = NumpyTransformerBlock(d_model=d_model, num_heads=num_heads, hidden_dim=hidden_dim, seed=42)
    X = np.random.randn(seq_len, d_model)
    dout = np.random.randn(seq_len, d_model)

    # NumPy Forward and Backward pass
    _ = block.forward(X)
    dX_np = block.backward(dout, lr=0.0)  # lr=0 to prevent SGD parameter update

    # --- PyTorch Equivalent ---
    X_pt = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    dout_pt = torch.tensor(dout, dtype=torch.float64)

    # Load weights into PyTorch
    gamma1 = torch.tensor(block.gamma1, dtype=torch.float64, requires_grad=True)
    beta1 = torch.tensor(block.beta1, dtype=torch.float64, requires_grad=True)
    gamma2 = torch.tensor(block.gamma2, dtype=torch.float64, requires_grad=True)
    beta2 = torch.tensor(block.beta2, dtype=torch.float64, requires_grad=True)

    W_Q_pt = [torch.tensor(block.W_Q[h], dtype=torch.float64, requires_grad=True) for h in range(num_heads)]
    W_K_pt = [torch.tensor(block.W_K[h], dtype=torch.float64, requires_grad=True) for h in range(num_heads)]
    W_V_pt = [torch.tensor(block.W_V[h], dtype=torch.float64, requires_grad=True) for h in range(num_heads)]
    W_O_pt = torch.tensor(block.W_O, dtype=torch.float64, requires_grad=True)

    W1_pt = torch.tensor(block.W1, dtype=torch.float64, requires_grad=True)
    b1_pt = torch.tensor(block.b1, dtype=torch.float64, requires_grad=True)
    W2_pt = torch.tensor(block.W2, dtype=torch.float64, requires_grad=True)
    b2_pt = torch.tensor(block.b2, dtype=torch.float64, requires_grad=True)

    # PyTorch forward pass matching block.forward exactly
    # LayerNorm 1
    mean1 = X_pt.mean(dim=-1, keepdim=True)
    var1 = X_pt.var(dim=-1, unbiased=False, keepdim=True)
    X_norm1 = (X_pt - mean1) / torch.sqrt(var1 + 1e-5)
    LN1_out = gamma1 * X_norm1 + beta1

    # Multi-Head Attention
    heads_O = []
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

    for h in range(num_heads):
        Q_h = torch.matmul(LN1_out, W_Q_pt[h])
        K_h = torch.matmul(LN1_out, W_K_pt[h])
        V_h = torch.matmul(LN1_out, W_V_pt[h])

        scores_h = torch.matmul(Q_h, K_h.t()) / np.sqrt(d_k)
        # Apply causal masking
        scores_h = torch.where(mask == 1, torch.tensor(-1e9, dtype=torch.float64), scores_h)
        A_h = torch.softmax(scores_h, dim=-1)
        O_h = torch.matmul(A_h, V_h)
        heads_O.append(O_h)

    O_pt = torch.cat(heads_O, dim=-1)
    attn_out = torch.matmul(O_pt, W_O_pt)

    # Residual 1
    X_res1 = X_pt + attn_out

    # LayerNorm 2
    mean2 = X_res1.mean(dim=-1, keepdim=True)
    var2 = X_res1.var(dim=-1, unbiased=False, keepdim=True)
    X_norm2 = (X_res1 - mean2) / torch.sqrt(var2 + 1e-5)
    LN2_out = gamma2 * X_norm2 + beta2

    # FFN
    H_pt = torch.matmul(LN2_out, W1_pt) + b1_pt
    A_relu_pt = torch.relu(H_pt)
    ffn_out = torch.matmul(A_relu_pt, W2_pt) + b2_pt

    # Residual 2
    out_pt = X_res1 + ffn_out

    # Backpropagation
    out_pt.backward(dout_pt)

    # Compare outputs
    np.testing.assert_allclose(dX_np, X_pt.grad.numpy(), rtol=1e-7, atol=1e-7)

    # Compare parameter weight gradients
    np.testing.assert_allclose(block.dW2, W2_pt.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(block.db2, b2_pt.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(block.dW1, W1_pt.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(block.db1, b1_pt.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(block.dW_O, W_O_pt.grad.numpy(), rtol=1e-7, atol=1e-7)

    for h in range(num_heads):
        np.testing.assert_allclose(block.dW_Q[h], W_Q_pt[h].grad.numpy(), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(block.dW_K[h], W_K_pt[h].grad.numpy(), rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(block.dW_V[h], W_V_pt[h].grad.numpy(), rtol=1e-7, atol=1e-7)

    np.testing.assert_allclose(block.dgamma1, gamma1.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(block.dbeta1, beta1.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(block.dgamma2, gamma2.grad.numpy(), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(block.dbeta2, beta2.grad.numpy(), rtol=1e-7, atol=1e-7)
