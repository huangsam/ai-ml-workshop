"""Model Quantization benchmark (FP32 vs dynamic INT8) in PyTorch."""

import io
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Simple multi-layer perceptron (MLP) supporting dynamic quantization
class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


# Helper to generate synthetic classification data
def generate_data(num_samples=1000):
    np.random.seed(42)
    X = np.random.randn(num_samples, 20).astype(np.float32)
    # Target class depends on simple linear combinations + bias
    scores = X[:, :5].sum(axis=1) + 0.5 * X[:, 5:10].sum(axis=1)
    Y = np.zeros(num_samples, dtype=np.int64)
    Y[scores > 1.0] = 1
    Y[scores < -1.0] = 2
    return torch.tensor(X), torch.tensor(Y)


def measure_model_size_kb(model):
    """Save model to memory buffer and measure its size in Kilobytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / 1024.0


def benchmark_model(model, X, Y):
    """Measure inference latency per 100 samples and accuracy on CPU."""
    model.eval()
    correct = 0
    total = len(X)

    # Measure latency using individual sample inferences to simulate live request serving
    start_time = time.perf_counter()
    with torch.no_grad():
        for i in range(total):
            sample = X[i].unsqueeze(0)  # Shape (1, 20)
            logits = model(sample)
            pred = torch.argmax(logits, dim=1).item()
            if pred == Y[i].item():
                correct += 1
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000.0
    latency_per_100_samples = (total_time_ms / total) * 100.0
    accuracy = correct / total

    return latency_per_100_samples, accuracy


def main(hook=None, config=None):
    from workshop.utils.hooks import NoOpProgressHook

    config = config or {}
    hook = hook or NoOpProgressHook()

    # Configure quantization engine dynamically based on host platform support
    supported_engines = torch.backends.quantized.supported_engines
    if "qnnpack" in supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in supported_engines:
        torch.backends.quantized.engine = "fbgemm"

    num_samples = int(config.get("num_samples", 500))

    print("PyTorch Model Quantization (FP32 vs INT8)")
    print("=" * 45)
    print(f"Evaluation samples: {num_samples}")
    print()

    if hook.is_cancelled():
        return
    hook.update_stage("Baseline Evaluation", 10)
    print("Generating dataset and training baseline MLP...")

    # 1. Setup training and evaluation data
    X_train, Y_train = generate_data(1000)
    X_test, Y_test = generate_data(num_samples)

    # 2. Train baseline MLP (FP32)
    model = MLPClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Quick training for 5 epochs to establish weights
    for epoch in range(5):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Benchmark FP32 Baseline
    fp32_size = measure_model_size_kb(model)
    fp32_latency, fp32_acc = benchmark_model(model, X_test, Y_test)

    print(f"FP32 Model size: {fp32_size:.2f} KB")
    print(f"FP32 Latency per 100 samples: {fp32_latency:.2f} ms")
    print(f"FP32 Test Accuracy: {fp32_acc:.2%}")
    print()

    if hook.is_cancelled():
        return
    hook.update_stage("Dynamic Quantization", 40)
    print("Applying PyTorch dynamic 8-bit quantization...")

    # 3. Apply dynamic quantization to Linear layers
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    print("Quantization complete.")

    if hook.is_cancelled():
        return
    hook.update_stage("Quantized Evaluation", 60)
    print("Benchmarking quantized INT8 model...")

    # Benchmark INT8 Quantized Model
    int8_size = measure_model_size_kb(quantized_model)
    int8_latency, int8_acc = benchmark_model(quantized_model, X_test, Y_test)

    print(f"INT8 Model size: {int8_size:.2f} KB")
    print(f"INT8 Latency per 100 samples: {int8_latency:.2f} ms")
    print(f"INT8 Test Accuracy: {int8_acc:.2%}")
    print()

    if hook.is_cancelled():
        return
    hook.update_stage("Metrics Comparison", 80)

    # Stream metrics
    hook.update_metrics(
        {
            "step": 1,
            "fp32_size_kb": float(fp32_size),
            "int8_size_kb": float(int8_size),
            "fp32_latency_ms": float(fp32_latency),
            "int8_latency_ms": float(int8_latency),
        }
    )

    if hook.is_cancelled():
        return
    hook.update_stage("Visualization", 90)

    # Plot FP32 vs INT8 performance comparisons
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))

    categories = ["FP32 (Original)", "INT8 (Quantized)"]
    colors = ["#3b82f6", "#10b981"]  # Indigo-blue and Emerald-green

    # Chart 1: Model Size
    axes[0].bar(categories, [fp32_size, int8_size], color=colors, edgecolor="black", alpha=0.8, width=0.5)
    axes[0].set_ylabel("Size (KB)")
    axes[0].set_title("Model File Size\n(Lower is better)")
    axes[0].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate([fp32_size, int8_size]):
        axes[0].text(i, v + (max(fp32_size, int8_size) * 0.02), f"{v:.1f} KB", ha="center", fontweight="bold")

    # Chart 2: Latency
    axes[1].bar(categories, [fp32_latency, int8_latency], color=colors, edgecolor="black", alpha=0.8, width=0.5)
    axes[1].set_ylabel("Time (ms)")
    axes[1].set_title("Latency per 100 Samples\n(Lower is better)")
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate([fp32_latency, int8_latency]):
        axes[1].text(i, v + (max(fp32_latency, int8_latency) * 0.02), f"{v:.2f} ms", ha="center", fontweight="bold")

    # Chart 3: Accuracy
    axes[2].bar(categories, [fp32_acc * 100.0, int8_acc * 100.0], color=colors, edgecolor="black", alpha=0.8, width=0.5)
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_title("Classification Accuracy\n(Higher is better)")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].set_ylim(0, 110)
    for i, v in enumerate([fp32_acc, int8_acc]):
        axes[2].text(i, (v * 100.0) + 2, f"{v:.1%}", ha="center", fontweight="bold")

    plt.suptitle("PyTorch Dynamic Quantization Performance Benchmarks", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    hook.save_plot("quantization_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    hook.update_stage("Complete", 100)
