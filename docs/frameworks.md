# ML Frameworks: Scikit-Learn & PyTorch

This document provides a unified guide for implementing machine learning using industry-standard libraries. It covers model selection, algorithm details, and best practices for both classical ML and deep learning.

---

## Model Selection Guide

Choose the best approach based on your data type and problem complexity:

| Data Type | Recommended Framework | Algorithm Examples |
|-----------|----------------------|--------------------|
| **Tabular** (Structured) | Scikit-Learn | Random Forest, SVM, Logistic Regression |
| **Images** | PyTorch (Torchvision) | CNNs (ResNet), Vision Transformers |
| **Text** (NLP) | PyTorch (Transformers) | BERT, DistilBERT, GPT |
| **Time Series** | PyTorch | LSTMs, Transformers |
| **Unsupervised** | Scikit-Learn | K-Means, PCA |

---

## Classical ML (Scikit-Learn)

Scikit-Learn is the gold standard for tabular data and traditional statistical learning.

### Supervised Algorithms
- **Linear/Logistic Regression**: Base baselines for continuous and binary prediction.
- **Support Vector Machines (SVM)**: Excellent for high-dimensional data and non-linear boundaries via kernels.
- **Random Forests**: Robust ensemble of decision trees; handles missing values and provides feature importance.
- **K-Nearest Neighbors (KNN)**: Simple distance-based classification; performance is highly sensitive to feature scaling.

### Unsupervised Algorithms
- **K-Means Clustering**: Partitions data into $k$ groups by minimizing variance.
- **Principal Component Analysis (PCA)**: Reduces feature count while preserving maximum variance.

---

## Deep Learning (PyTorch)

PyTorch provides maximum flexibility for building complex neural architectures using $nn.Module$ and Autograd.

### Core Architectures
- **CNNs (Convolutional Neural Networks)**: Optimized for spatial data like images using shared weights and pooling.
- **LSTMs (Long Short-Term Memory)**: Reccurent networks designed to capture temporal dependencies in sequences.
- **Transformers (BERT/LLMs)**: Leverage attention mechanisms to understand global context in unstructured text.

### Efficient Fine-Tuning (PEFT/LoRA)
Adapting large pre-trained models on consumer hardware:
- **LoRA (Low-Rank Adaptation)**: Injects small trainable matrices into frozen weights.
- **Benefit**: Reduces trainable parameters by over 98%, enabling fine-tuning on Apple Silicon (MPS) or single GPUs.
- **Best Practice**: Focus on attention projections ($q\_proj, v\_proj$) and use higher learning rates (1e-4) than full fine-tuning.

---

## Production Practices

### Training & Evaluation
- **Feature Scaling**: Must-have for distance-based (KNN, SVM) and gradient-based (Neural Nets) models.
- **Cross-Validation**: Use `GridSearchCV` to optimize hyperparameters without overfitting.
- **Metrics**: Accuracy is rarely enough; monitor Precision/Recall for imbalanced classes and MSE/MAE for regression.

### Key Takeaways
- **Classical ML**: SVM and Random Forests remain strong baselines for tabular data.
- **Deep Learning**: Transformers' attention mechanism is transformative for long-range dependencies.
- **Fine-tuning**: LoRA can reduce trainable parameters by over 98% with minimal performance loss.

### Production Considerations
- **Latency vs. Accuracy**: Local models (e.g., Ollama) offer privacy and low cost but may trade off reasoning speed and accuracy.
- **Observability**: Track retrieval hits/misses and LLM token usage to optimize for both quality and cost.

### Deployment & Scale
- **Experiment Tracking**: Use Weights & Biases (W&B) or MLflow to track hyperparameters and artifacts.
- **Model Optimization**: For inference, use 4-bit/8-bit quantization and FlashAttention to reduce latency.
- **Inference Servers**: Deploy using specialized servers like vLLM, Triton, or TGI for production-grade scale.

### Coding Patterns
- **Device-Agnostic Code**: Detect MPS (Apple Silicon) or CUDA (NVIDIA) to ensure scripts run everywhere.
- **DataLoader**: Decouple data loading from model logic to enable efficient batching and multi-processing.
