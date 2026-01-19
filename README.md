# AI/ML workshop

The goal of this Git repository is to learn and practice AI/ML concepts and techniques through code examples, tutorials, and projects that enhance understanding of artificial intelligence and machine learning.

This covers practical experience with PyTorch, TensorFlow, Hugging Face, LangChain, and other relevant libraries, alongside foundational work in Python programming, data manipulation (NumPy, Pandas), data visualization (Matplotlib, Seaborn), and reinforcement of ML theory from Coursera materials.

## Getting Started

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd ai-ml-workshop

# Install dependencies with uv
uv sync

# Or install in editable mode for development
uv pip install -e .
```

### Running Examples
This workshop uses a CLI interface to run examples. After installation, you can run examples using:

```bash
# PyTorch examples
uv run workshop pytorch fine-tuning          # PEFT/LoRA fine-tuning
uv run workshop pytorch image-classification # ResNet-18 on CIFAR-10
uv run workshop pytorch text-classification  # BERT on IMDb
uv run workshop pytorch question-answering   # BERT on SQuAD
uv run workshop pytorch time-series          # LSTM forecasting
uv run workshop pytorch tabular              # MLP with embeddings

# Scikit-learn examples
uv run workshop sklearn linear-regression
uv run workshop sklearn logistic-regression
uv run workshop sklearn knn
uv run workshop sklearn decision-tree
uv run workshop sklearn svm
uv run workshop sklearn random-forest
uv run workshop sklearn kmeans
uv run workshop sklearn pca

# NumPy examples
uv run workshop numpy fundamentals
uv run workshop numpy backpropagation
```

## Learnings & Progress

### Project 1: Custom Text Classifier (PyTorch)
PyTorch is more intuitive for ML on macOS, with native MPS acceleration. Hugging Face automates datasets, tokenizers, and classifiers, but understanding BERT internals is valuable. Covered tokenization, training, and evaluation; prediction should be straightforward to add.

### Projects 2-3: AI Agents & RAG Applications
Developed semantic search engines and code review agents using LangChain, LlamaIndex, and LangGraph. Implemented RAG with hybrid retrieval (BM25 + semantic search), optimized chunking strategies, and evaluated using LLM-as-judge metrics. **[Full details in AGENTS.md](AGENTS.md)**

### Project 4: Sklearn Machine Learning Algorithms
Implemented 8 classical ML algorithms across supervised learning (linear/logistic regression, decision trees, random forest, KNN, SVM) and unsupervised learning (K-means clustering, PCA). Built complete examples with hyperparameter tuning, cross-validation, and evaluation metrics. **[Full details in docs/sklearn.md](docs/sklearn.md#project-4-classical-ml-algorithms)**

### Project 5: ML Fundamentals with NumPy
Implemented core ML concepts from scratch: linear algebra operations, linear regression (normal equation + gradient descent), cost functions, feature scaling, and PCA. **[Full details in docs/numpy.md](docs/numpy.md#project-5-ml-fundamentals-with-numpy)**

### Project 6: PyTorch Deep Learning Across Multiple Domains
Built 5 production-quality neural networks: image classification (CIFAR-10, ResNet-18), text classification (IMDb, BERT), question answering (SQuAD), time series forecasting (LSTM), and tabular classification (MLP with embeddings). Demonstrated MPS acceleration, transfer learning, and proper data handling across domains. **[Full details in docs/pytorch.md](docs/pytorch.md#project-6-deep-learning-across-multiple-domains)**

### Project 7: Parameter-Efficient Fine-Tuning with LoRA
Fine-tuned DistilBERT on AG News using LoRA adapters, achieving 86% accuracy with 98.89% parameter reduction. Demonstrated PEFT for efficient adaptation of large models on consumer hardware. **[Full details in docs/pytorch.md](docs/pytorch.md#project-7-parameter-efficient-fine-tuning-with-lora)**

### Bonus: Backpropagation from Scratch & ML Theory
Implemented a 2-layer neural network with manual backpropagation to understand gradient computation via chain rule. Included gradient verification and trained on the XOR problem. Comprehensive ML theory documentation covering calculus, optimization, regularization, and convergence. **[Full details in docs/numpy.md](docs/numpy.md#bonus-backpropagation-from-scratch)** and **[docs/ml_theory.md](docs/ml_theory.md)**
