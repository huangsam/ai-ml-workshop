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
uv run workshop sklearn linear-regression    # Linear Regression
uv run workshop sklearn logistic-regression  # Logistic Regression
uv run workshop sklearn knn                  # K-Nearest Neighbors
uv run workshop sklearn decision-tree        # Decision Tree
uv run workshop sklearn svm                  # Support Vector Machine
uv run workshop sklearn random-forest        # Random Forest
uv run workshop sklearn kmeans               # K-Means Clustering
uv run workshop sklearn pca                  # Principal Component Analysis

# NumPy examples
uv run workshop numpy fundamentals           # Fundamentals
uv run workshop numpy backpropagation        # Backpropagation
```

## Learnings & Progress

### Core Workshop Projects (Code in this Repo)

- **Custom Text Classifier**: PyTorch & BERT fundamentals.
- **AI Agents & RAG**: Semantic search and code review agents. **[LESSONS.md](LESSONS.md)**
- **Scikit-Learn Algorithms**: 8 classical ML algorithm implementations. **[docs/frameworks.md](docs/frameworks.md)**
- **NumPy Fundamentals**: ML concepts from scratch. **[docs/fundamentals.md](docs/fundamentals.md)**
- **PyTorch Deep Learning**: CNNs, RNNs, and Transformers. **[docs/frameworks.md](docs/frameworks.md)**
- **PEFT/LoRA Fine-Tuning**: Efficient LLM adaptation. **[docs/frameworks.md](docs/frameworks.md)**
- **Backpropagation & Theory**: Chain rule and math foundations. **[docs/fundamentals.md](docs/fundamentals.md)**

### Related Integrations (External Repos)
Documentation for how external data engineering and media analysis tools bridge to this workshop. **[See docs/integrations.md](docs/integrations.md)**
