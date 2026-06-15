# User Guide: AI/ML Workshop

This guide provides instructions on how to interact with the AI/ML workshop codebase using both the command-line interface (CLI) and the web dashboard.

---

## Command-Line Interface (CLI)

The workshop is built with a CLI tool called `workshop` for executing specific models and learning modules.

### Prerequisites

Make sure you have installed the project and its dependencies using `uv`:

```bash
# Install dependencies with uv
uv sync

# Or install in editable mode for development
uv pip install -e .
```

### Running the Examples

Run any of the following commands using `uv run` to execute the scripts and see their outputs (logs, accuracy metrics, and generated plots):

#### Deep Learning (PyTorch)

```bash
# PEFT/LoRA fine-tuning
uv run workshop pytorch fine-tuning

# ResNet-18 on CIFAR-10
uv run workshop pytorch image-classification

# BERT on IMDb
uv run workshop pytorch text-classification

# BERT on SQuAD
uv run workshop pytorch question-answering

# LSTM forecasting
uv run workshop pytorch time-series

# MLP with embeddings
uv run workshop pytorch tabular-classification
```

#### Classical ML (Scikit-learn)

Running these examples will output metric scores and save visualization plots (e.g., confusion matrices, decision boundaries) in the root directory.

```bash
# Linear Regression
uv run workshop sklearn linear-regression

# Logistic Regression
uv run workshop sklearn logistic-regression

# K-Nearest Neighbors
uv run workshop sklearn knn

# Decision Tree
uv run workshop sklearn decision-tree

# Support Vector Machine
uv run workshop sklearn svm

# Random Forest
uv run workshop sklearn random-forest

# K-Means Clustering
uv run workshop sklearn kmeans

# Principal Component Analysis
uv run workshop sklearn pca
```

#### Fundamentals (NumPy)

Learn machine learning and backpropagation from scratch without framework dependencies:

```bash
# Basic array manipulations and operations
uv run workshop numpy fundamentals

# Manual backpropagation & chain rule computation
uv run workshop numpy backpropagation
```

---

## Web Dashboard (Full-Stack)

Alternatively, you can run, configure, and monitor all of these tasks visually through the web dashboard.

### 1. Start the FastAPI Backend

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

### 2. Start the Next.js Frontend

```bash
# Install frontend dependencies
npm install --prefix frontend

# Start the development server
npm run dev --prefix frontend
```

Once running, navigate to [http://localhost:3000](http://localhost:3000) to start, stop, configure, and inspect tasks in real time.
