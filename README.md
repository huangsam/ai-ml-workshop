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

The workshop includes a CLI interface to run individual machine learning models and tutorials. See [USERGUIDE.md](file:///Users/samhuang/Playground/practice/ai-ml-workshop/USERGUIDE.md) for the complete list of CLI commands:

```bash
# Example: Run PEFT/LoRA fine-tuning
uv run workshop pytorch fine-tuning

# Example: Run Linear Regression
uv run workshop sklearn linear-regression
```

For detailed instructions on running all PyTorch, Scikit-learn, and NumPy tutorials, refer to the [User Guide](file:///Users/samhuang/Playground/practice/ai-ml-workshop/USERGUIDE.md).

### Running via Web Dashboard (Full-Stack)

You can also run, configure, and monitor all workshop tasks using the real-time web dashboard:

1. **Start the FastAPI Backend**:

   ```bash
   uv run workshop server
   ```

2. **Start the Next.js Frontend**:

   ```bash
   npm install --prefix frontend
   npm run dev --prefix frontend
   ```

Open [http://localhost:3000](http://localhost:3000) in your browser to access the dashboard.

## Learnings & Progress

Our learning modules and accomplishments are organized into three primary areas:

- **[Fundamentals](docs/fundamentals.md)**: ML theory, backpropagation, and fundamental array operations from scratch using NumPy.
- **[Frameworks](docs/frameworks.md)**: Scikit-learn (classical ML algorithms) and PyTorch (deep learning models, custom text classifiers, and PEFT/LoRA fine-tuning).
- **[Integrations](docs/integrations.md)**: RAG systems, AI agents (semantic search and code review), and documentation on bridges to external data tools.
- **[Dashboard Architecture](docs/dashboard.md)**: Real-time asynchronous job scheduler, dynamic Pydantic configuration forms, in-memory Matplotlib interception hooks, and Server-Sent Events (SSE) telemetry pipeline.
