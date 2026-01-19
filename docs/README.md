# ML/AI Learning Documentation

This folder contains comprehensive documentation of learnings, best practices, and theory across the workshop projects.

## Quick Navigation

### Core Concepts & Theory
- **[ML Theory](ml_theory.md)** - Mathematical foundations, optimization, regularization, backpropagation
- **[Model Selection](model_selection.md)** - Algorithm selection guide and decision trees

### Implementation Guides
- **[NumPy & Fundamentals](numpy.md)** - Linear algebra, matrix operations, numerical computing from scratch
- **[Scikit-Learn](sklearn.md)** - Classical ML algorithms, hyperparameter tuning, evaluation metrics
- **[PyTorch](pytorch.md)** - Deep learning, CNNs, RNNs, Transformers, PEFT/LoRA fine-tuning

### Application Domains
- **[RAG & LLMs](rag.md)** - Retrieval-Augmented Generation, vector stores, prompt optimization

---

## Learning Path

### Beginner: ML Fundamentals
1. Start with **NumPy** - understand vectors, matrices, and basic linear algebra
2. Review **ML Theory** - grasp optimization, loss functions, and regularization
3. Explore **Scikit-Learn** - implement classical algorithms with real datasets

### Intermediate: Deep Learning
1. Study **PyTorch** - build neural networks, understand backpropagation in practice
2. Work through **Model Selection** - choose the right architecture for your problem
3. Implement domain-specific examples (images, text, tabular, time series)

### Advanced: LLMs & Production
1. Master **PyTorch PEFT/LoRA** - parameter-efficient fine-tuning for large models
2. Build with **RAG** - combine retrieval with generative models
3. Deploy and optimize (see main README for MLOps gaps)

---

## Key Insights by Domain

| Domain | Key Insight | Reference |
|--------|-------------|-----------|
| **Linear Algebra** | Eigenvalues/eigenvectors power dimensionality reduction | numpy.md |
| **Optimization** | SGD + momentum outperforms vanilla gradient descent | ml_theory.md |
| **Classification** | SVM and Random Forests are effective baselines | sklearn.md |
| **Deep Learning** | RNNs capture temporal patterns; Transformers handle long-range dependencies | pytorch.md |
| **Fine-tuning** | LoRA achieves 95-99% parameter reduction without performance loss | pytorch.md |
| **RAG** | Context window, chunk size, and retriever choice heavily influence quality | rag.md |

---

## File Descriptions

### [numpy.md](numpy.md)
Covers fundamental numerical computing concepts implemented from scratch using NumPy:
- Vector/matrix operations and broadcasting
- Cost functions and gradients
- Feature scaling and normalization
- Linear regression (normal equation + gradient descent)
- Eigenvalue decomposition for PCA

### [ml_theory.md](ml_theory.md)
Mathematical foundations of machine learning:
- Calculus and optimization theory
- Loss functions and regularization
- Backpropagation and gradient computation
- Convergence analysis
- Numerical stability considerations

### [sklearn.md](sklearn.md)
Classical machine learning with scikit-learn:
- Supervised learning (regression, classification)
- Unsupervised learning (clustering, dimensionality reduction)
- Hyperparameter tuning and cross-validation
- Evaluation metrics and visualization
- Model comparison and selection strategies

### [pytorch.md](pytorch.md)
Deep learning with PyTorch across multiple domains:
- PyTorch fundamentals and tensor operations
- CNNs for image classification (ResNet-18 example)
- RNNs/LSTMs for sequential data (time series, text)
- Transformers and BERT for NLP
- MLPs for tabular data
- PEFT and LoRA for parameter-efficient fine-tuning

### [rag.md](rag.md)
Retrieval-Augmented Generation and LLM applications:
- RAG architecture and workflow
- Vector stores and embeddings (Chroma, FAISS)
- BM25 and hybrid retrieval
- Prompt engineering and optimization
- Evaluation: LLM-as-judge metrics
- Tools: LangChain, LlamaIndex, LangGraph

### [model_selection.md](model_selection.md)
Decision framework for choosing the right algorithm:
- Algorithm selection by problem type
- Trade-offs (accuracy, interpretability, speed)
- When to use classical ML vs. deep learning
- Benchmark results and performance comparisons

---

## How to Use These Docs

1. **Learning**: Read in sequence based on your current level
2. **Reference**: Use Cmd+F to find specific topics
3. **Implementation**: Link examples back to code in `examples/` folder
4. **Teaching**: Share specific sections with others learning ML/AI

---

## Contributing

When adding new learnings:
1. Add to the appropriate `.md` file or create a new one
2. Update this README with navigation link
3. Include code examples and references
4. Link to corresponding implementations in `examples/`
