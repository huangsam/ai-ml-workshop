# AI/ML workshop

The goal of this Git repository is for me to learn and practice AI/ML concepts and techniques. It contains code examples, tutorials, and projects that I work on to enhance my understanding of artificial intelligence and machine learning.

I will be covering the usage of PyTorch, TensorFlow, Hugging Face, LangChain and any other relevant libraries and frameworks.

Prior to this, I will also brush up on my knowledge of Python programming, data manipulation with libraries like NumPy and Pandas, and data visualization using Matplotlib and Seaborn. Furthermore, I will revisit my Coursera ML course materials to reinforce foundational concepts.

## Learnings & Progress

### Project 1: Custom Text Classifier (PyTorch)
- PyTorch is more intuitive for ML on macOS, with native MPS acceleration.
- Hugging Face automates datasets, tokenizers, and classifiers, but understanding BERT internals is valuable.
- Covered tokenization, training, and evaluation; prediction should be straightforward to add.

### Project 2: Semantic Search Engine with RAG
- Built RAG apps for Wikipedia, source code, and mechanical keyboards using LangChain and LlamaIndex.
- Learned optimization: context windows, chunk size/overlap, retrieval k, RRF with BM25/Chroma.
- Evaluated using LLM-as-judge on correctness, faithfulness, relevance.
- Ollama is simpler but less customizable than Hugging Face; LangGraph improved over basic LangChain.

### Project 3: LLM Code Review Agent
- Applied RAG concepts with LangChain tools (MCP).
- Basic implementation; could be enhanced with custom MCP servers for deeper learning.

### Project 4: Sklearn Machine Learning Algorithms
- Implemented supervised learning algorithms: linear/logistic regression, decision trees, random forest, KNN, SVM.
- Covered unsupervised learning: K-means clustering and PCA for dimensionality reduction.
- Learned hyperparameter tuning with GridSearchCV and RandomizedSearchCV.
- Understood evaluation metrics, feature scaling, cross-validation, and algorithm selection.
- Built complete examples with visualization and educational comments.

### Project 5: ML Fundamentals with NumPy
- Implemented core mathematical concepts from ML: linear algebra operations, vector/matrix calculations.
- Built linear regression from scratch using normal equation and gradient descent.
- Explored cost functions, feature scaling techniques (standardization, min-max scaling).
- Demonstrated eigenvalues/eigenvectors for PCA foundations.
- **Detailed learnings documented in [`docs/numpy.md`](docs/numpy.md)**.

## Current Readiness: 8-9/10 for AI/LLM Practitioner

### Strengths
- ✅ Practical experience with modern ML stack (PyTorch, Hugging Face, LangChain, LlamaIndex)
- ✅ Traditional ML proficiency (sklearn: regression, classification, clustering, dimensionality reduction)
- ✅ RAG expertise including optimization (chunking, retrieval, RRF with BM25/Chroma, evaluation)
- ✅ Production deployment skills (FastAPI + LangChain/LangGraph from previous work)
- ✅ Data engineering proficiency (NumPy/Pandas from industry experience)
- ✅ Practical prompt engineering (debugging bad responses, optimization through iteration)
- ✅ End-to-end project completion across classification, RAG, and agentic workflows

### Key Gaps to Address
- ⚠️ **Fine-tuning & training**: Haven't trained models from scratch or used PEFT/LoRA
- ⚠️ **ML fundamentals**: Need to revisit theory (gradient descent, backpropagation, loss functions)
- ⚠️ **MLOps**: CI/CD for ML, model versioning, experiment tracking, monitoring
- ⚠️ **Advanced RAG**: Multi-hop reasoning, agentic RAG, query decomposition
- ⚠️ **Cost optimization**: Token usage, caching strategies, batch processing at scale

## Next Steps (Priority Order)
1. **Fine-tune a model** using PEFT/LoRA on a domain-specific task ← Biggest ROI for career advancement
2. **Revisit Coursera ML fundamentals** to solidify theoretical understanding (see numpy/ folder)
3. **Build an advanced RAG project** with multi-agent system or complex reasoning
4. **Learn MLOps basics** (Weights & Biases for experiment tracking, model registry)
5. **Contribute to open source** (LangChain/LlamaIndex issues or PRs)
6. **Brush up on visualization** (Matplotlib, Seaborn for model evaluation and EDA)
