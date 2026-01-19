# AI/ML Workshop: Complete Guide

The goal of this repository is to learn and practice AI/ML concepts through code examples, tutorials, and projects. This document covers projects focused on building agents and retrieval-augmented generation (RAG) systems, as well as providing navigation to all learning materials.

---

## Repository Structure
```
├── workshop/                 # Main package
│   ├── core/                 # Organized ML examples by domain
│   │   ├── sklearn/          # Classical ML (8 algorithms + evaluation)
│   │   ├── pytorch/          # Deep learning (6 domain-specific examples)
│   │   └── numpy/            # ML fundamentals from scratch
│   ├── utils/                # Shared utilities (DRY principle)
│   │   ├── device_utils.py   # Device detection (MPS/CUDA/CPU)
│   │   ├── data_utils.py     # Data loading & preprocessing
│   │   └── eval_utils.py     # Evaluation metrics & visualization
│   ├── cli.py                # CLI interface for running examples
│   └── __init__.py           # Package initialization
├── docs/                     # Comprehensive learning documentation
│   ├── ml_theory.md          # Mathematical foundations
│   ├── numpy.md              # Linear algebra & fundamentals
│   ├── sklearn.md            # ML algorithms & model selection guide
│   ├── pytorch.md            # Deep learning & PEFT/LoRA
│   └── rag.md                # RAG & LLM applications
├── pyproject.toml            # Project configuration with dependencies
├── README.md                 # High-level overview & projects
├── AGENTS.md                 # This file (projects 2-3 + navigation)
└── uv.lock                   # Dependency lock file
```

---

## Quick Navigation to Learning Materials

### Core Concepts & Theory
- **[ML Theory](docs/ml_theory.md)** - Mathematical foundations, optimization, regularization, backpropagation

### Implementation Guides by Domain
- **[NumPy & Fundamentals](docs/numpy.md)** - Linear algebra, matrix operations, numerical computing from scratch
  - Project 5: ML Fundamentals implementation
  - Bonus: Backpropagation from scratch
- **[Scikit-Learn](docs/sklearn.md)** - Classical ML algorithms, hyperparameter tuning, evaluation metrics
  - Project 4: 8 complete algorithm implementations
- **[PyTorch](docs/pytorch.md)** - Deep learning, CNNs, RNNs, Transformers, PEFT/LoRA fine-tuning
  - Project 6: 5 domain-specific neural networks
  - Project 7: Parameter-efficient fine-tuning

### Application Domains
- **[RAG & LLMs](docs/rag.md)** - Retrieval-Augmented Generation, vector stores, prompt optimization

---

## Projects Reference

| Project | Document | Focus |
|---------|----------|-------|
| Project 1 | [README.md](README.md#project-1-custom-text-classifier) | PyTorch fundamentals with BERT |
| Project 2 | [This file - Section below](#project-2-semantic-search-engine-with-rag) | Semantic search with RAG |
| Project 3 | [This file - Section below](#project-3-llm-code-review-agent) | Code review agent with tools |
| Project 4 | [docs/sklearn.md](docs/sklearn.md#project-4-classical-ml-algorithms) | Classical ML algorithms |
| Project 5 | [docs/numpy.md](docs/numpy.md#project-5-ml-fundamentals-with-numpy) | ML fundamentals from scratch |
| Project 6 | [docs/pytorch.md](docs/pytorch.md#project-6-deep-learning-across-multiple-domains) | Deep learning across 5 domains |
| Project 7 | [docs/pytorch.md](docs/pytorch.md#project-7-parameter-efficient-fine-tuning-with-lora) | PEFT/LoRA fine-tuning |
| Bonus | [docs/numpy.md](docs/numpy.md#bonus-backpropagation-from-scratch) + [docs/ml_theory.md](docs/ml_theory.md) | Backpropagation & theory |

---

## Learning Paths

### Beginner: ML Fundamentals
1. Start with **[NumPy](docs/numpy.md)** - understand vectors, matrices, and basic linear algebra
2. Review **[ML Theory](docs/ml_theory.md)** - grasp optimization, loss functions, and regularization
3. Explore **[Scikit-Learn](docs/sklearn.md)** - implement classical algorithms with real datasets

### Intermediate: Deep Learning
1. Study **[PyTorch](docs/pytorch.md)** - build neural networks, understand backpropagation in practice
2. Work through **[Model Selection](docs/model_selection.md)** - choose the right architecture for your problem
3. Implement domain-specific examples (images, text, tabular, time series)

### Advanced: LLMs & Production
1. Master **[PyTorch PEFT/LoRA](docs/pytorch.md#best-practices)** - parameter-efficient fine-tuning for large models
2. Build with **[RAG](docs/rag.md)** - combine retrieval with generative models (see Projects 2-3 below)
3. Deploy and optimize

---

## Project 2: Semantic Search Engine with RAG

A comprehensive RAG application demonstrating retrieval-augmented generation across multiple domains.

**Implementation**: [ragchain](https://github.com/huangsam/ragchain)

### Implementation Details
- **Domains**: Wikipedia, source code repositories, mechanical keyboards knowledge base
- **Stack**: LangChain, LlamaIndex, Chroma (vector store), BM25 (sparse retrieval)
- **LLM Backends**: Ollama (local), Hugging Face (cloud)
- **Architecture**: Hybrid retrieval (dense + sparse embeddings), ranking with RRF

### Key Optimizations
- **Context windows**: Configured based on model capabilities and use case
- **Chunking strategy**: Experimented with different chunk sizes and overlap values
- **Retrieval k**: Balanced between relevance and computational cost
- **Reciprocal Rank Fusion (RRF)**: Combined BM25 and semantic similarity for better results
- **Framework evolution**: Moved from basic LangChain chains to LangGraph for complex workflows

### Evaluation Metrics
- **LLM-as-judge**: Custom evaluation using Claude or GPT-4 to score responses
- **Correctness**: Factual accuracy of retrieved and generated content
- **Faithfulness**: How well responses stick to retrieved context
- **Relevance**: Whether returned documents match query intent

### Key Learnings
- **Trade-offs**: Ollama is simpler to set up but less customizable than Hugging Face models
- **Framework maturity**: LangGraph provides better control than basic LangChain for complex agentic flows
- **Retrieval quality**: Better retrieval often more important than LLM quality
- **Cost efficiency**: Local models (Ollama) reduce API costs for experimentation

### Challenges & Solutions
- **Long contexts**: Addressed through intelligent chunking and retrieval ranking
- **Irrelevant retrievals**: Improved with hybrid search and better semantic embeddings
- **Response quality**: Enhanced through careful prompt engineering and RRF ranking

---

## Project 3: LLM Code Review Agent

**Implementation**: [codebot](https://github.com/huangsam/codebot)

An agentic system that reviews code using RAG patterns and tool integration.

### Implementation Details
- **Purpose**: Automated code review with contextual suggestions
- **Stack**: LangChain tools, Model Context Protocol (MCP)
- **Architecture**: Agent loop with tool calling capability
- **Status**: Complete implementation with real-world GitHub integration

### Agent Capabilities
- **Tool integration**: Leverages MCP servers for code analysis
- **RAG patterns**: Retrieves relevant code standards and examples for context
- **Code understanding**: Applies learned patterns from retrieved documentation
- **Iterative refinement**: Can request clarifications and make multiple passes

### Key Components
- **Tool calling**: Agent decides when and how to use available tools
- **Context retrieval**: Pulls relevant code patterns and standards
- **Analysis**: Combines retrieved knowledge with LLM reasoning
- **Suggestions**: Generates actionable code improvements

### Extension Opportunities
- **Custom MCP servers**: Build domain-specific code analysis tools
- **Enhanced reasoning**: Multi-step agent loops for complex reviews
- **Learning capability**: Build knowledge base from reviewed code patterns
- **Integration**: Connect to GitHub/GitLab for real-world workflows

---

## Architecture Patterns

### RAG Pipeline
```
Query → Retrieval (Dense + Sparse) → Ranking (RRF) → Context Assembly → LLM Generation
```

### Agent Loop
```
Query → Tool Selection → Tool Execution → Observation → Reasoning → Response
```

### Technology Stack Comparison

| Feature | LangChain | LlamaIndex | LangGraph |
|---------|-----------|-----------|-----------|
| **Chains** | Built-in | Built-in | Custom graphs |
| **Agents** | Simple | Basic | Advanced |
| **Flexibility** | Good | Good | Excellent |
| **Learning curve** | Medium | Medium | Steeper |
| **Best for** | Simple RAG | Data indexing | Complex agents |

---

## Lessons Learned

### RAG Success Factors
1. **Retrieval quality > LLM quality**: Better to retrieve right context than have powerful LLM
2. **Hybrid search matters**: BM25 + semantic search outperforms either alone
3. **Chunk size critical**: Too small loses context, too large hurts retrieval
4. **Ranking improves results**: RRF and other ranking methods significantly boost quality

### Agent Development Tips
1. **Start simple**: Basic tool calling before complex multi-step agents
2. **Test tools early**: Ensure tools return reliable, parseable outputs
3. **Prompt engineering**: Clear tool descriptions crucial for agent to use them correctly
4. **Error handling**: Build fallbacks for tool failures and invalid outputs
5. **Iterative refinement**: Start with basic prompts, improve based on failures

### Production Considerations
- **Cost monitoring**: Track LLM API costs and retrieval overhead
- **Latency**: Consider caching for repeated queries
- **Reliability**: Implement retries and fallbacks for tool failures
- **Quality metrics**: Define and monitor success metrics for your use case
- **User feedback**: Build loops to improve from real usage

---

## Key Insights by Domain

| Domain | Key Insight | Reference |
|--------|-------------|-----------|
| **Linear Algebra** | Eigenvalues/eigenvectors power dimensionality reduction | [docs/numpy.md](docs/numpy.md) |
| **Optimization** | SGD + momentum outperforms vanilla gradient descent | [docs/ml_theory.md](docs/ml_theory.md) |
| **Classification** | SVM and Random Forests are effective baselines | [docs/sklearn.md](docs/sklearn.md) |
| **Deep Learning** | RNNs capture temporal patterns; Transformers handle long-range dependencies | [docs/pytorch.md](docs/pytorch.md) |
| **Fine-tuning** | LoRA achieves 95-99% parameter reduction without performance loss | [docs/pytorch.md](docs/pytorch.md) |
| **RAG** | Context window, chunk size, and retriever choice heavily influence quality | [docs/rag.md](docs/rag.md) |

---

## How to Use This Guide

1. **Learning**: Read [README.md](README.md) for high-level overview, then follow learning paths above
2. **Projects**: Navigate to specific project documentation for detailed implementation
3. **Reference**: Use Cmd+F to find specific topics across any document
4. **Implementation**: Link examples back to code in `workshop/core/` folder
5. **Teaching**: Share specific sections with others learning ML/AI

A comprehensive RAG application demonstrating retrieval-augmented generation across multiple domains.

### Implementation Details
- **Domains**: Wikipedia, source code repositories, mechanical keyboards knowledge base
- **Stack**: LangChain, LlamaIndex, Chroma (vector store), BM25 (sparse retrieval)
- **LLM Backends**: Ollama (local), Hugging Face (cloud)
- **Architecture**: Hybrid retrieval (dense + sparse embeddings), ranking with RRF

### Key Optimizations
- **Context windows**: Configured based on model capabilities and use case
- **Chunking strategy**: Experimented with different chunk sizes and overlap values
- **Retrieval k**: Balanced between relevance and computational cost
- **Reciprocal Rank Fusion (RRF)**: Combined BM25 and semantic similarity for better results
- **Framework evolution**: Moved from basic LangChain chains to LangGraph for complex workflows

### Evaluation Metrics
- **LLM-as-judge**: Custom evaluation using Claude or GPT-4 to score responses
- **Correctness**: Factual accuracy of retrieved and generated content
- **Faithfulness**: How well responses stick to retrieved context
- **Relevance**: Whether returned documents match query intent

### Key Learnings
- **Trade-offs**: Ollama is simpler to set up but less customizable than Hugging Face models
- **Framework maturity**: LangGraph provides better control than basic LangChain for complex agentic flows
- **Retrieval quality**: Better retrieval often more important than LLM quality
- **Cost efficiency**: Local models (Ollama) reduce API costs for experimentation

### Challenges & Solutions
- **Long contexts**: Addressed through intelligent chunking and retrieval ranking
- **Irrelevant retrievals**: Improved with hybrid search and better semantic embeddings
- **Response quality**: Enhanced through careful prompt engineering and RRF ranking

---

## Lessons Learned

### RAG Success Factors
1. **Retrieval quality > LLM quality**: Better to retrieve right context than have powerful LLM
2. **Hybrid search matters**: BM25 + semantic search outperforms either alone
3. **Chunk size critical**: Too small loses context, too large hurts retrieval
4. **Ranking improves results**: RRF and other ranking methods significantly boost quality

### Agent Development Tips
1. **Start simple**: Basic tool calling before complex multi-step agents
2. **Test tools early**: Ensure tools return reliable, parseable outputs
3. **Prompt engineering**: Clear tool descriptions crucial for agent to use them correctly
4. **Error handling**: Build fallbacks for tool failures and invalid outputs
5. **Iterative refinement**: Start with basic prompts, improve based on failures

### Production Considerations
- **Cost monitoring**: Track LLM API costs and retrieval overhead
- **Latency**: Consider caching for repeated queries
- **Reliability**: Implement retries and fallbacks for tool failures
- **Quality metrics**: Define and monitor success metrics for your use case
- **User feedback**: Build loops to improve from real usage

---

## Related Documentation
- Theory: See [docs/rag.md](docs/rag.md) for RAG concepts and components
- ML Theory: See [docs/ml_theory.md](docs/ml_theory.md) for underlying optimization
- Full workshop overview: See [README.md](README.md)
