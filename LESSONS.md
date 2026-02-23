# AI/ML Workshop: Key Learnings

This document summarizes the key takeaways, architecture patterns, and implementation insights gained from the projects in this workshop.

---

## Quick Navigation

### Implementation Guides
- **[Fundamentals](docs/fundamentals.md)**: Math theory and NumPy scratch implementations.
- **[Frameworks](docs/frameworks.md)**: Classical ML (Scikit-Learn) and Deep Learning (PyTorch).
- **[Integrations](docs/integrations.md)**: RAG, Data Pipelines, and Media Analysis.
- **[RAG & LLMs](LESSONS.md#agentic-systems--rag)**: Retrieval-Augmented Generation and agentic systems.

---

## Agentic Systems & RAG

Focuses on combining retrieval with generative models to build intelligent agents.

### Implementation Insights
- **Hybrid Retrieval**: Combining dense embeddings with sparse search (BM25) via **Reciprocal Rank Fusion (RRF)** significantly improves document relevance.
- **Chunking Strategy**: Small chunks lose context, while large chunks dilute signals. Overlapping chunks help preserve continuity.
- **Agent Loops**: Multi-step reasoning (e.g., LangGraph) provides much better control for complex tasks like code review than simple linear chains.
- **Evaluation**: Using "LLM-as-judge" (e.g., Claude or GPT-4) allows for automated scoring of faithfulness, relevance, and correctness.

### Technology Stack
- **LangChain/LlamaIndex**: Excellent for rapid prototyping of RAG pipelines.
- **LangGraph**: Preferred for stateful, multi-agent workflows requiring iterative refinement.
- **Ollama**: Ideal for local experimentation and cost-sensitive development.

---

## Architecture Patterns

### RAG Pipeline
`Query → Hybrid Retrieval (Dense + Sparse) → RRF Ranking → Context Assembly → LLM Generation`

### Agentic Loop
`Query → Tool Selection → Tool Execution → Observation → Reasoning → Iterative Refinement`

---

## Lessons Learned

### RAG Success Factors
1. **Retrieval > LLM**: A better retriever often has a higher impact on quality than a more powerful LLM.
2. **Context Matters**: Tuning chunk size and `k` (number of retrieved docs) is critical for performance and cost.
3. **Hybrid Search**: Always use hybrid search if the domain has specific terminology (e.g., code, medical).

### Agent Development
1. **Tool Reliability**: Ensure tools return structured, predictable outputs for the agent to parse correctly.
2. **Clear Tool Descriptions**: The LLM relies heavily on tool names and descriptions to decide when to use them.
3. **Error Handling**: Build robust fallbacks for when tools fail or return invalid data to prevent agent loop crashes.

### Production Considerations
- **Latency vs. Accuracy**: Local models (Ollama) offer privacy and low cost but may trade off reasoning speed/accuracy.
- **Observability**: Track retrieval hits/misses and LLM token usage to optimize for both quality and cost.

---

## Key Insights by Domain

| Domain | Key Takeaway |
|--------|--------------|
| **Linear Algebra** | Dimensionality reduction (PCA) is powered by eigenvalues/vectors. |
| **Optimization** | SGD + momentum is the workhorse of deep learning. |
| **Classical ML** | SVM and Random Forests remain strong baselines for tabular data. |
| **Deep Learning** | Transformers' attention mechanism is transformative for long-range dependencies. |
| **Fine-tuning** | LoRA can reduce trainable parameters by >98% with minimal performance loss. |
| **RAG** | The quality of the vector database and embedding model is the foundation of RAG. |

---

## Related Documentation
- [README.md](README.md): High-level project overview and setup.
- [docs/integrations.md](docs/integrations.md): Data engineering and media analysis tools.
