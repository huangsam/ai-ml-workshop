# External Integrations: RAG, Pipelines & Analysis

This document summarizes how external systems and repositories connect to the AI/ML workshop core. These tools handle data at scale and extract features for the models in this repository.

---

## Agentic Systems & RAG

Retrieval-Augmented Generation (RAG) bridges LLMs with domain-specific knowledge. These insights focus on building reliable, context-aware agents.

### Implementation Insights
- **Hybrid Retrieval**: Combining dense embeddings with sparse search (BM25) via **Reciprocal Rank Fusion (RRF)** significantly improves document relevance.
- **Chunking Strategy**: Small chunks lose context, while large chunks dilute signals. Overlapping chunks help preserve continuity.
- **Agent Loops**: Multi-step reasoning (e.g., LangGraph) provides better control for complex tasks like code review than simple linear chains.
- **Evaluation**: Using "LLM-as-judge" allows for automated scoring of faithfulness, relevance, and correctness.

### Architecture Patterns
- **RAG Pipeline**: `Query → Hybrid Retrieval (Dense + Sparse) → RRF Ranking → Context Assembly → LLM Generation`
- **Agentic Loop**: `Query → Tool Selection → Tool Execution → Observation → Reasoning → Iterative Refinement`

### RAG Success Factors
1. **Retrieval > LLM**: A better retriever often has a higher impact on quality than a more powerful LLM.
2. **Context Matters Tuning**: Chunk size and $k$ (number of retrieved docs) are critical for performance and cost.
3. **Hybrid Search**: Always use hybrid search if the domain has specific terminology (e.g., code, medical).

### External Repositories (RAG)
- **[huangsam/ragchain](https://github.com/huangsam/ragchain)**: Full RAG application demonstrating hybrid retrieval and observability.
- **[huangsam/codebot](https://github.com/huangsam/codebot)**: Agentic system for automated code review using function calling and tool integration.

---

## Data Engineering & Pipelines

Large-scale ML requires robust data ingestion and transformation. These projects explore distributed processing frameworks.

### Repositories
- **[huangsam/flink-trial](https://github.com/huangsam/flink-trial)**: Low-latency streaming pipelines with Apache Flink.
- **[huangsam/beam-trial](https://github.com/huangsam/beam-trial)**: Unified batch and streaming processing with Apache Beam.
- **[huangsam/spark-trial](https://github.com/huangsam/spark-trial)**: Distributed analytics and ML preprocessing using PySpark.

---

## Media Analysis & Computer Vision

Specialized tools for converting raw pixels/frames into structured features (JSON/Python objects) for neural networks.

### Repositories
- **[huangsam/vidicant](https://github.com/huangsam/vidicant)**: C++/OpenCV based analysis (brightness, motion) with Python bindings.
- **[huangsam/xcode-trial](https://github.com/huangsam/xcode-trial)**: Swift/Native Apple frameworks optimized for performance on Apple Silicon.

---

## Evaluation & Observability

- **LLM-as-Judge**: Using powerful models to score RAG outputs for faithfulness and relevance.
- **Distance Metrics**: Using Cosine similarity or L2 distance in vector databases (Chroma, FAISS) to measure document relevance.
- **Pipeline Workflow**: `Raw Data → External Analysis (Feature Extraction) → Structured Storage → Workshop Model (Inference)`.
