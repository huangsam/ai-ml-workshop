# External Integrations: RAG, Pipelines & Analysis

This document summarizes how external systems and repositories connect to the AI/ML workshop core. These tools handle data at scale and extract features for the models in this repository.

---

## Agentic Systems & RAG

Retrieval-Augmented Generation (RAG) bridges LLMs with domain-specific knowledge. These insights focus on building reliable, context-aware agents.

### Core Techniques
- **Hybrid Retrieval**: Combining dense embeddings (semantic) with sparse search (keyword/BM25) via **Reciprocal Rank Fusion (RRF)**.
- **Reranking**: Using cross-encoders to refine the top results from the initial retriever for higher precision.
- **Chunking**: Balancing semantic continuity and retrieval granularity. Small chunks lose context; large chunks dilute signals.
- **Agent Loops (ReAct)**: Enabling LLMs to reason and act iteratively using tools (e.g., via LangGraph).

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
