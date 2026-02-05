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

### Projects 8-9: Data Pipelines & Distributed Processing
To support MLOps and production ML workflows, explored distributed data processing frameworks for scalable ETL, streaming, and batch analytics. These complement ML projects by handling large-scale data ingestion, transformation, and feature engineering before model training.

- **Project 8: Apache Flink & Beam Trials** - Built streaming and batch data pipelines with Flink (event-time processing, state management) and Beam (unified batch/streaming with portability). Demonstrated windowed aggregations, transforms, and integration with Kafka/Dataflow. **[Repositories: huangsam/flink-trial](https://github.com/huangsam/flink-trial)** and **[huangsam/beam-trial](https://github.com/huangsam/beam-trial)**

- **Project 9: Apache Spark Trial** - Developed distributed data processing jobs using PySpark for ETL, MLlib integration, and RDD/DataFrame operations. Optimized for large datasets with partitioning and caching. **[Repository: huangsam/spark-trial](https://github.com/huangsam/spark-trial)**

These projects emphasize distributed systems principles (fault tolerance, scalability) and connect to ML via data preprocessing for Projects 4-7.

### Projects 10-11: Media Analysis & Computer Vision
Explored image and video analysis through two complementary approaches: generic cross-platform implementation and Apple-specific optimization. These projects serve as feature extractors for ML pipelines and demonstrate the trade-offs between portability and platform-specific performance.

- **Project 10: Vidicant** - A cross-platform C++ library with Python bindings using OpenCV for image and video feature extraction (brightness, colors, motion, edge detection). Supports Windows, macOS, and Linux. Demonstrates pybind11 integration for Python interoperability. **[Repository: huangsam/vidicant](https://github.com/huangsam/vidicant)**

- **Project 11: xcode-trial** - A Swift-based multimodal video analysis tool leveraging Apple's native frameworks (Vision Framework, AVFoundation, Core Image) for high-performance analysis on macOS. Extracts faces, scenes, audio, text, colors, and motion with JSON output. **[Repository: huangsam/xcode-trial](https://github.com/huangsam/xcode-trial)**

Key learnings explored generic vs native approaches: cross-platform flexibility with OpenCV vs peak performance with Apple silicon acceleration. Both projects output structured features suitable for ML pipelines (Projects 4-7). **[Full details in docs/media_analysis.md](docs/media_analysis.md)**

## Related Repositories & Broader Portfolio

This AI/ML workshop focuses on machine learning concepts and implementations, plus image/video analysis for feature extraction. For a complete view of my engineering experience, including production deployments and DevOps, check out these related projects:

- **[chowist](https://github.com/huangsam/chowist)**: A Django-based web application for restaurant discovery and reviews, deployed with Docker and Kubernetes.
- **[terraform-aws](https://github.com/huangsam/terraform-aws)**: Infrastructure as Code for AWS deployments, including EC2, RDS, and networking setups.
- **[ansible-vagrant](https://github.com/huangsam/ansible-vagrant)**: Automated provisioning and configuration management using Ansible with Vagrant for local development environments.
- **[spring-demo](https://github.com/huangsam/spring-demo)**: A Spring Boot application in Kotlin, demonstrating microservices architecture and REST APIs.
- **[flink-trial](https://github.com/huangsam/flink-trial)** and **[beam-trial](https://github.com/huangsam/beam-trial)**: Distributed data processing with Apache Flink and Apache Beam
- **[spark-trial](https://github.com/huangsam/spark-trial)**: Distributed ML preprocessing with Apache Spark

These projects complement the ML work here by providing the production engineering foundation needed for MLOps and distributed systems.
