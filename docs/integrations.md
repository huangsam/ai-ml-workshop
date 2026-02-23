# External Integrations

This document summarizes how external tools and projects bridge to the AI/ML workshop core. These tools serve as scalable feature extractors and data processors for the ML models in this repository.

---

## Data Engineering & Pipelines

Large-scale ML requires robust data ingestion and transformation. These projects explore distributed processing frameworks.

### Repositories
- **[huangsam/flink-trial](https://github.com/huangsam/flink-trial)**: Streaming pipelines with event-time processing and state management using Apache Flink.
- **[huangsam/beam-trial](https://github.com/huangsam/beam-trial)**: Unified batch and streaming data processing with Apache Beam.
- **[huangsam/spark-trial](https://github.com/huangsam/spark-trial)**: Distributed analytics and ML preprocessing using PySpark.

### Key Learnings
- **Flink**: Best for low-latency streaming and complex event processing.
- **Spark**: Industry standard for large-scale batch ETL and historical data analysis.
- **Beam**: Offers portability across different runners (GCP Dataflow, Flink, Spark).
- **ML Connectivity**: Use these for feature engineering (aggregation, normalization) before feeding data into Scikit-Learn or PyTorch models.

---

## Media Analysis & Computer Vision

Specialized tools for extracting structured features from raw image and video content.

### Projects
#### 1. Vidicant (Generic Cross-Platform)
- **Repo**: [huangsam/vidicant](https://github.com/huangsam/vidicant)
- **Stack**: C++ & OpenCV with Python bindings via `pybind11`.
- **Focus**: Consistent analysis (brightness, colors, motion) across Windows, macOS, and Linux.

#### 2. xcode-trial (Apple-Specific)
- **Repo**: [huangsam/xcode-trial](https://github.com/huangsam/xcode-trial)
- **Stack**: Swift & Native Apple Frameworks (Vision, AVFoundation).
- **Focus**: High-performance multimodal analysis optimized for Apple silicon.

### ML Pipeline Integration
Both tools serve as **feature extractors**. They convert raw pixels/frames into structured JSON or Python objects that can be directly consumed by the neural networks in `workshop/core/pytorch/`.

Example workflow:
`Raw Video → xcode-trial (Feature Extraction) → Structured JSON → PyTorch LSTM (Time Series forecasting)`
