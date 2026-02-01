# Data Pipelines & Distributed Systems

This document covers foundational data engineering concepts for MLOps, focusing on scalable processing frameworks that prepare data for ML workflows.

## Overview
Data pipelines are critical for MLOps, handling ingestion, transformation, and feature engineering at scale. These projects explore Apache Flink, Beam, and Spark for batch/streaming processing, distributed computing, and integration with ML libraries.

## Key Concepts
- **Distributed Processing**: Parallel execution across clusters for large datasets.
- **Streaming vs. Batch**: Real-time (Flink/Beam) vs. historical (Spark) data handling.
- **ETL Patterns**: Extract, Transform, Load with fault tolerance and monitoring.
- **ML Integration**: Preprocessing data for models in sklearn/PyTorch projects.

## Projects
- **Apache Flink Trial**: [huangsam/flink-trial](https://github.com/huangsam/flink-trial) - Streaming pipelines with event-time semantics.
- **Apache Beam Trial**: [huangsam/beam-trial](https://github.com/huangsam/beam-trial) - Portable batch/streaming with unified model.
- **Apache Spark Trial**: [huangsam/spark-trial](https://github.com/huangsam/spark-trial) - Distributed analytics with PySpark and MLlib.

## Learnings
- Trade-offs: Flink for low-latency streaming, Spark for batch analytics, Beam for portability.
- Production Considerations: Monitoring, scaling, and cloud deployment (e.g., GCP Dataflow, AWS EMR).
- Connection to ML: Use these for feature engineering before training models in Projects 4-7.

## Resources
- [Apache Flink Docs](https://flink.apache.org/)
- [Apache Beam Docs](https://beam.apache.org/)
- [Apache Spark Docs](https://spark.apache.org/)
