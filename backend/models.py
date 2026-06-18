"""Pydantic models for task configuration payloads.

Each task config class defines the hyperparameters accepted via the
POST /run/{module}/{task} endpoint.  All fields have sensible defaults so
the frontend can fire requests without filling every field.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# NumPy tasks
# ---------------------------------------------------------------------------


class BackpropagationConfig(BaseModel):
    num_samples: int = Field(default=50, ge=10, le=500, description="XOR samples per class")
    num_epochs: int = Field(default=100, ge=10, le=1000, description="Training epochs")
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0, description="Gradient descent learning rate")
    hidden_size: int = Field(default=4, ge=2, le=64, description="Hidden layer width")


class NumpyFundamentalsConfig(BaseModel):
    """Numpy fundamentals has no tunable hyperparameters."""

    pass


class QLearningConfig(BaseModel):
    epochs: int = Field(default=200, ge=10, le=2000, description="Number of episodes")
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0, description="Q-value updates learning rate")
    discount_factor: float = Field(default=0.9, ge=0.5, le=0.99, description="Discount factor gamma")
    epsilon: float = Field(default=0.9, ge=0.0, le=1.0, description="Initial exploration rate epsilon")
    epsilon_decay: float = Field(default=0.98, ge=0.8, le=0.999, description="Epsilon decay rate")


class SelfAttentionConfig(BaseModel):
    epochs: int = Field(default=150, ge=10, le=500, description="Training epochs")
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0, description="Gradient descent learning rate")
    embedding_dim: int = Field(default=16, ge=4, le=128, description="Embedding vector size")
    sequence_length: int = Field(default=6, ge=2, le=20, description="Sequence length")


# ---------------------------------------------------------------------------
# Scikit-learn tasks
# ---------------------------------------------------------------------------


class LinearRegressionConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=5, ge=2, le=10)


class LogisticRegressionConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=5, ge=2, le=10)
    n_iter: int = Field(default=20, ge=5, le=100)


class KNNConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=5, ge=2, le=10)
    n_iter: int = Field(default=20, ge=5, le=100)


class DecisionTreeConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=5, ge=2, le=10)
    n_iter: int = Field(default=20, ge=5, le=100)


class SVMConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=5, ge=2, le=10)
    n_iter: int = Field(default=20, ge=5, le=100)


class RandomForestConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    cv_folds: int = Field(default=5, ge=2, le=10)
    n_iter: int = Field(default=20, ge=5, le=100)


class KMeansConfig(BaseModel):
    n_clusters: int = Field(default=3, ge=2, le=20)
    max_iter: int = Field(default=300, ge=10, le=1000)


class PCAConfig(BaseModel):
    n_components: int = Field(default=2, ge=1, le=10)


class XGBoostConfig(BaseModel):
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    n_estimators: int = Field(default=100, ge=10, le=500)
    learning_rate: float = Field(default=0.1, gt=0.0, le=1.0)
    max_depth: int = Field(default=6, ge=1, le=15)
    cv_folds: int = Field(default=5, ge=2, le=10)
    n_iter: int = Field(default=20, ge=5, le=100)


# ---------------------------------------------------------------------------
# PyTorch tasks
# ---------------------------------------------------------------------------


class TabularClassificationConfig(BaseModel):
    num_epochs: int = Field(default=20, ge=1, le=200)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    batch_size: int = Field(default=32, ge=8, le=512)


class ImageClassificationConfig(BaseModel):
    num_epochs: int = Field(default=5, ge=1, le=50)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    batch_size: int = Field(default=64, ge=8, le=512)


class TextClassificationConfig(BaseModel):
    num_epochs: int = Field(default=2, ge=1, le=20)
    learning_rate: float = Field(default=2e-5, gt=0.0, le=1.0)
    batch_size: int = Field(default=16, ge=4, le=128)


class TimeSeriesForecastingConfig(BaseModel):
    num_epochs: int = Field(default=10, ge=5, le=500)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    sequence_length: int = Field(default=24, ge=5, le=200)
    batch_size: int = Field(default=32, ge=8, le=512)


class FineTuningConfig(BaseModel):
    num_epochs: int = Field(default=3, ge=1, le=20)
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0)
    lora_r: int = Field(default=8, ge=1, le=64)
    lora_alpha: int = Field(default=16, ge=1, le=128)
    batch_size: int = Field(default=32, ge=4, le=128)


class QuestionAnsweringConfig(BaseModel):
    max_length: int = Field(default=384, ge=64, le=1024)
    num_samples: int = Field(default=5, ge=1, le=50)


class CNNConfig(BaseModel):
    epochs: int = Field(default=5, ge=1, le=20, description="Training epochs")
    batch_size: int = Field(default=32, ge=8, le=256, description="Training batch size")
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0, description="Learning rate")
    filter_count: int = Field(default=8, ge=2, le=64, description="Channels in first conv layer")


class GANConfig(BaseModel):
    epochs: int = Field(default=150, ge=10, le=500, description="Training epochs")
    latent_dim: int = Field(default=8, ge=2, le=64, description="Generator noise vector size")
    learning_rate: float = Field(default=0.001, gt=0.0, le=0.1, description="Learning rate")


class LSTMConfig(BaseModel):
    epochs: int = Field(default=10, ge=1, le=100, description="Training epochs")
    hidden_dim: int = Field(default=64, ge=8, le=512, description="LSTM hidden state dimension")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Softmax sampling temperature")


class QuantizationConfig(BaseModel):
    num_samples: int = Field(default=500, ge=10, le=2000, description="Evaluation samples")


class TransformerScratchConfig(BaseModel):
    epochs: int = Field(default=120, ge=10, le=500, description="Training epochs")
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0, description="Learning rate")
    embedding_dim: int = Field(default=16, ge=8, le=64, description="Embedding vector size (must be divisible by heads)")
    num_heads: int = Field(default=2, ge=1, le=4, description="Number of attention heads")
    hidden_dim: int = Field(default=32, ge=8, le=128, description="Feedforward network hidden dimension")


class RAGConfig(BaseModel):
    query_index: int = Field(default=1, ge=1, le=5, description="Query index (1: Antigravity, 2: Workshop, 3: Project Triton, 4: Babbage, 5: Arthur Samuel)")
    top_k: int = Field(default=2, ge=1, le=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum cosine similarity for retrieval")


# ---------------------------------------------------------------------------
# Registry: maps (module, task) -> config class
# ---------------------------------------------------------------------------

TASK_CONFIG_MAP: dict[tuple[str, str], type[BaseModel]] = {
    ("numpy", "backpropagation"): BackpropagationConfig,
    ("numpy", "fundamentals"): NumpyFundamentalsConfig,
    ("numpy", "q_learning"): QLearningConfig,
    ("numpy", "attention"): SelfAttentionConfig,
    ("numpy", "transformer"): TransformerScratchConfig,
    ("sklearn", "linear_regression"): LinearRegressionConfig,
    ("sklearn", "logistic_regression"): LogisticRegressionConfig,
    ("sklearn", "knn"): KNNConfig,
    ("sklearn", "decision_tree"): DecisionTreeConfig,
    ("sklearn", "svm"): SVMConfig,
    ("sklearn", "random_forest"): RandomForestConfig,
    ("sklearn", "kmeans"): KMeansConfig,
    ("sklearn", "pca"): PCAConfig,
    ("sklearn", "xgboost"): XGBoostConfig,
    ("pytorch", "tabular_classification"): TabularClassificationConfig,
    ("pytorch", "image_classification"): ImageClassificationConfig,
    ("pytorch", "text_classification"): TextClassificationConfig,
    ("pytorch", "time_series_forecasting"): TimeSeriesForecastingConfig,
    ("pytorch", "fine_tuning"): FineTuningConfig,
    ("pytorch", "question_answering"): QuestionAnsweringConfig,
    ("pytorch", "cnn"): CNNConfig,
    ("pytorch", "gan"): GANConfig,
    ("pytorch", "lstm"): LSTMConfig,
    ("pytorch", "quantization"): QuantizationConfig,
    ("pytorch", "rag"): RAGConfig,
}
