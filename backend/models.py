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


# ---------------------------------------------------------------------------
# Registry: maps (module, task) -> config class
# ---------------------------------------------------------------------------

TASK_CONFIG_MAP: dict[tuple[str, str], type[BaseModel]] = {
    ("numpy", "backpropagation"): BackpropagationConfig,
    ("numpy", "fundamentals"): NumpyFundamentalsConfig,
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
}
