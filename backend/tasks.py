"""Task runner – resolves (module, task) strings to callable main() functions.

Each entry in TASK_RUNNER_MAP is a lazy import wrapper that:
1. Imports the relevant workshop module.
2. Calls its ``main(hook=hook, config=config)`` with the injected ProgressHook and configuration dict.

ML modules are imported *inside* the function to avoid loading heavy
libraries (torch, transformers, etc.) at server startup.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from workshop.utils.hooks import ProgressHook

# Type alias: a runner is a callable that accepts a hook and config dict and returns None.
TaskRunner = Callable[[ProgressHook, dict[str, Any]], None]


def _numpy_backpropagation(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.numpy.backpropagation import main

    main(hook=hook, config=config)


def _numpy_fundamentals(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.numpy.main import main

    main(hook=hook, config=config)


def _sklearn_linear_regression(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.sklearn.linear_regression import main

    main(hook=hook, config=config)


def _sklearn_logistic_regression(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.sklearn.logistic_regression import main

    main(hook=hook, config=config)


def _sklearn_knn(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.sklearn.knn import main

    main(hook=hook, config=config)


def _sklearn_decision_tree(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.sklearn.decision_tree import main

    main(hook=hook, config=config)


def _sklearn_svm(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.sklearn.svm import main

    main(hook=hook, config=config)


def _sklearn_random_forest(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.sklearn.random_forest import main

    main(hook=hook, config=config)


def _sklearn_kmeans(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.sklearn.kmeans import main

    main(hook=hook, config=config)


def _sklearn_pca(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.sklearn.pca import main

    main(hook=hook, config=config)


def _sklearn_xgboost(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.sklearn.xgboost import main

    main(hook=hook, config=config)


def _pytorch_tabular_classification(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.tabular_classification import main

    main(hook=hook, config=config)


def _pytorch_image_classification(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.image_classification import main

    main(hook=hook, config=config)


def _pytorch_text_classification(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.text_classification import main

    main(hook=hook, config=config)


def _pytorch_time_series_forecasting(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.time_series_forecasting import main

    main(hook=hook, config=config)


def _pytorch_fine_tuning(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.fine_tuning import main

    main(hook=hook, config=config)


def _pytorch_question_answering(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.question_answering import main

    main(hook=hook, config=config)


def _numpy_q_learning(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.numpy.q_learning import main

    main(hook=hook, config=config)


def _numpy_attention(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.numpy.attention import main

    main(hook=hook, config=config)


def _pytorch_cnn(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.cnn import main

    main(hook=hook, config=config)


def _pytorch_gan(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.gan import main

    main(hook=hook, config=config)


def _pytorch_lstm(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.lstm import main

    main(hook=hook, config=config)


def _pytorch_quantization(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.quantization import main

    main(hook=hook, config=config)


def _numpy_transformer(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.numpy.transformer import main

    main(hook=hook, config=config)


def _pytorch_rag(hook: ProgressHook, config: dict[str, Any]) -> None:
    from workshop.core.pytorch.rag import main

    main(hook=hook, config=config)


TASK_RUNNER_MAP: dict[tuple[str, str], TaskRunner] = {
    ("numpy", "backpropagation"): _numpy_backpropagation,
    ("numpy", "fundamentals"): _numpy_fundamentals,
    ("numpy", "q_learning"): _numpy_q_learning,
    ("numpy", "attention"): _numpy_attention,
    ("numpy", "transformer"): _numpy_transformer,
    ("sklearn", "linear_regression"): _sklearn_linear_regression,
    ("sklearn", "logistic_regression"): _sklearn_logistic_regression,
    ("sklearn", "knn"): _sklearn_knn,
    ("sklearn", "decision_tree"): _sklearn_decision_tree,
    ("sklearn", "svm"): _sklearn_svm,
    ("sklearn", "random_forest"): _sklearn_random_forest,
    ("sklearn", "kmeans"): _sklearn_kmeans,
    ("sklearn", "pca"): _sklearn_pca,
    ("sklearn", "xgboost"): _sklearn_xgboost,
    ("pytorch", "tabular_classification"): _pytorch_tabular_classification,
    ("pytorch", "image_classification"): _pytorch_image_classification,
    ("pytorch", "text_classification"): _pytorch_text_classification,
    ("pytorch", "time_series_forecasting"): _pytorch_time_series_forecasting,
    ("pytorch", "fine_tuning"): _pytorch_fine_tuning,
    ("pytorch", "question_answering"): _pytorch_question_answering,
    ("pytorch", "cnn"): _pytorch_cnn,
    ("pytorch", "gan"): _pytorch_gan,
    ("pytorch", "lstm"): _pytorch_lstm,
    ("pytorch", "quantization"): _pytorch_quantization,
    ("pytorch", "rag"): _pytorch_rag,
}
