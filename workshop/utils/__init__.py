"""Shared utilities for ML examples across the workshop."""

from workshop.utils.data_utils import create_data_loaders, load_and_split_data, scale_features
from workshop.utils.device_utils import get_device
from workshop.utils.eval_utils import plot_confusion_matrix, plot_training_history, print_classification_metrics

__all__ = [
    "get_device",
    "load_and_split_data",
    "scale_features",
    "create_data_loaders",
    "print_classification_metrics",
    "plot_confusion_matrix",
    "plot_training_history",
]
