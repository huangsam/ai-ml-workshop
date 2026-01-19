"""AI/ML Workshop - Learn and practice ML concepts through code examples."""

__version__ = "1.0.0"
__author__ = "Workshop Contributors"

from workshop.utils import (
    create_data_loaders,
    get_device,
    load_and_split_data,
    plot_confusion_matrix,
    plot_training_history,
    print_classification_metrics,
    scale_features,
)

__all__ = [
    "get_device",
    "load_and_split_data",
    "scale_features",
    "create_data_loaders",
    "print_classification_metrics",
    "plot_confusion_matrix",
    "plot_training_history",
]
