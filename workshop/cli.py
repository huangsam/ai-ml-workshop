"""CLI interface for running workshop examples."""

import sys
from pathlib import Path

import click

# Add workshop to path if running from repo root
sys.path.insert(0, str(Path(__file__).parent))


@click.group()
@click.version_option()
def cli():
    """AI/ML Workshop - Run examples and experiments."""
    pass


@cli.group()
def pytorch():
    """PyTorch deep learning examples."""
    pass


@cli.group()
def sklearn():
    """Scikit-learn classical ML examples."""
    pass


@cli.group()
def numpy():
    """NumPy fundamentals and backpropagation."""
    pass


@pytorch.command()
def fine_tuning():
    """Run PEFT/LoRA fine-tuning example."""
    from workshop.core.pytorch.fine_tuning import main

    main()


@pytorch.command()
def image_classification():
    """Run image classification (ResNet-18) example."""
    from workshop.core.pytorch.image_classification import main

    main()


@pytorch.command()
def text_classification():
    """Run text classification (BERT) example."""
    from workshop.core.pytorch.text_classification import main

    main()


@pytorch.command()
def question_answering():
    """Run question answering (SQuAD) example."""
    from workshop.core.pytorch.question_answering import main

    main()


@pytorch.command()
def time_series_forecasting():
    """Run time series forecasting (LSTM) example."""
    from workshop.core.pytorch.time_series_forecasting import main

    main()


@pytorch.command()
def tabular_classification():
    """Run tabular classification (MLP) example."""
    from workshop.core.pytorch.tabular_classification import main

    main()


@sklearn.command()
def linear_regression():
    """Run linear regression example."""
    from workshop.core.sklearn.linear_regression import main

    main()


@sklearn.command()
def logistic_regression():
    """Run logistic regression example."""
    from workshop.core.sklearn.logistic_regression import main

    main()


@sklearn.command()
def knn():
    """Run K-nearest neighbors example."""
    from workshop.core.sklearn.knn import main

    main()


@sklearn.command()
def decision_tree():
    """Run decision tree example."""
    from workshop.core.sklearn.decision_tree import main

    main()


@sklearn.command()
def svm():
    """Run support vector machine example."""
    from workshop.core.sklearn.svm import main

    main()


@sklearn.command()
def random_forest():
    """Run random forest example."""
    from workshop.core.sklearn.random_forest import main

    main()


@sklearn.command()
def kmeans():
    """Run K-means clustering example."""
    from workshop.core.sklearn.kmeans import main

    main()


@sklearn.command()
def pca():
    """Run principal component analysis example."""
    from workshop.core.sklearn.pca import main

    main()


@sklearn.command()
def xgboost():
    """Run XGBoost example."""
    from workshop.core.sklearn.xgboost import main

    main()


@numpy.command()
def fundamentals():
    """Run NumPy fundamentals example."""
    from workshop.core.numpy.main import main

    main()


@numpy.command()
def backpropagation():
    """Run backpropagation from scratch example."""
    from workshop.core.numpy.backpropagation import main

    main()


if __name__ == "__main__":
    cli()
