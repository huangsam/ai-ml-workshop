"""CLI interface for running workshop examples."""

import sys
from pathlib import Path

import click

# Add workshop to path if running from repo root
sys.path.insert(0, str(Path(__file__).parent))


@click.group()
@click.version_option(package_name="ai-ml-workshop")
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


def _hook(total_stages: int = 5):
    """Return a ConsoleProgressHook configured for the given number of stages."""
    from workshop.utils.hooks import ConsoleProgressHook

    return ConsoleProgressHook(total_stages=total_stages)


@pytorch.command()
def fine_tuning():
    """Run PEFT/LoRA fine-tuning example."""
    from workshop.core.pytorch.fine_tuning import main

    main(hook=_hook(5))


@pytorch.command()
def image_classification():
    """Run image classification (ResNet-18) example."""
    from workshop.core.pytorch.image_classification import main

    main(hook=_hook(5))


@pytorch.command()
def text_classification():
    """Run text classification (BERT) example."""
    from workshop.core.pytorch.text_classification import main

    main(hook=_hook(5))


@pytorch.command()
def question_answering():
    """Run question answering (SQuAD) example."""
    from workshop.core.pytorch.question_answering import main

    main(hook=_hook(5))


@pytorch.command()
def time_series_forecasting():
    """Run time series forecasting (LSTM) example."""
    from workshop.core.pytorch.time_series_forecasting import main

    main(hook=_hook(5))


@pytorch.command()
def tabular_classification():
    """Run tabular classification (MLP) example."""
    from workshop.core.pytorch.tabular_classification import main

    main(hook=_hook(5))


@pytorch.command()
def cnn():
    """Run Convolutional Neural Network (CNN) example."""
    from workshop.core.pytorch.cnn import main

    main(hook=_hook(6))


@pytorch.command()
def gan():
    """Run Generative Adversarial Network (GAN) example."""
    from workshop.core.pytorch.gan import main

    main(hook=_hook(6))


@pytorch.command()
def lstm():
    """Run character-level LSTM text generation example."""
    from workshop.core.pytorch.lstm import main

    main(hook=_hook(6))


@pytorch.command()
def quantization():
    """Run PyTorch dynamic quantization example."""
    from workshop.core.pytorch.quantization import main

    main(hook=_hook(5))


@pytorch.command()
def rag():
    """Run Retrieval-Augmented Generation (RAG) example."""
    from workshop.core.pytorch.rag import main

    main(hook=_hook(8))


@sklearn.command()
def linear_regression():
    """Run linear regression example."""
    from workshop.core.sklearn.linear_regression import main

    main(hook=_hook(4))


@sklearn.command()
def logistic_regression():
    """Run logistic regression example."""
    from workshop.core.sklearn.logistic_regression import main

    main(hook=_hook(4))


@sklearn.command()
def knn():
    """Run K-nearest neighbors example."""
    from workshop.core.sklearn.knn import main

    main(hook=_hook(4))


@sklearn.command()
def decision_tree():
    """Run decision tree example."""
    from workshop.core.sklearn.decision_tree import main

    main(hook=_hook(4))


@sklearn.command()
def svm():
    """Run support vector machine example."""
    from workshop.core.sklearn.svm import main

    main(hook=_hook(4))


@sklearn.command()
def random_forest():
    """Run random forest example."""
    from workshop.core.sklearn.random_forest import main

    main(hook=_hook(4))


@sklearn.command()
def kmeans():
    """Run K-means clustering example."""
    from workshop.core.sklearn.kmeans import main

    main(hook=_hook(4))


@sklearn.command()
def pca():
    """Run principal component analysis example."""
    from workshop.core.sklearn.pca import main

    main(hook=_hook(4))


@sklearn.command()
def xgboost():
    """Run XGBoost example."""
    from workshop.core.sklearn.xgboost import main

    main(hook=_hook(4))


@numpy.command()
def fundamentals():
    """Run NumPy fundamentals example."""
    from workshop.core.numpy.main import main

    main(hook=_hook(6))


@numpy.command()
def backpropagation():
    """Run backpropagation from scratch example."""
    from workshop.core.numpy.backpropagation import main

    main(hook=_hook(6))


@numpy.command()
def q_learning():
    """Run Q-Learning GridWorld maze navigation example."""
    from workshop.core.numpy.q_learning import main

    main(hook=_hook(5))


@numpy.command()
def attention():
    """Run single-head self-attention layer example."""
    from workshop.core.numpy.attention import main

    main(hook=_hook(6))


@numpy.command()
def transformer():
    """Run causal Transformer block example."""
    from workshop.core.numpy.transformer import main

    main(hook=_hook(6))


@cli.command()
@click.option("--port", default=8000, help="Port to run the server on.")
@click.option("--reload/--no-reload", default=True, help="Enable or disable auto-reload.")
def server(port, reload):
    """Start the FastAPI backend server with auto-reload."""
    import os

    import uvicorn

    # Suppress the resource tracker warnings in this process and all spawned subprocesses
    os.environ["PYTHONWARNINGS"] = "ignore:resource_tracker:UserWarning"

    uvicorn.run("backend.main:app", host="127.0.0.1", port=port, reload=reload)


if __name__ == "__main__":
    cli()
