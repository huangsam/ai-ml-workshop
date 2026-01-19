"""Device management utilities for ML examples."""

import torch


def get_device() -> str:
    """
    Detect and return the best available device for computation.

    Returns:
        Device string: "mps" (Apple Silicon), "cuda" (NVIDIA), or "cpu"
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
