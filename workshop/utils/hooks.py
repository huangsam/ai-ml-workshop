"""Progress hook abstractions for ML task instrumentation.

Provides a structural Protocol (ProgressHook) that decouples ML tasks from
their execution environment (CLI, API worker, tests).  Any object that
implements ``update_stage`` and ``update_metrics`` satisfies the protocol.

Concrete implementations:
- ``NoOpProgressHook``   – silent no-op used as the default when no hook is supplied.
- ``ConsoleProgressHook`` – structured terminal output for CLI usage.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressHook(Protocol):
    """Structural contract for reporting task progress.

    ML task modules accept an optional ``hook: ProgressHook`` parameter.
    They call these methods at key milestones without knowing whether the hook
    writes to a terminal, an HTTP state registry, or does nothing at all.
    """

    def update_stage(self, stage_name: str, percentage: float) -> None:
        """Report a macroscopic lifecycle phase.

        Args:
            stage_name: Human-readable label for the current stage
                        (e.g. "Training", "Evaluation").
            percentage: Overall completion percentage in [0, 100].
        """
        ...

    def update_metrics(self, metrics: dict) -> None:
        """Stream fast-updating numerical parameters.

        Args:
            metrics: Key/value pairs such as
                     ``{"epoch": 5, "loss": 0.312, "accuracy": 0.87}``.
        """
        ...

    def is_cancelled(self) -> bool:
        """Check if the execution has been cancelled by the caller.

        Returns:
            True if execution should be aborted, False otherwise.
        """
        ...


class NoOpProgressHook:
    """Silent hook used as the default when no hook is provided.

    Calling either method is a no-op, preserving existing task output.
    """

    def update_stage(self, stage_name: str, percentage: float) -> None:  # noqa: ARG002
        pass

    def update_metrics(self, metrics: dict) -> None:  # noqa: ARG002
        pass

    def is_cancelled(self) -> bool:
        return False


class ConsoleProgressHook:
    """Structured terminal progress reporter for CLI usage.

    Formats updates as::

        [Stage 2/5] Training... 40%
          metrics: epoch=5, loss=0.3124

    Args:
        total_stages: Expected number of stages, used only for display.
    """

    def __init__(self, total_stages: int = 5) -> None:
        self._total = total_stages
        self._current = 0
        self._last_stage = None

    def update_stage(self, stage_name: str, percentage: float) -> None:
        if stage_name != self._last_stage:
            self._current += 1
            self._last_stage = stage_name
        print(f"[Stage {self._current}/{self._total}] {stage_name}... {percentage:.0f}%")

    def update_metrics(self, metrics: dict) -> None:
        parts = []
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        print(f"  metrics: {', '.join(parts)}")

    def is_cancelled(self) -> bool:
        return False
