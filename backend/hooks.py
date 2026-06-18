"""HTTPProgressHook – writes directly to the in-memory job registry.

This is the web-tier implementation of the ProgressHook protocol.
ML tasks remain entirely unaware of HTTP; they simply call the hook methods,
and this class routes those calls into the shared job state dictionary.
"""

from __future__ import annotations

from backend import registry


class HTTPProgressHook:
    """ProgressHook implementation that persists state to the job registry.

    Args:
        job_id: The UUID string identifying the active job.
    """

    def __init__(self, job_id: str) -> None:
        self._job_id = job_id

    def update_stage(self, stage_name: str, percentage: float) -> None:
        """Persist stage name and overall percentage to the registry."""
        if self.is_cancelled():
            return
        registry.update_job(
            self._job_id,
            stage=stage_name,
            percentage=percentage,
            status="RUNNING",
        )

    def update_metrics(self, metrics: dict) -> None:
        """Append a metrics snapshot to the job's time-series list."""
        registry.append_metrics(self._job_id, metrics)

    def is_cancelled(self) -> bool:
        """Check if the job has been cancelled by the user."""
        job = registry.get_job(self._job_id)
        return job is not None and job.get("status") == "CANCELLED"

    def save_plot(self, fname: str, *args, **kwargs) -> None:
        """Capture the current figure as bytes and save to registry."""
        import io
        import os

        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        if "format" not in kwargs:
            kwargs["format"] = "png"
        plt.gcf().savefig(buf, *args, **kwargs)
        buf.seek(0)
        image_bytes = buf.getvalue()

        filename = os.path.basename(str(fname))
        registry.save_job_plot(self._job_id, filename, image_bytes)
