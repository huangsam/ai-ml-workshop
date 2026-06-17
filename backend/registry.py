"""Thread-safe in-memory job state registry.

Each job entry is a plain dict with the following shape::

    {
        "status":     "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED",
        "stage":      str,          # current stage label
        "percentage": float,        # 0-100
        "metrics":    list[dict],   # time-series of metric snapshots
        "error":      str | None,   # exception message on failure
        "created_at": float,        # time.time() timestamp
    }
"""

from __future__ import annotations

import threading
import time
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def create_job(job_id: str) -> dict[str, Any]:
    """Initialise a new job entry and return it."""
    entry: dict[str, Any] = {
        "status": "PENDING",
        "stage": "",
        "percentage": 0.0,
        "metrics": [],
        "plots": {},
        "error": None,
        "created_at": time.time(),
    }
    with _lock:
        _jobs[job_id] = entry
    return entry


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return a copy of the job dict or *None* if not found.

    Copying prevents concurrent modifications of the dictionary and its metrics
    list by background threads from affecting the caller.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        copied = job.copy()
        copied["metrics"] = list(job["metrics"])
        if "plots" in copied:
            del copied["plots"]
        return copied


def get_job_metrics_slice(job_id: str, start: int) -> list[dict[str, Any]]:
    """Return a thread-safe copy of metrics from *start* index onwards."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return []
        return list(job["metrics"][start:])


def update_job(job_id: str, **kwargs: Any) -> None:
    """Atomically update one or more fields of an existing job."""
    with _lock:
        if job_id in _jobs:
            job = _jobs[job_id]
            if job["status"] in ("CANCELLED", "COMPLETED", "FAILED"):
                return
            job.update(kwargs)


def append_metrics(job_id: str, metrics: dict) -> None:
    """Append a metrics snapshot to the job's time-series list."""
    with _lock:
        if job_id in _jobs:
            if _jobs[job_id]["status"] in ("CANCELLED", "COMPLETED", "FAILED"):
                return
            _jobs[job_id]["metrics"].append(metrics)


def cancel_job(job_id: str) -> bool:
    """Set the job status to CANCELLED if it is running or pending."""
    with _lock:
        if job_id in _jobs:
            job = _jobs[job_id]
            if job["status"] in ("PENDING", "RUNNING"):
                job["status"] = "CANCELLED"
                return True
        return False


def cancel_active_jobs(exclude_job_id: str | None = None) -> list[str]:
    """Cancel any active PENDING or RUNNING jobs, returning their job IDs."""
    cancelled = []
    with _lock:
        for jid, job in _jobs.items():
            if jid != exclude_job_id and job["status"] in ("PENDING", "RUNNING"):
                job["status"] = "CANCELLED"
                cancelled.append(jid)
    return cancelled


def clean_old_jobs(max_age_seconds: int = 3600) -> None:
    """Remove jobs created more than *max_age_seconds* ago."""
    now = time.time()
    with _lock:
        to_remove = [jid for jid, job in _jobs.items() if now - job.get("created_at", 0.0) > max_age_seconds]
        for jid in to_remove:
            del _jobs[jid]


def save_job_plot(job_id: str, filename: str, image_bytes: bytes) -> None:
    """Save plot image bytes to the job registry."""
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["plots"][filename] = image_bytes


def get_job_plot(job_id: str, filename: str) -> bytes | None:
    """Retrieve plot image bytes from the job registry."""
    with _lock:
        job = _jobs.get(job_id)
        if job and "plots" in job:
            return job["plots"].get(filename)
        return None
