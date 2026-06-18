"""FastAPI application entry point.

Run the server::

    uv run uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import os

# Prevent OpenMP thread-pool crashes on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import matplotlib

matplotlib.use("Agg")

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend import registry
from backend.hooks import HTTPProgressHook
from backend.models import TASK_CONFIG_MAP
from backend.tasks import TASK_RUNNER_MAP

# Bounded thread pool scaled to the number of system cores (leaving at least 1 core for async loop/OS if possible).
_NUM_CORES = os.cpu_count() or 2
_MAX_WORKERS = max(1, _NUM_CORES - 1)
_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)


async def _clean_old_jobs_loop() -> None:
    """Background loop that purges jobs older than 1 hour (3600 seconds) from memory."""
    while True:
        try:
            registry.clean_old_jobs(max_age_seconds=3600)
        except Exception:
            pass
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the thread pool executor and background cleanup task lifetime."""
    cleanup_task = asyncio.create_task(_clean_old_jobs_loop())
    yield
    cleanup_task.cancel()
    _executor.shutdown(wait=True)


app = FastAPI(title="ML Workshop API", version="0.1.0", lifespan=lifespan)

# Allow the Next.js dev server (port 3000) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dynamic stages configuration map
_TASK_STAGES_MAP = {
    ("numpy", "backpropagation"): ["Data Generation", "Model Initialization", "Training", "Evaluation", "Visualization", "Complete"],
    ("numpy", "fundamentals"): [
        "Vector Operations",
        "Matrix Operations",
        "Linear Algebra",
        "Linear Regression",
        "Cost Function Visualization",
        "Feature Scaling",
        "Complete",
    ],
    ("sklearn", "linear_regression"): ["Data Loading", "Model Training", "Evaluation", "Complete"],
    ("sklearn", "logistic_regression"): ["Data Loading", "Model Training", "Evaluation", "Complete"],
    ("sklearn", "knn"): ["Data Loading", "Model Training", "Evaluation", "Complete"],
    ("sklearn", "decision_tree"): ["Data Loading", "Model Training", "Evaluation", "Complete"],
    ("sklearn", "svm"): ["Data Loading", "Model Training", "Evaluation", "Complete"],
    ("sklearn", "random_forest"): ["Data Loading", "Model Training", "Evaluation", "Complete"],
    ("sklearn", "kmeans"): ["Data Loading", "Model Training", "Evaluation", "Complete"],
    ("sklearn", "pca"): ["Data Loading", "Dimensionality Reduction", "Evaluation", "Complete"],
    ("sklearn", "xgboost"): ["Data Loading", "Model Training", "Evaluation", "Complete"],
    ("pytorch", "tabular_classification"): ["Data Loading", "Model Setup", "Training", "Evaluation", "Complete"],
    ("pytorch", "image_classification"): ["Data Loading", "Model Setup", "Training", "Evaluation", "Complete"],
    ("pytorch", "text_classification"): ["Data Loading", "Model Setup", "Training", "Evaluation", "Complete"],
    ("pytorch", "time_series_forecasting"): ["Data Generation", "Model Setup", "Training", "Evaluation", "Complete"],
    ("pytorch", "fine_tuning"): ["Model Loading", "LoRA Configuration", "Training", "Evaluation", "Complete"],
    ("pytorch", "question_answering"): ["Model Loading", "Data Processing", "Training", "Evaluation", "Complete"],
    ("numpy", "q_learning"): ["Environment Setup", "Agent Training", "Policy Evaluation", "Visualization", "Complete"],
    ("numpy", "attention"): ["Vector Embeddings", "Attention Computation", "Optimization", "Weights Extraction", "Visualization", "Complete"],
    ("pytorch", "cnn"): ["Data Ingestion", "Model Initialization", "Training", "Testing", "Filter Extraction", "Complete"],
    ("pytorch", "gan"): ["Distribution Setup", "Model Initialization", "Adversarial Training", "Sampling", "Visualization", "Complete"],
    ("pytorch", "lstm"): ["Text Tokenization", "Model Setup", "Training", "Sampling Text", "Visualization", "Complete"],
    ("pytorch", "quantization"): ["Baseline Evaluation", "Dynamic Quantization", "Quantized Evaluation", "Metrics Comparison", "Visualization", "Complete"],
    ("numpy", "transformer"): ["Tokenization & Setup", "Embeddings & PE", "Training Loop", "Text Generation", "Visualization", "Complete"],
    ("pytorch", "rag"): [
        "Loading Models",
        "Encoding Documents",
        "Processing Query",
        "Retrieval & Search",
        "Generation Without Context",
        "Generation With Context",
        "Visualization",
        "Complete",
    ],
}

# Dictionary mapping tasks to their expected plot filenames
_TASK_PLOTS_MAP = {
    ("numpy", "backpropagation"): ["backpropagation_results.png"],
    ("numpy", "q_learning"): ["q_learning_grid_path.png", "q_learning_policy_map.png"],
    ("numpy", "attention"): ["attention_weights_matrix.png"],
    ("numpy", "transformer"): ["transformer_attention_heads.png"],
    ("sklearn", "linear_regression"): ["linear_regression_results.png"],
    ("sklearn", "logistic_regression"): ["logistic_regression_confusion_matrix.png"],
    ("sklearn", "knn"): ["knn_confusion_matrix.png", "knn_accuracy_vs_k.png"],
    ("sklearn", "decision_tree"): ["decision_tree_visualization.png", "decision_tree_confusion_matrix.png"],
    ("sklearn", "svm"): ["svm_confusion_matrix.png", "svm_accuracy_vs_kernel.png"],
    ("sklearn", "random_forest"): ["random_forest_confusion_matrix.png", "random_forest_feature_importance.png"],
    ("sklearn", "kmeans"): ["kmeans_clustering_results.png", "kmeans_elbow_plot.png"],
    ("sklearn", "pca"): ["pca_results.png", "pca_loadings.png", "pca_reconstruction_error.png"],
    ("sklearn", "xgboost"): ["xgboost_confusion_matrix.png", "xgboost_feature_importance.png"],
    ("pytorch", "cnn"): ["cnn_feature_activations.png", "cnn_confusion_matrix.png"],
    ("pytorch", "gan"): ["gan_distribution_scatter.png", "gan_loss_curves.png"],
    ("pytorch", "lstm"): ["lstm_token_probabilities.png"],
    ("pytorch", "quantization"): ["quantization_comparison.png"],
    ("pytorch", "rag"): ["rag_similarity_scores.png", "rag_embedding_space.png"],
}

# Catalogue mapping incorporating dynamic stages and plots lists
_TASK_CATALOGUE = [
    {
        "module": mod,
        "task": task,
        "stages": _TASK_STAGES_MAP.get((mod, task), []),
        "plots": _TASK_PLOTS_MAP.get((mod, task), []),
    }
    for (mod, task) in sorted(TASK_RUNNER_MAP.keys())
]


@app.get("/tasks")
def list_tasks() -> list[dict]:
    """Return all available (module, task) pairs with stage names."""
    return _TASK_CATALOGUE


@app.get("/tasks/{module}/{task}/schema")
def get_task_schema(module: str, task: str) -> dict:
    """Return the Pydantic JSON schema config model for a specific task."""
    key = (module, task)
    config_cls = TASK_CONFIG_MAP.get(key)
    if not config_cls:
        raise HTTPException(status_code=404, detail=f"Task schema for '{module}/{task}' not found.")
    return config_cls.model_json_schema()


@app.post("/run/{module}/{task}", status_code=202)
async def run_task(module: str, task: str, background_tasks: BackgroundTasks, body: dict | None = None) -> dict:
    """Create a background job and return its *job_id* immediately."""
    key = (module, task)
    if key not in TASK_RUNNER_MAP:
        raise HTTPException(status_code=404, detail=f"Task '{module}/{task}' not found.")

    validated_config = {}
    if body is not None:
        config_cls = TASK_CONFIG_MAP.get(key)
        if config_cls:
            # Raises ValidationError -> 422 if invalid
            validated_config = config_cls(**body).model_dump()

    # Auto-cancel any previous active training jobs to prevent local CPU overload (single-user constraint)
    registry.cancel_active_jobs()

    job_id = str(uuid.uuid4())
    registry.create_job(job_id, module=module, task=task, config=validated_config)

    async def _run_job_async():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_executor, _run_in_thread, job_id, module, task, validated_config)

    background_tasks.add_task(_run_job_async)

    return {"job_id": job_id}


def _run_in_thread(job_id: str, module: str, task: str, config: dict) -> None:
    """Execute the ML task in the background worker thread pool with clean cancellation check."""
    try:
        runner = TASK_RUNNER_MAP[(module, task)]
        hook = HTTPProgressHook(job_id)
        registry.update_job(job_id, status="RUNNING")
        try:
            runner(hook, config)
            job = registry.get_job(job_id)
            if job and job.get("status") == "CANCELLED":
                return
            registry.update_job(job_id, status="COMPLETED", percentage=100.0)
        except Exception as exc:
            job = registry.get_job(job_id)
            if job and job.get("status") == "CANCELLED":
                return
            registry.update_job(
                job_id,
                status="FAILED",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
    except Exception:
        # Prevent thread crash propagation if runner resolution fails
        pass


@app.post("/cancel/{job_id}")
def cancel_job(job_id: str) -> dict:
    """Cancel a pending or running training job."""
    success = registry.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled (does not exist or already finished).")
    return {"status": "CANCELLED"}


_POLL_INTERVAL = 0.3  # seconds between registry polls


@app.get("/stream/{job_id}")
async def stream_job(job_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events endpoint that streams job progress.

    Accepts standard Last-Event-ID header for client reconnection resume.
    """
    job = registry.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    # Recover the last message index from the EventSource request header
    last_event_id = request.headers.get("last-event-id")
    start_idx = 0
    if last_event_id:
        try:
            start_idx = int(last_event_id)
        except ValueError:
            pass

    return StreamingResponse(
        _sse_generator(job_id, start_idx, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _sse_generator(job_id: str, start_idx: int, request: Request):
    """Async generator that yields SSE-formatted messages."""
    last_status = None
    last_stage = None
    last_percentage = None
    last_metrics_len = start_idx

    while True:
        # Auto-cancel job if client disconnected (tab closed / refreshed)
        if await request.is_disconnected():
            registry.cancel_job(job_id)
            return

        job = registry.get_job(job_id)
        if job is None:
            yield _sse(
                {
                    "status": "FAILED",
                    "stage": "",
                    "percentage": 0.0,
                    "new_metrics": [],
                    "error": "Job not found",
                }
            )
            return

        status = job["status"]
        stage = job["stage"]
        percentage = job["percentage"]
        new_metrics = registry.get_job_metrics_slice(job_id, last_metrics_len)

        # Emit event if first poll, or if status/stage/percentage/metrics changed
        changed = last_status is None or status != last_status or stage != last_stage or percentage != last_percentage or len(new_metrics) > 0

        if changed:
            last_metrics_len += len(new_metrics)
            payload: dict = {
                "status": status,
                "stage": stage,
                "percentage": percentage,
                "new_metrics": new_metrics,
            }
            if status == "FAILED":
                payload["error"] = job.get("error")

            yield _sse(payload, id=last_metrics_len)

            last_status = status
            last_stage = stage
            last_percentage = percentage

        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return

        await asyncio.sleep(_POLL_INTERVAL)


def _sse(data: dict, id: int | None = None) -> str:
    """Format a dict as a standard SSE ``data:`` block with optional event id."""
    out = f"data: {json.dumps(data)}\n"
    if id is not None:
        out += f"id: {id}\n"
    out += "\n"
    return out


@app.get("/jobs")
def list_jobs(module: str | None = None, task: str | None = None) -> list[dict]:
    """List recent jobs in the registry, optionally filtered by module and task."""
    return registry.list_jobs(module=module, task=task)


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict:
    """Return the current state of a job (polling alternative to SSE)."""
    job = registry.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


# Dynamically generate allowed plot filenames for safety from task plots metadata
ALLOWED_PLOTS = {plot for plots in _TASK_PLOTS_MAP.values() for plot in plots}


@app.get("/plots/{job_id}/{filename}")
def get_plot(job_id: str, filename: str):
    """Retrieve a generated plot image stored in memory for a specific job."""
    safe_filename = os.path.basename(filename)
    if safe_filename not in ALLOWED_PLOTS:
        raise HTTPException(status_code=400, detail="Requested file is not a valid plot name")

    image_bytes = registry.get_job_plot(job_id, safe_filename)
    if not image_bytes:
        raise HTTPException(status_code=404, detail=f"Plot file '{safe_filename}' not found for job '{job_id}'.")

    return Response(content=image_bytes, media_type="image/png")
