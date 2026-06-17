import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import registry
from backend.hooks import HTTPProgressHook
from backend.main import app
from backend.tasks import TASK_RUNNER_MAP

client = TestClient(app)


class TestBackendAPI(unittest.TestCase):
    def setUp(self):
        # Clear jobs registry before each test
        registry._jobs.clear()

    def test_list_tasks(self):
        response = client.get("/tasks")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(len(data) > 0)
        self.assertIn("stages", data[0])
        self.assertIn("module", data[0])
        self.assertIn("task", data[0])

    def test_get_task_schema(self):
        response = client.get("/tasks/numpy/backpropagation/schema")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertEqual(schema["title"], "BackpropagationConfig")
        self.assertIn("properties", schema)
        self.assertIn("num_epochs", schema["properties"])

    def test_get_task_schema_not_found(self):
        response = client.get("/tasks/invalid_module/invalid_task/schema")
        self.assertEqual(response.status_code, 404)

    def test_run_task_route(self):
        def dummy_runner(hook, config):
            pass

        with patch.dict(TASK_RUNNER_MAP, {("numpy", "backpropagation"): dummy_runner}):
            response = client.post("/run/numpy/backpropagation", json={"num_epochs": 10})
            self.assertEqual(response.status_code, 202)
            job_id = response.json()["job_id"]
            self.assertIsNotNone(job_id)

            job = registry.get_job(job_id)
            self.assertIsNotNone(job)
            self.assertIn(job["status"], ("PENDING", "RUNNING", "COMPLETED"))

    def test_sse_event_id_resumption(self):
        job_id = "test-sse-resume-job"
        registry.create_job(job_id)
        # Set to RUNNING first so metrics can be appended
        registry.update_job(job_id, status="RUNNING", stage="Training", percentage=100.0)
        registry.append_metrics(job_id, {"epoch": 1, "loss": 0.8})
        registry.append_metrics(job_id, {"epoch": 2, "loss": 0.6})
        registry.append_metrics(job_id, {"epoch": 3, "loss": 0.4})
        registry.update_job(job_id, status="COMPLETED")

        # Test resumption starting from metric index 2 (Last-Event-ID: 2)
        headers = {"Last-Event-ID": "2"}
        response = client.get(f"/stream/{job_id}", headers=headers)
        self.assertEqual(response.status_code, 200)

        # Verify the SSE response body
        body = response.text
        self.assertIn("Training", body)
        self.assertIn("new_metrics", body)
        self.assertNotIn('"epoch": 1', body)
        self.assertNotIn('"epoch": 2', body)
        self.assertIn('"epoch": 3', body)

    def test_cooperative_cancellation(self):
        job_id = "test-cancel-id"
        registry.create_job(job_id)
        registry.update_job(job_id, status="RUNNING")

        hook = HTTPProgressHook(job_id)
        self.assertFalse(hook.is_cancelled())

        # Trigger cancellation in registry
        success = registry.cancel_job(job_id)
        self.assertTrue(success)
        self.assertTrue(hook.is_cancelled())

    def test_terminal_status_immutability(self):
        job_id = "test-terminal-immutability-id"
        registry.create_job(job_id)
        registry.update_job(job_id, status="RUNNING")

        hook = HTTPProgressHook(job_id)

        # Cancel job
        registry.cancel_job(job_id)
        self.assertEqual(registry.get_job(job_id)["status"], "CANCELLED")

        # Attempting hook updates should be ignored and not change status
        hook.update_stage("Epoch 2", 50.0)
        self.assertEqual(registry.get_job(job_id)["status"], "CANCELLED")
        self.assertEqual(registry.get_job(job_id)["stage"], "")

        # Directly updating the job via registry when terminal should also be blocked
        registry.update_job(job_id, status="RUNNING", stage="Attempted Bypass")
        self.assertEqual(registry.get_job(job_id)["status"], "CANCELLED")
        self.assertEqual(registry.get_job(job_id)["stage"], "")

    def test_cancel_route(self):
        job_id = "test-cancel-route-id"
        registry.create_job(job_id)
        registry.update_job(job_id, status="RUNNING")

        # Call cancel endpoint
        response = client.post(f"/cancel/{job_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "CANCELLED")

        self.assertEqual(registry.get_job(job_id)["status"], "CANCELLED")

    def test_ttl_eviction(self):
        job_id = "old-job-id"
        registry.create_job(job_id)

        # Alter creation timestamp to be older than 1 hour (7200 seconds ago)
        registry.update_job(job_id, created_at=time.time() - 7200)

        # Run TTL cleanup
        registry.clean_old_jobs(max_age_seconds=3600)

        # Verify job is removed
        self.assertIsNone(registry.get_job(job_id))

    def test_global_auto_cancellation_on_run(self):
        # Create an existing job and set it to running
        old_job_id = "running-job-id"
        registry.create_job(old_job_id)
        registry.update_job(old_job_id, status="RUNNING")

        def dummy_runner(hook, config):
            pass

        with patch.dict(TASK_RUNNER_MAP, {("numpy", "backpropagation"): dummy_runner}):
            # Start a new job which should trigger auto-cancellation of the running one
            response = client.post("/run/numpy/backpropagation", json={"num_epochs": 10})
            self.assertEqual(response.status_code, 202)

            # Verify the old job is cancelled
            self.assertEqual(registry.get_job(old_job_id)["status"], "CANCELLED")

    def test_get_plot_not_found(self):
        # Request a valid plot name but for a job that doesn't have it
        response = client.get("/plots/nonexistent-job/kmeans_clustering_results.png")
        self.assertEqual(response.status_code, 404)

    def test_get_plot_invalid_filename(self):
        # Request a disallowed file name
        response = client.get("/plots/some-job/unauthorized_file.txt")
        self.assertEqual(response.status_code, 400)

        # Test directory traversal prevention
        response = client.get("/plots/some-job/../../etc/passwd")
        self.assertIn(response.status_code, (400, 404))

    def test_get_plot_success(self):
        # Create a job and manually inject a mock plot
        job_id = "test-plot-job"
        registry.create_job(job_id)
        mock_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR..."
        registry.save_job_plot(job_id, "kmeans_clustering_results.png", mock_png)

        response = client.get(f"/plots/{job_id}/kmeans_clustering_results.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, mock_png)
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_new_task_schemas_and_listing(self):
        # Verify the list tasks includes our new modules
        list_response = client.get("/tasks")
        self.assertEqual(list_response.status_code, 200)
        tasks = list_response.json()

        numpy_tasks = [t["task"] for t in tasks if t["module"] == "numpy"]
        pytorch_tasks = [t["task"] for t in tasks if t["module"] == "pytorch"]

        self.assertIn("q_learning", numpy_tasks)
        self.assertIn("attention", numpy_tasks)
        self.assertIn("cnn", pytorch_tasks)
        self.assertIn("gan", pytorch_tasks)
        self.assertIn("lstm", pytorch_tasks)
        self.assertIn("quantization", pytorch_tasks)

        # Verify new schemas load correctly
        for module, task in [
            ("numpy", "q_learning"),
            ("numpy", "attention"),
            ("pytorch", "cnn"),
            ("pytorch", "gan"),
            ("pytorch", "lstm"),
            ("pytorch", "quantization"),
        ]:
            schema_resp = client.get(f"/tasks/{module}/{task}/schema")
            self.assertEqual(schema_resp.status_code, 200, f"Failed schema fetch for {module}/{task}")
            schema = schema_resp.json()
            self.assertIn("properties", schema)


if __name__ == "__main__":
    unittest.main()
