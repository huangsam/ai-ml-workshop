/**
 * Centralised API client – wraps fetch calls to the FastAPI backend.
 */

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Task {
  module: string;
  task: string;
  stages: string[];
}

export interface JobState {
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";
  stage: string;
  percentage: number;
  metrics: Record<string, number>[];
  error?: string | null;
}

export interface SsePayload {
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";
  stage: string;
  percentage: number;
  new_metrics: Record<string, number>[];
  error?: string | null;
}

export async function fetchTasks(): Promise<Task[]> {
  const res = await fetch(`${API_BASE}/tasks`);
  if (!res.ok) throw new Error("Failed to fetch task list");
  return res.json();
}

export async function fetchTaskSchema(
  module: string,
  task: string
): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/tasks/${module}/${task}/schema`);
  if (!res.ok) throw new Error(`Failed to fetch config schema for ${module}/${task}`);
  return res.json();
}

export async function launchJob(
  module: string,
  task: string,
  config: Record<string, unknown>
): Promise<string> {
  const res = await fetch(`${API_BASE}/run/${module}/${task}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.detail ?? `HTTP ${res.status}`);
  }
  const data: { job_id: string } = await res.json();
  return data.job_id;
}

export async function cancelJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/cancel/${jobId}`, {
    method: "POST",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.detail ?? `HTTP ${res.status}`);
  }
}

export function openEventSource(jobId: string): EventSource {
  return new EventSource(`${API_BASE}/stream/${jobId}`);
}
