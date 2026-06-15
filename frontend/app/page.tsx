"use client";

import { useEffect, useRef, useState } from "react";
import {
  fetchTasks,
  launchJob,
  cancelJob,
  openEventSource,
  JobState,
  SsePayload,
  Task,
} from "./api";
import { TASK_LABELS } from "./constants";
import Sidebar from "./components/Sidebar";
import ConfigForm from "./components/ConfigForm";
import ProgressPanel from "./components/ProgressPanel";

export default function Home() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [jobState, setJobState] = useState<JobState | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  // Load task catalogue on mount
  useEffect(() => {
    fetchTasks()
      .then(setTasks)
      .catch(() => setError("Could not reach the ML Workshop API. Is the backend running?"));
  }, []);

  // Clean up EventSource on unmount
  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  function handleTaskSelect(task: Task) {
    esRef.current?.close();
    setSelectedTask(task);
    setJobState(null);
    setCurrentJobId(null);
    setIsRunning(false);
    setError(null);
  }

  async function handleRun(config: Record<string, number>) {
    if (!selectedTask) return;
    setError(null);
    setIsRunning(true);
    setCurrentJobId(null);
    setJobState({ status: "PENDING", stage: "", percentage: 0, metrics: [] });

    let jobId: string;
    try {
      jobId = await launchJob(selectedTask.module, selectedTask.task, config);
      setCurrentJobId(jobId);
    } catch (err) {
      setError(String(err));
      setIsRunning(false);
      return;
    }

    // Open SSE stream
    esRef.current?.close();
    const es = openEventSource(jobId);
    esRef.current = es;

    es.onmessage = (event: MessageEvent) => {
      const payload: SsePayload = JSON.parse(event.data);

      setJobState((prev) => {
        const prevMetrics = prev?.metrics ?? [];
        return {
          status: payload.status,
          stage: payload.stage,
          percentage: payload.percentage,
          metrics: [...prevMetrics, ...payload.new_metrics],
          error: payload.error ?? null,
        };
      });

      if (
        payload.status === "COMPLETED" ||
        payload.status === "FAILED" ||
        payload.status === "CANCELLED"
      ) {
        es.close();
        setIsRunning(false);
      }
    };

    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) {
        es.close();
        setIsRunning(false);
      }
    };
  }

  async function handleCancel() {
    if (!currentJobId) return;
    try {
      await cancelJob(currentJobId);
      setJobState((prev) => (prev ? { ...prev, status: "CANCELLED" } : null));
      esRef.current?.close();
      setIsRunning(false);
    } catch (err) {
      setError(`Failed to cancel job: ${err}`);
    }
  }

  const stages = selectedTask?.stages ?? [];

  return (
    <div className="flex min-h-screen bg-[#0a0a0a] text-gray-100">
      {/* Ambient Background Gradients */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] bg-indigo-950/20 rounded-full blur-[120px]" />
        <div className="absolute top-[40%] -right-[10%] w-[40%] h-[40%] bg-purple-950/20 rounded-full blur-[100px]" />
        <div className="absolute -bottom-[10%] left-[20%] w-[30%] h-[30%] bg-blue-950/20 rounded-full blur-[80px]" />
      </div>

      <Sidebar tasks={tasks} selected={selectedTask} onSelect={handleTaskSelect} />

      <main className="flex-1 p-8 overflow-y-auto relative z-10">
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-700 rounded-lg text-red-300 text-sm backdrop-blur-sm">
            {error}
          </div>
        )}

        {!selectedTask ? (
          /* Premium Landing Page Hero */
          <div className="min-h-[calc(100vh-8rem)] flex flex-col items-center justify-center space-y-12 animate-fade-in-up">
            {/* Header */}
            <div className="text-center space-y-6 max-w-4xl mx-auto px-4">
              <h1 className="text-5xl md:text-7xl font-bold tracking-tight bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent pb-2">
                Master Machine Learning
                <br />
                <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Through Hands-On Practice
                </span>
              </h1>

              <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed">
                Configure hyperparameters, watch training in real-time, and deepen your
                understanding of NumPy, scikit-learn, PyTorch, and more through interactive
                workshops.
              </p>

              <div className="flex flex-wrap justify-center gap-4 pt-4">
                <div className="px-6 py-3 bg-white/5 border border-white/10 rounded-full text-sm font-medium backdrop-blur-md hover:bg-white/10 transition-colors cursor-default">
                  📊 Real-time Metrics Visualization
                </div>
                <div className="px-6 py-3 bg-white/5 border border-white/10 rounded-full text-sm font-medium backdrop-blur-md hover:bg-white/10 transition-colors cursor-default">
                  🚀 Interactive Hyperparameter Tuning
                </div>
                <div className="px-6 py-3 bg-white/5 border border-white/10 rounded-full text-sm font-medium backdrop-blur-md hover:bg-white/10 transition-colors cursor-default">
                  🎯 Multiple ML Frameworks
                </div>
              </div>
            </div>

            {/* Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto px-4 w-full">
              {[
                {
                  title: "NumPy Fundamentals",
                  description:
                    "Build neural networks from scratch using only NumPy. Understand backpropagation, gradient descent, and matrix operations at a deep level.",
                  icon: (
                    <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
                      />
                    </svg>
                  ),
                  gradient: "from-blue-500 to-cyan-500",
                },
                {
                  title: "Classical ML",
                  description:
                    "Master scikit-learn with algorithms like SVM, Random Forest, XGBoost, and K-Means. Learn feature engineering, model evaluation, and hyperparameter tuning.",
                  icon: (
                    <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                      />
                    </svg>
                  ),
                  gradient: "from-green-500 to-emerald-500",
                },
                {
                  title: "Deep Learning with PyTorch",
                  description:
                    "Build and train neural networks for image classification, text processing, time series forecasting, and question answering.",
                  icon: (
                    <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M13 10V3L4 14h7v7l9-11h-7z"
                      />
                    </svg>
                  ),
                  gradient: "from-purple-500 to-pink-500",
                },
              ].map((card, idx) => (
                <div
                  key={idx}
                  className="group glass-panel rounded-xl p-6 hover:scale-[1.02] transition-transform duration-300 transform-gpu will-change-transform cursor-default"
                >
                  <div
                    className={`w-12 h-12 rounded-lg bg-gradient-to-br ${card.gradient} flex items-center justify-center mb-4 shadow-lg`}
                  >
                    {card.icon}
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2 group-hover:text-indigo-300 transition-colors">
                    {card.title}
                  </h3>
                  <p className="text-gray-400 text-sm leading-relaxed">{card.description}</p>
                </div>
              ))}
            </div>

            {/* Shimmering Accent Ring */}
            <div className="relative w-64 h-64 md:w-80 md:h-80 flex items-center justify-center">
              <div className="absolute inset-0 border-2 border-indigo-500/30 rounded-full animate-[spin_10s_linear_infinite] transform-gpu will-change-transform" />
              <div className="absolute inset-4 border border-purple-500/20 rounded-full animate-[spin_15s_linear_infinite_reverse] transform-gpu will-change-transform" />
              <div className="absolute inset-8 w-48 h-48 md:w-64 md:h-64 bg-gradient-to-b from-indigo-900/30 to-transparent rounded-full blur-xl" />

              <div className="text-center z-10">
                <p className="text-6xl mb-2">🚀</p>
                <p className="text-gray-300 font-medium">Ready to start?</p>
                <p className="text-sm text-gray-500 mt-1">Select a task from the sidebar</p>
              </div>
            </div>
          </div>
        ) : (
          /* Task Detail View - Split-Screen Layout */
          <div className="split:grid split:grid-cols-12 split:gap-8 split:items-start animate-fade-in">
            {/* Left Column: Configuration (col-span-5) */}
            <div className="split:col-span-5 space-y-6">
              <div>
                <p className="text-xs uppercase tracking-widest text-indigo-400 mb-1 font-semibold">
                  {selectedTask.module}
                </p>
                <h2 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent pb-2">
                  {TASK_LABELS[selectedTask.task] ?? selectedTask.task}
                </h2>
              </div>

              <section>
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"
                    />
                  </svg>
                  Configuration
                </h3>
                <div className="glass-panel rounded-xl p-6">
                  <ConfigForm
                    key={`${selectedTask.module}/${selectedTask.task}`}
                    task={selectedTask}
                    disabled={isRunning}
                    onSubmit={handleRun}
                  />
                </div>
              </section>
            </div>

            {/* Right Column: Progress Panel (col-span-7) */}
            <div className="split:col-span-7 space-y-6">
              <section>
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                    />
                  </svg>
                  Progress & Metrics
                </h3>
                <div className="glass-panel rounded-xl p-6">
                  <ProgressPanel jobState={jobState} stages={stages} onCancel={handleCancel} />
                </div>
              </section>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
