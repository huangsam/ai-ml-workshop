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

  // Listen to hash changes for browser back/forward navigation and deep-linking
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1);
      if (!hash) {
        if (selectedTask !== null) {
          handleTaskSelect(null);
        }
        return;
      }
      const [module, taskName] = hash.split("/");
      if (module && taskName) {
        if (selectedTask?.module === module && selectedTask?.task === taskName) {
          return;
        }
        const found = tasks.find((t) => t.module === module && t.task === taskName);
        if (found) {
          esRef.current?.close();
          setSelectedTask(found);
          setJobState(null);
          setCurrentJobId(null);
          setIsRunning(false);
          setError(null);
        } else {
          handleTaskSelect(null);
        }
      } else {
        handleTaskSelect(null);
      }
    };

    if (tasks.length > 0) {
      handleHashChange();
    }

    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, [tasks, selectedTask]);

  function handleTaskSelect(task: Task | null) {
    esRef.current?.close();
    setSelectedTask(task);
    setJobState(null);
    setCurrentJobId(null);
    setIsRunning(false);
    setError(null);

    const targetHash = task ? `#${task.module}/${task.task}` : "";
    if (window.location.hash !== targetHash) {
      if (targetHash) {
        window.location.hash = targetHash;
      } else {
        window.history.pushState(null, "", window.location.pathname + window.location.search);
      }
    }
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
            </div>

            {/* Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto px-4 w-full">
              {[
                {
                  title: "NumPy Fundamentals",
                  module: "numpy",
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
                  module: "sklearn",
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
                  module: "pytorch",
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
              ].map((card, idx) => {
                const targetTask = tasks.find((t) => t.module === card.module);
                return (
                  <div
                    key={idx}
                    onClick={() => targetTask && handleTaskSelect(targetTask)}
                    className="group glass-panel rounded-xl p-6 hover:scale-[1.02] active:scale-[0.99] transition-all duration-300 transform-gpu will-change-transform cursor-pointer border border-white/5 hover:border-indigo-500/30 shadow-lg hover:shadow-indigo-500/5 flex flex-col justify-between"
                  >
                    <div>
                      <div
                        className={`w-12 h-12 rounded-lg bg-gradient-to-br ${card.gradient} flex items-center justify-center mb-4 shadow-lg`}
                      >
                        {card.icon}
                      </div>
                      <h3 className="text-xl font-bold text-white mb-2 group-hover:text-indigo-300 transition-colors">
                        {card.title}
                      </h3>
                      <p className="text-gray-400 text-sm leading-relaxed mb-4">
                        {card.description}
                      </p>
                    </div>
                    {targetTask && (
                      <div className="flex items-center gap-1.5 text-xs font-semibold text-indigo-400 group-hover:text-indigo-300 transition-colors mt-auto">
                        <span>Explore Tasks</span>
                        <span className="transform group-hover:translate-x-1 transition-transform">
                          →
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Shimmering Accent Ring */}
            <div className="relative w-64 h-64 md:w-80 md:h-80 flex items-center justify-center">
              <div className="absolute inset-0 border-2 border-indigo-500/30 rounded-full animate-[spin_10s_linear_infinite] transform-gpu will-change-transform" />
              <div className="absolute inset-4 border border-purple-500/20 rounded-full animate-[spin_15s_linear_infinite_reverse] transform-gpu will-change-transform" />
              <div className="absolute inset-8 w-48 h-48 md:w-64 md:h-64 bg-gradient-to-b from-indigo-900/30 to-transparent rounded-full blur-xl" />

              <div className="text-center z-10 flex flex-col items-center">
                <svg
                  className="w-16 h-16 mb-4 animate-float text-indigo-400 filter drop-shadow-[0_0_15px_rgba(99,102,241,0.5)]"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <defs>
                    <linearGradient id="rocketGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#818CF8" />
                      <stop offset="50%" stopColor="#A78BFA" />
                      <stop offset="100%" stopColor="#F472B6" />
                    </linearGradient>
                    <linearGradient id="fireGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#F59E0B" />
                      <stop offset="100%" stopColor="#EF4444" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  {/* Flame */}
                  <path
                    d="M12 17.5C11.5 19 10 21.5 10 21.5C10 21.5 12 20.5 12 19.5C12 20.5 14 21.5 14 21.5C14 21.5 12.5 19 12 17.5Z"
                    fill="url(#fireGrad)"
                  />
                  <path
                    d="M12 18C11.8 19 11 20.5 11 20.5C11 20.5 12 20 12 19.3C12 20 13 20.5 13 20.5C13 20.5 12.2 19 12 18Z"
                    fill="#FCD34D"
                  />
                  {/* Rocket wings */}
                  <path
                    d="M7 14.5L4.5 17C4.2 17.3 4 17.8 4 18.2V20L7.5 18L7 14.5Z"
                    fill="#4F46E5"
                    opacity="0.8"
                  />
                  <path
                    d="M17 14.5L19.5 17C19.8 17.3 20 17.8 20 18.2V20L16.5 18L17 14.5Z"
                    fill="#7C3AED"
                    opacity="0.8"
                  />
                  {/* Rocket main body */}
                  <path
                    d="M12 2C12 2 7 6 7 13C7 16 9.5 18 12 18C14.5 18 17 16 17 13C17 6 12 2 12 2Z"
                    fill="url(#rocketGrad)"
                  />
                  {/* Window */}
                  <circle cx="12" cy="10" r="2.5" fill="#0f172a" stroke="#ffffff" strokeWidth="1" />
                  <circle cx="11.5" cy="9.5" r="0.8" fill="#ffffff" />
                  {/* Fine lines/details */}
                  <path d="M12 2V5" stroke="#ffffff" strokeWidth="0.8" opacity="0.6" />
                </svg>
                <p className="text-gray-200 font-semibold text-lg">Ready to start?</p>
                <p className="text-xs text-gray-500 mt-1">Select a task from the sidebar or</p>
                <button
                  onClick={() => {
                    if (tasks.length > 0) {
                      const randomIdx = Math.floor(Math.random() * tasks.length);
                      handleTaskSelect(tasks[randomIdx]);
                    }
                  }}
                  className="mt-3 px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-xs font-bold text-white rounded-full transition-all duration-300 shadow-md hover:shadow-indigo-500/20 active:scale-[0.97] cursor-pointer"
                >
                  🎲 Surprise Me!
                </button>
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
