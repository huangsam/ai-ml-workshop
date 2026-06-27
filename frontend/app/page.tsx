"use client";

import { useEffect, useRef, useState } from "react";
import {
  fetchTasks,
  launchJob,
  cancelJob,
  openEventSource,
  pingServer,
  JobState,
  SsePayload,
  Task,
} from "./api";
import { Shuffle, Info, Sliders, TrendingUp, FlaskConical, BarChart3, Zap } from "lucide-react";
import { TASK_LABELS } from "./constants";
import Sidebar from "./components/Sidebar";
import ConfigForm from "./components/ConfigForm";
import ProgressPanel from "./components/ProgressPanel";
import { THEORY_DATA } from "./theory";
import TheoryModal from "./components/TheoryModal";

export default function Home() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [jobState, setJobState] = useState<JobState | null>(null);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isTheoryOpen, setIsTheoryOpen] = useState(false);
  const [loadedConfig, setLoadedConfig] = useState<Record<string, number> | null>(null);
  const [apiConnected, setApiConnected] = useState<"connected" | "disconnected" | "checking">(
    "checking"
  );
  const esRef = useRef<EventSource | null>(null);
  const mainRef = useRef<HTMLElement | null>(null);

  // Scroll back to top when task selection changes (e.g. from Surprise Me or Sidebar)
  useEffect(() => {
    window.scrollTo(0, 0);
    if (mainRef.current) {
      mainRef.current.scrollTop = 0;
    }
  }, [selectedTask]);

  // Background health check polling loop
  useEffect(() => {
    let active = true;

    async function checkHealth() {
      const isAlive = await pingServer();
      if (!active) return;

      if (isAlive) {
        setApiConnected("connected");
        setError((prev) => {
          if (prev?.includes("Could not reach the ML Workshop API")) {
            return null;
          }
          return prev;
        });
      } else {
        setApiConnected("disconnected");
        setIsLoading(false);
        if (tasks.length === 0) {
          setError("Could not reach the ML Workshop API. Is the backend running?");
        }
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 5000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [tasks.length]);

  // Load task catalogue when API becomes connected
  useEffect(() => {
    if (apiConnected !== "connected" || tasks.length > 0) return;

    // eslint-disable-next-line react-hooks/set-state-in-effect
    setIsLoading(true);
    fetchTasks()
      .then((data) => {
        setTasks(data);
        // Resolve hash immediately on mount
        const hash = window.location.hash.slice(1);
        if (hash) {
          const [module, taskName] = hash.split("/");
          if (module && taskName) {
            const found = data.find((t) => t.module === module && t.task === taskName);
            if (found) {
              setSelectedTask(found);
            }
          }
        }
        setIsLoading(false);
        setError(null);
      })
      .catch(() => {
        setError("Could not reach the ML Workshop API. Is the backend running?");
        setIsLoading(false);
      });
  }, [apiConnected, tasks.length]);

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
    setLoadedConfig(null);

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
    setJobState({ status: "PENDING", stage: "", percentage: 0, metrics: [], logs: "" });

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
        const prevLogs = prev?.logs ?? "";
        return {
          status: payload.status,
          stage: payload.stage,
          percentage: payload.percentage,
          metrics: [...prevMetrics, ...payload.new_metrics],
          logs: prevLogs + (payload.new_logs ?? ""),
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

      <Sidebar
        tasks={tasks}
        selected={selectedTask}
        onSelect={handleTaskSelect}
        apiConnected={apiConnected}
      />

      <main ref={mainRef} className="flex-1 p-8 overflow-y-auto relative z-10">
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-700 rounded-lg text-red-300 text-sm backdrop-blur-sm">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="min-h-[calc(100vh-8rem)] flex flex-col items-center justify-center">
            {/* Premium Loading Spinner */}
            <div className="relative w-12 h-12 animate-pulse">
              <div className="absolute inset-0 border-2 border-indigo-500/20 rounded-full" />
              <div className="absolute inset-0 border-t-2 border-indigo-500 rounded-full animate-spin shadow-[0_0_15px_rgba(99,102,241,0.4)]" />
            </div>
          </div>
        ) : !selectedTask ? (
          /* Premium Landing Page Hero */
          <div className="min-h-[calc(100vh-8rem)] flex flex-col items-center justify-center space-y-12 animate-fade-in-up">
            {/* Header */}
            <div className="text-center space-y-6 max-w-5xl mx-auto px-4 pt-6 md:pt-10">
              <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent pb-4 text-balance leading-tight">
                Master Machine Learning{" "}
                <span className="bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  Through Hands-On Practice
                </span>
              </h1>

              <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto leading-relaxed text-balance">
                Tweak hyperparameters, stream live training telemetry, and build core model
                intuition—all in real time.
              </p>
            </div>

            {/* Feature Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto px-4 w-full">
              {[
                {
                  title: "NumPy from Scratch",
                  module: "numpy",
                  description:
                    "Implement neural networks and backpropagation using raw matrix math—no library abstractions.",
                  icon: FlaskConical,
                  gradient: "from-blue-500 to-cyan-500",
                },
                {
                  title: "Classical Algorithms",
                  module: "sklearn",
                  description:
                    "Train and evaluate SVMs, decision trees, XGBoost, and K-Means models with scikit-learn.",
                  icon: BarChart3,
                  gradient: "from-green-500 to-emerald-500",
                },
                {
                  title: "Deep Learning",
                  module: "pytorch",
                  description:
                    "Train advanced PyTorch models—CNNs, text generation, LSTMs, and LoRA fine-tuning.",
                  icon: Zap,
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
                        <card.icon className="w-8 h-8 text-white" />
                      </div>
                      <h3 className="text-xl font-bold text-white mb-2 group-hover:text-indigo-300 transition-colors">
                        {card.title}
                      </h3>
                      <p className="text-gray-400 text-sm leading-relaxed mb-4 text-balance">
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

            {/* Glowing Accent Ring */}
            <div className="relative w-64 h-64 md:w-80 md:h-80 flex items-center justify-center mt-6 mb-12">
              <div className="absolute inset-0 border border-indigo-500/20 rounded-full shadow-[0_0_35px_rgba(99,102,241,0.25)] animate-[pulse_4s_ease-in-out_infinite] transform-gpu will-change-transform" />
              <div className="absolute inset-8 w-48 h-48 md:w-64 md:h-64 bg-gradient-to-b from-indigo-950/30 to-transparent rounded-full blur-xl" />

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
                  className="mt-3 px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-xs font-bold text-white rounded-full transition-all duration-300 shadow-md hover:shadow-indigo-500/20 active:scale-[0.97] cursor-pointer flex items-center justify-center gap-1.5"
                >
                  <Shuffle className="w-3.5 h-3.5 animate-pulse" />
                  <span>Surprise Me!</span>
                </button>
              </div>
            </div>
          </div>
        ) : (
          /* Task Detail View - Split-Screen Layout */
          <div className="grid grid-cols-1 gap-8 split:grid-cols-12 split:items-start animate-fade-in">
            {/* Left Column: Configuration (col-span-5) */}
            <div className="split:col-span-5 space-y-6">
              <div>
                <p className="text-xs uppercase tracking-widest text-indigo-400 mb-1 font-semibold">
                  {selectedTask.module}
                </p>
                <div
                  onClick={() => setIsTheoryOpen(true)}
                  className="inline-flex items-center gap-3 group cursor-pointer select-none"
                  title="Click to view theory and concept explanation"
                >
                  <h2 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-white via-gray-100 to-gray-300 bg-clip-text text-transparent group-hover:from-white group-hover:to-indigo-300 transition-all duration-300 pb-1">
                    {TASK_LABELS[selectedTask.task] ?? selectedTask.task}
                  </h2>
                  <div className="w-6 h-6 rounded-full bg-white/5 group-hover:bg-indigo-500/15 border border-white/5 group-hover:border-indigo-500/30 flex items-center justify-center text-gray-400 group-hover:text-indigo-400 transition-all duration-300 shadow-sm">
                    <Info className="w-3.5 h-3.5" />
                  </div>
                </div>
              </div>

              <section>
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-indigo-400" />
                  <span>Configuration</span>
                </h3>
                <div className="glass-panel rounded-xl p-6">
                  <ConfigForm
                    key={`${selectedTask.module}/${selectedTask.task}`}
                    task={selectedTask}
                    disabled={isRunning}
                    onSubmit={handleRun}
                    initialValues={loadedConfig}
                    apiConnected={apiConnected}
                  />
                </div>
              </section>
            </div>

            {/* Right Column: Progress Panel (col-span-7) */}
            <div className="split:col-span-7 space-y-6">
              <section>
                <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-indigo-400" />
                  <span>Progress & Metrics</span>
                </h3>
                <div className="glass-panel rounded-xl p-6">
                  <ProgressPanel
                    key={`${selectedTask.module}/${selectedTask.task}`}
                    jobState={jobState}
                    stages={stages}
                    onCancel={handleCancel}
                    jobId={currentJobId}
                    plots={selectedTask.plots}
                    onLoadConfig={setLoadedConfig}
                    selectedTask={selectedTask}
                    apiConnected={apiConnected}
                  />
                </div>
              </section>
            </div>
          </div>
        )}
      </main>

      {selectedTask && (
        <TheoryModal
          isOpen={isTheoryOpen}
          onClose={() => setIsTheoryOpen(false)}
          content={THEORY_DATA[`${selectedTask.module}/${selectedTask.task}`]}
        />
      )}
    </div>
  );
}
