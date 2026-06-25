"use client";

import { useEffect, useState } from "react";
import {
  Square,
  TrendingUp,
  Sparkles,
  History,
  Scale,
  Activity,
  BarChart3,
  Maximize2,
  RefreshCw,
  AlertTriangle,
} from "lucide-react";
import { JobState, API_BASE, JobInfo, fetchJobs, Task } from "../api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface ProgressPanelProps {
  jobState: JobState | null;
  stages: string[];
  onCancel?: () => void;
  jobId?: string | null;
  plots: string[];
  selectedTask?: Task | null;
  onLoadConfig?: (config: Record<string, number>) => void;
  apiConnected?: "connected" | "disconnected" | "checking";
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{
    name?: string | number | symbol;
    value?: unknown;
    color?: string;
    stroke?: string;
  }>;
  label?: unknown;
}

const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-tooltip rounded-lg px-3 py-2 text-xs space-y-1">
        <p className="text-gray-400 font-medium mb-1 border-b border-white/5 pb-1">
          {label !== undefined ? `Step/Epoch: ${String(label)}` : "Metrics"}
        </p>
        {payload.map((entry, index) => (
          <div key={entry.name?.toString() ?? index} className="flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: entry.stroke || entry.color }}
            />
            <span className="text-gray-300 font-medium text-left truncate max-w-[180px]">
              {entry.name?.toString()}:
            </span>
            <span className="text-white font-semibold ml-auto">
              {typeof entry.value === "number" ? entry.value.toFixed(4) : String(entry.value ?? "")}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function ProgressPanel({
  jobState,
  stages,
  onCancel,
  jobId,
  plots,
  selectedTask,
  onLoadConfig,
  apiConnected = "connected",
}: ProgressPanelProps) {
  const availablePlots = plots ?? [];

  const [activeTab, setActiveTab] = useState<"metrics" | "visualizations" | "compare">("metrics");
  const [selectedPlot, setSelectedPlot] = useState<string>(availablePlots[0] ?? "");
  const [isZoomed, setIsZoomed] = useState<boolean>(false);
  const [zoomedJobId, setZoomedJobId] = useState<string | null>(null);

  // History and comparison states
  const [recentJobs, setRecentJobs] = useState<JobInfo[]>([]);
  const [selectedJobIds, setSelectedJobIds] = useState<string[]>([]);
  const [compareMetric, setCompareMetric] = useState<string>("");

  const status = jobState?.status;
  const stage = jobState?.stage;
  const percentage = jobState?.percentage ?? 0.0;
  const metrics = jobState?.metrics ?? [];
  const error = jobState?.error;

  // Fetch job history for this task
  useEffect(() => {
    if (!selectedTask || apiConnected !== "connected") return;
    fetchJobs(selectedTask.module, selectedTask.task)
      .then((data) => {
        setRecentJobs(data);
      })
      .catch((err) => {
        console.error("Failed to load run history", err);
      });
  }, [selectedTask, status, apiConnected]);

  // Determine which metrics keys exist across all snapshots for active run
  const metricKeys = Array.from(
    new Set(metrics.flatMap((m) => Object.keys(m).filter((k) => k !== "percentage")))
  );

  // Use the first metric key as the X-axis (e.g. "epoch", "step"), or fall back to index.
  const hasEpoch = metricKeys.includes("epoch");
  const xAxisKey = hasEpoch
    ? "epoch"
    : (metricKeys.find((k) => k !== "loss" && k !== "accuracy") ?? "step");
  const lineKeys = metricKeys.filter((k) => k !== xAxisKey);

  // Map metrics to ensure every point has an X-axis value
  const chartData = metrics.map((m, idx) => {
    const pt = { ...m };
    if (pt[xAxisKey] === undefined) {
      pt[xAxisKey] = idx + 1;
    }
    return pt;
  });

  // Build stage list dynamically
  const allStages = stages.length > 0 ? [...stages] : [];
  if (stage && !allStages.includes(stage)) {
    allStages.push(stage);
  }
  const currentStageIdx = stage ? allStages.indexOf(stage) : -1;

  return (
    <div className="space-y-6">
      {/* Active Run Status/Progress Card */}
      {jobState ? (
        <div className="space-y-4 pb-4 border-b border-white/5">
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-3">
              <span
                className={`font-semibold px-2.5 py-1 rounded-full text-xs ${
                  status === "PENDING"
                    ? "bg-yellow-900/40 text-yellow-300 border border-yellow-700/50"
                    : status === "RUNNING"
                      ? "bg-blue-900/40 text-blue-300 border border-blue-700/50 pulse-glow"
                      : status === "COMPLETED"
                        ? "bg-green-900/40 text-green-300 border border-green-700/50"
                        : status === "FAILED"
                          ? "bg-red-900/40 text-red-300 border border-red-700/50"
                          : "bg-gray-800/40 text-gray-300 border border-gray-600/50"
                }`}
              >
                {status}
              </span>
              {stage && (
                <span className="text-sm text-gray-400 italic bg-white/5 px-2 py-1 rounded">
                  {stage}
                </span>
              )}
              {apiConnected === "disconnected" &&
                (status === "RUNNING" || status === "PENDING") && (
                  <span className="text-xs text-red-400 bg-red-950/40 border border-red-500/30 px-2.5 py-1 rounded-full font-semibold animate-pulse flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-ping" />
                    Telemetry Offline
                  </span>
                )}
            </div>
            {onCancel && (status === "RUNNING" || status === "PENDING") && (
              <button
                onClick={onCancel}
                className="px-3 py-1.5 bg-red-900/40 hover:bg-red-800/60 border border-red-700/50 text-xs font-semibold text-red-200 rounded-lg transition-all duration-300 cursor-pointer shadow-lg shadow-red-900/10 flex items-center gap-1.5"
              >
                <Square className="w-3 h-3 fill-current" />
                <span>Cancel Task</span>
              </button>
            )}
          </div>

          <div>
            <div className="flex justify-between text-xs text-gray-400 mb-2">
              <span>Progress</span>
              <span className="font-semibold">{percentage.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-800/50 rounded-full h-3 overflow-hidden shadow-inner">
              <div
                className={`h-3 rounded-full transition-[width] duration-300 ease-out relative overflow-hidden ${
                  status === "FAILED"
                    ? "bg-gradient-to-r from-red-600 to-orange-600"
                    : status === "COMPLETED"
                      ? "bg-gradient-to-r from-green-500 to-emerald-500"
                      : status === "CANCELLED"
                        ? "bg-gradient-to-r from-gray-500 to-slate-500"
                        : "bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 shimmer"
                }`}
                style={{ width: `${Math.min(100, percentage)}%` }}
              >
                {status === "RUNNING" && (
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/25 to-transparent bg-[size:200%_100%] animate-[shimmer_2s_infinite] z-0" />
                )}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-4 bg-white/[0.02] border border-white/5 rounded-xl text-center text-xs text-gray-400">
          No active training job. Click{" "}
          <strong className="text-indigo-400 font-semibold">Run Task</strong> on the left to start a
          new run, or view history and compare results below.
        </div>
      )}

      {/* Tab Switcher */}
      <div className="flex border-b border-white/5 pb-1 gap-5">
        <button
          type="button"
          onClick={() => setActiveTab("metrics")}
          className={`pb-2 text-sm font-semibold transition-all border-b-2 cursor-pointer flex items-center gap-2 ${
            activeTab === "metrics"
              ? "text-indigo-400 border-indigo-500"
              : "text-gray-400 border-transparent hover:text-gray-200"
          }`}
        >
          <TrendingUp className="w-4 h-4" />
          <span>Metrics & Timeline</span>
        </button>
        {availablePlots.length > 0 && (
          <button
            type="button"
            onClick={() => setActiveTab("visualizations")}
            className={`pb-2 text-sm font-semibold transition-all border-b-2 cursor-pointer flex items-center gap-2 ${
              activeTab === "visualizations"
                ? "text-indigo-400 border-indigo-500"
                : "text-gray-400 border-transparent hover:text-gray-200"
            }`}
          >
            <Sparkles className="w-4 h-4" />
            <span>Model Visualizations</span>
          </button>
        )}
        <button
          type="button"
          onClick={() => setActiveTab("compare")}
          className={`pb-2 text-sm font-semibold transition-all border-b-2 cursor-pointer flex items-center gap-2 ${
            activeTab === "compare"
              ? "text-indigo-400 border-indigo-500"
              : "text-gray-400 border-transparent hover:text-gray-200"
          }`}
        >
          <History className="w-4 h-4" />
          <span>Compare & History</span>
        </button>
      </div>

      {/* Tab Contents */}
      {activeTab === "metrics" && (
        <div className="space-y-6">
          {!jobState ? (
            <div className="flex flex-col items-center justify-center py-12 px-6 text-center space-y-5 bg-white/[0.01] rounded-xl border border-white/5 animate-fade-in">
              <div className="w-16 h-16 rounded-full bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shadow-[0_0_20px_rgba(99,102,241,0.05)]">
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2m0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>
              <div className="space-y-1.5 max-w-sm">
                <h4 className="text-sm font-semibold text-white">Ready for Training Metrics</h4>
                <p className="text-xs text-gray-400 leading-relaxed">
                  Configure your hyperparameters on the left and click{" "}
                  <strong className="text-indigo-400 font-medium">Run Task</strong> to start
                  training. Live telemetry metrics and accuracy curves will appear here.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3 w-full max-w-xs pt-2 text-[10px] text-gray-500 font-medium justify-center">
                <div className="flex items-center gap-1.5 justify-center bg-white/[0.02] border border-white/5 py-1.5 px-2.5 rounded-lg">
                  <Activity className="w-3.5 h-3.5 text-amber-400" />
                  <span>Live Telemetry</span>
                </div>
                <div className="flex items-center gap-1.5 justify-center bg-white/[0.02] border border-white/5 py-1.5 px-2.5 rounded-lg">
                  <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Epoch Tracking</span>
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Connected Stepper Timeline */}
              {allStages.length > 0 && (
                <div>
                  <p className="text-xs text-gray-400 uppercase tracking-widest mb-4">
                    Training Pipeline
                  </p>
                  <ol className="flex items-start justify-between w-full overflow-x-auto py-4 custom-scrollbar">
                    {allStages.map((s, idx) => {
                      const isPast = currentStageIdx > idx;
                      const isCurrent = currentStageIdx === idx;

                      return (
                        <li
                          key={s}
                          className="flex flex-col items-center gap-2 flex-1 min-w-[85px] px-1.5 relative"
                        >
                          {/* Connector line */}
                          {idx < allStages.length - 1 && (
                            <div
                              className={`absolute top-2 left-1/2 w-full h-0.5 -translate-y-1/2 transition-colors duration-300 z-0 ${
                                isPast
                                  ? "bg-green-500"
                                  : isCurrent
                                    ? "bg-indigo-500"
                                    : "bg-gray-800"
                              }`}
                            />
                          )}

                          {/* Status Dot */}
                          <div
                            className={`relative w-4 h-4 rounded-full border-2 transition-all duration-300 z-10 ${
                              isPast
                                ? "bg-green-500 border-green-600"
                                : isCurrent
                                  ? "bg-indigo-500 border-[#0a0a0a] shadow-[0_0_10px_rgba(99,102,241,0.8)] pulse-glow"
                                  : "bg-gray-700 border-gray-600"
                            }`}
                          >
                            {isPast && (
                              <svg
                                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 text-white"
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                                strokeWidth={3}
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  d="M5 13l4 4L19 7"
                                />
                              </svg>
                            )}
                          </div>

                          {/* Stage Label */}
                          <span
                            className={`text-[10px] font-medium px-2 py-1 rounded transition-colors duration-300 min-h-[36px] flex items-center justify-center text-center w-full ${
                              isPast
                                ? "bg-green-900/20 text-green-300"
                                : isCurrent
                                  ? "bg-indigo-900/40 text-white border border-indigo-500/30"
                                  : "text-gray-500"
                            }`}
                          >
                            {s}
                          </span>
                        </li>
                      );
                    })}
                  </ol>
                </div>
              )}

              {/* Metrics chart */}
              {chartData.length > 0 && lineKeys.length > 0 ? (
                <div>
                  <p className="text-xs text-gray-400 uppercase tracking-widest mb-3">
                    Live Metrics
                  </p>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#1f2937"
                        vertical={false}
                        opacity={0.3}
                      />
                      <XAxis
                        dataKey={xAxisKey}
                        tick={{ fill: "#6B7280", fontSize: 11 }}
                        axisLine={{ stroke: "#374151" }}
                        label={{
                          value: xAxisKey,
                          position: "insideBottomRight",
                          offset: 0,
                          fill: "#6B7280",
                          fontSize: 11,
                        }}
                      />
                      <YAxis tick={{ fill: "#9CA3AF", fontSize: 11 }} />
                      <Tooltip content={CustomTooltip} />
                      <Legend wrapperStyle={{ fontSize: 12, color: "#D1D5DB" }} />
                      {lineKeys.map((key, i) => {
                        const colors = ["#6366F1", "#10B981", "#F59E0B", "#EF4444"];
                        const color = colors[i % 4];

                        return (
                          <Line
                            key={key}
                            type="monotone"
                            dataKey={key}
                            stroke={color}
                            strokeWidth={2.5}
                            dot={chartData.length < 30 ? { r: 3, strokeWidth: 1 } : false}
                            activeDot={{ r: 6, fill: color, stroke: "#0a0a0a", strokeWidth: 2 }}
                            isAnimationActive={false}
                          />
                        );
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="flex items-center justify-center h-48 text-gray-500 text-sm italic">
                  Running task initialization...
                </div>
              )}
            </>
          )}
        </div>
      )}

      {activeTab === "visualizations" && (
        <div className="space-y-4 animate-fade-in">
          {!jobState ? (
            /* Cold Start Empty State */
            <div className="flex flex-col items-center justify-center py-12 px-6 text-center space-y-4 bg-white/[0.01] rounded-xl border border-white/5">
              <div className="w-16 h-16 rounded-full bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 shadow-[0_0_20px_rgba(168,85,247,0.05)]">
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <div className="space-y-1 max-w-sm">
                <h4 className="text-sm font-semibold text-white">No Visualizations Yet</h4>
                <p className="text-xs text-gray-400 leading-relaxed">
                  Model decision boundaries, loss surfaces, or feature plots will be rendered here
                  once a training job completes successfully.
                </p>
              </div>
            </div>
          ) : status === "PENDING" || status === "RUNNING" ? (
            /* Active Run Loading Spinner */
            <div className="flex flex-col items-center justify-center py-16 text-gray-400 space-y-4 bg-white/[0.01] rounded-xl border border-white/5">
              <div className="relative w-8 h-8">
                <div className="absolute inset-0 border-2 border-indigo-500/20 rounded-full" />
                <div className="absolute inset-0 border-t-2 border-indigo-500 rounded-full animate-spin" />
              </div>
              <p className="text-xs font-medium animate-pulse text-gray-300">
                Training model and generating plots. Waiting for completion...
              </p>
            </div>
          ) : status === "FAILED" || status === "CANCELLED" ? (
            /* Run failed or cancelled */
            <div className="flex flex-col items-center justify-center py-12 px-6 text-center space-y-4 bg-white/[0.01] rounded-xl border border-white/5">
              <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                  />
                </svg>
              </div>
              <div className="space-y-1 max-w-sm">
                <h4 className="text-sm font-semibold text-red-300">Visualization Failed</h4>
                <p className="text-xs text-gray-400 leading-relaxed">
                  The training run{" "}
                  {status === "CANCELLED" ? "was cancelled" : "encountered an error"}. No new plots
                  could be generated.
                </p>
              </div>
            </div>
          ) : (
            jobId && (
              <>
                {/* Sub-tabs if multiple plots exist */}
                {availablePlots.length > 1 && (
                  <div className="flex gap-2">
                    {availablePlots.map((plot) => {
                      const cleanName = plot
                        .replace(".png", "")
                        .replace(/_/g, " ")
                        .replace(/\b\w/g, (c) => c.toUpperCase());
                      return (
                        <button
                          key={plot}
                          type="button"
                          onClick={() => setSelectedPlot(plot)}
                          className={`px-3 py-1.5 rounded-lg border text-xs font-semibold transition-all cursor-pointer ${
                            selectedPlot === plot
                              ? "bg-indigo-600/20 border-indigo-500 text-indigo-200"
                              : "bg-white/[0.02] border-white/5 text-gray-400 hover:text-gray-200 hover:border-white/10"
                          }`}
                        >
                          {cleanName}
                        </button>
                      );
                    })}
                  </div>
                )}

                {/* Plot Image Container */}
                <div className="relative group rounded-xl overflow-hidden border border-white/5 bg-black/40 flex items-center justify-center p-3 hover:border-indigo-500/20 transition-all duration-300">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`${API_BASE}/plots/${jobId}/${selectedPlot}`}
                    alt={selectedPlot}
                    onClick={() => {
                      setZoomedJobId(jobId);
                      setIsZoomed(true);
                    }}
                    className="max-h-[300px] object-contain rounded-lg cursor-zoom-in group-hover:scale-[1.01] transition-transform duration-300 select-none bg-white/[0.02]"
                  />
                  <div className="absolute bottom-4 right-4 bg-black/75 backdrop-blur-md border border-white/10 px-3 py-1.5 rounded-lg text-[10px] font-semibold text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center gap-1.5 pointer-events-none shadow-lg">
                    <Maximize2 className="w-3 h-3 text-indigo-400" />
                    <span>Click to expand</span>
                  </div>
                </div>
              </>
            )
          )}
        </div>
      )}

      {activeTab === "compare" && (
        <div className="space-y-6 animate-fade-in">
          {/* Run History */}
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest">
                Run History
              </h4>
              <span className="text-[11px] text-gray-500 font-medium">
                {recentJobs.length} {recentJobs.length === 1 ? "run" : "runs"} in session
              </span>
            </div>

            {recentJobs.length === 0 ? (
              <div className="text-center py-8 bg-white/[0.01] border border-white/5 rounded-xl text-gray-500 text-sm italic">
                No runs recorded yet. Start a run on the left!
              </div>
            ) : (
              <div className="overflow-x-auto rounded-xl border border-white/5 bg-black/20">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-white/5 text-gray-400 bg-white/[0.02]">
                      <th className="py-2.5 px-3 w-8">{/* Checkbox */}</th>
                      <th className="py-2.5 px-3 w-14">Run</th>
                      <th className="py-2.5 px-3">Time</th>
                      <th className="py-2.5 px-3">Status</th>
                      <th className="py-2.5 px-3">Hyperparameters</th>
                      <th className="py-2.5 px-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentJobs.map((job, idx) => {
                      const isCompleted = job.status === "COMPLETED";
                      const runNum = recentJobs.length - idx;
                      const formattedTime = new Date(job.created_at * 1000).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      });
                      const hpItems = Object.entries(job.config).map(([k, v]) => `${k}: ${v}`);

                      return (
                        <tr
                          key={job.job_id}
                          className="border-b border-white/5 hover:bg-white/[0.02] transition-colors"
                        >
                          <td className="py-2.5 px-3">
                            {isCompleted && (
                              <input
                                type="checkbox"
                                checked={selectedJobIds.includes(job.job_id)}
                                onChange={(e) => {
                                  if (e.target.checked) {
                                    setSelectedJobIds((prev) => [...prev, job.job_id]);
                                  } else {
                                    setSelectedJobIds((prev) =>
                                      prev.filter((id) => id !== job.job_id)
                                    );
                                  }
                                }}
                                className="rounded border-white/10 text-indigo-600 focus:ring-indigo-500 bg-gray-900 cursor-pointer"
                              />
                            )}
                          </td>
                          <td className="py-2.5 px-3 text-indigo-400 font-bold">Run #{runNum}</td>
                          <td className="py-2.5 px-3 text-gray-300 font-medium">{formattedTime}</td>
                          <td className="py-2.5 px-3">
                            <span
                              className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                                job.status === "PENDING"
                                  ? "bg-yellow-900/40 text-yellow-300 border border-yellow-700/50"
                                  : job.status === "RUNNING"
                                    ? "bg-blue-900/40 text-blue-300 border border-blue-700/50 pulse-glow"
                                    : job.status === "COMPLETED"
                                      ? "bg-green-900/40 text-green-300 border border-green-700/50"
                                      : job.status === "FAILED"
                                        ? "bg-red-900/40 text-red-300 border border-red-700/50"
                                        : "bg-gray-800/40 text-gray-300 border border-gray-600/50"
                              }`}
                            >
                              {job.status}
                            </span>
                          </td>
                          <td
                            className="py-2.5 px-3 text-gray-400 truncate max-w-xs"
                            title={hpItems.join(", ")}
                          >
                            {hpItems.length > 0 ? (
                              hpItems.join(", ")
                            ) : (
                              <span className="italic text-gray-600">none</span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 text-right">
                            {onLoadConfig && Object.keys(job.config).length > 0 && (
                              <button
                                type="button"
                                onClick={() => onLoadConfig(job.config)}
                                className="px-2 py-1 bg-indigo-600/15 hover:bg-indigo-600/35 border border-indigo-500/30 text-indigo-300 font-semibold rounded text-[10px] transition-all cursor-pointer hover:scale-[1.02] active:scale-[0.98] inline-flex items-center gap-1"
                                title="Load hyperparameters back to input form"
                              >
                                <RefreshCw className="w-2.5 h-2.5" />
                                <span>Load</span>
                              </button>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Comparison Details */}
          {selectedJobIds.length >= 2 ? (
            (() => {
              const comparedJobs = recentJobs.filter((j) => selectedJobIds.includes(j.job_id));

              // Get union of all metric keys across compared jobs
              const allMetricKeys = Array.from(
                new Set(
                  comparedJobs.flatMap((j) =>
                    j.metrics.flatMap((m) => Object.keys(m).filter((k) => k !== "percentage"))
                  )
                )
              );
              const compXAxisKey = hasEpoch
                ? "epoch"
                : (allMetricKeys.find((k) => k !== "loss" && k !== "accuracy") ?? "step");
              const compLineKeys = allMetricKeys.filter((k) => k !== compXAxisKey);

              // Let's set a default metric to compare if not selected yet
              let activeMetric = compareMetric;
              if (!activeMetric && compLineKeys.length > 0) {
                activeMetric = compLineKeys.includes("loss") ? "loss" : compLineKeys[0];
              }

              // Construct combined chart data
              const combinedDataMap: Record<number, Record<string, number>> = {};
              comparedJobs.forEach((job) => {
                const jobIdx = recentJobs.findIndex((j) => j.job_id === job.job_id);
                const runNum = jobIdx !== -1 ? recentJobs.length - jobIdx : 0;
                const formattedTime = new Date(job.created_at * 1000).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                });

                const runLabel = `Run #${runNum} @ ${formattedTime}`;

                job.metrics.forEach((m, stepIdx) => {
                  const xVal = m[compXAxisKey] !== undefined ? m[compXAxisKey] : stepIdx + 1;
                  if (!combinedDataMap[xVal]) {
                    combinedDataMap[xVal] = { [compXAxisKey]: xVal };
                  }
                  if (m[activeMetric] !== undefined) {
                    combinedDataMap[xVal][runLabel] = m[activeMetric];
                  }
                });
              });
              const compareChartData = Object.values(combinedDataMap).sort(
                (a, b) => a[compXAxisKey] - b[compXAxisKey]
              );

              // Get union of all configuration keys
              const allConfigKeys = Array.from(
                new Set(comparedJobs.flatMap((j) => Object.keys(j.config)))
              );

              return (
                <div className="space-y-6 pt-4 border-t border-white/5 animate-fade-in">
                  {/* Comparison Title */}
                  <div className="flex justify-between items-center">
                    <h3 className="text-sm font-semibold text-white flex items-center gap-1.5">
                      <Scale className="w-4 h-4 text-indigo-400" />
                      <span>Comparing {comparedJobs.length} Runs</span>
                    </h3>
                    <button
                      type="button"
                      onClick={() => setSelectedJobIds([])}
                      className="text-[10px] text-gray-400 hover:text-white underline cursor-pointer"
                    >
                      Clear Selection
                    </button>
                  </div>

                  {/* Compare Line Chart */}
                  {compLineKeys.length > 0 && activeMetric && (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-400 uppercase tracking-widest font-semibold">
                          Metric Convergence
                        </span>
                        {compLineKeys.length > 1 && (
                          <div className="flex items-center gap-2">
                            <span className="text-[10px] text-gray-500">Metric:</span>
                            <select
                              value={activeMetric}
                              onChange={(e) => setCompareMetric(e.target.value)}
                              className="bg-gray-800 border border-white/10 text-xs text-gray-200 rounded px-2 py-0.5 focus:outline-none focus:border-indigo-500"
                            >
                              {compLineKeys.map((k) => (
                                <option key={k} value={k}>
                                  {k}
                                </option>
                              ))}
                            </select>
                          </div>
                        )}
                      </div>

                      <div className="bg-black/10 rounded-xl p-3 border border-white/5">
                        <ResponsiveContainer width="100%" height={240}>
                          <LineChart
                            data={compareChartData}
                            margin={{ top: 8, right: 24, left: 0, bottom: 8 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#1f2937"
                              vertical={false}
                              opacity={0.3}
                            />
                            <XAxis
                              dataKey={compXAxisKey}
                              tick={{ fill: "#6B7280", fontSize: 11 }}
                              axisLine={{ stroke: "#374151" }}
                            />
                            <YAxis tick={{ fill: "#9CA3AF", fontSize: 11 }} />
                            <Tooltip content={CustomTooltip} />
                            <Legend wrapperStyle={{ fontSize: 10, color: "#D1D5DB" }} />
                            {comparedJobs.map((job) => {
                              const jobIdx = recentJobs.findIndex((j) => j.job_id === job.job_id);
                              const runNum = jobIdx !== -1 ? recentJobs.length - jobIdx : 0;
                              const formattedTime = new Date(
                                job.created_at * 1000
                              ).toLocaleTimeString([], {
                                hour: "2-digit",
                                minute: "2-digit",
                                second: "2-digit",
                              });

                              const runLabel = `Run #${runNum} @ ${formattedTime}`;
                              const colors = [
                                "#6366F1",
                                "#10B981",
                                "#F59E0B",
                                "#EF4444",
                                "#EC4899",
                                "#8B5CF6",
                              ];
                              const color = colors[jobIdx % colors.length];

                              return (
                                <Line
                                  key={job.job_id}
                                  type="monotone"
                                  dataKey={runLabel}
                                  stroke={color}
                                  strokeWidth={2}
                                  dot={compareChartData.length < 30 ? { r: 2 } : false}
                                  activeDot={{
                                    r: 5,
                                    fill: color,
                                    stroke: "#0a0a0a",
                                    strokeWidth: 1.5,
                                  }}
                                  isAnimationActive={false}
                                />
                              );
                            })}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}

                  {/* Hyperparameters Comparison Table */}
                  {allConfigKeys.length > 0 && (
                    <div className="space-y-3">
                      <span className="text-xs text-gray-400 uppercase tracking-widest font-semibold block">
                        Parameter Values
                      </span>
                      <div className="overflow-x-auto rounded-xl border border-white/5 bg-black/20">
                        <table className="w-full text-left border-collapse text-xs">
                          <thead>
                            <tr className="border-b border-white/10 text-gray-400 bg-white/[0.02]">
                              <th className="py-2 px-3">Parameter</th>
                              {comparedJobs.map((job) => {
                                const jobIdx = recentJobs.findIndex((j) => j.job_id === job.job_id);
                                const runNum = jobIdx !== -1 ? recentJobs.length - jobIdx : 0;
                                const formattedTime = new Date(
                                  job.created_at * 1000
                                ).toLocaleTimeString([], {
                                  hour: "2-digit",
                                  minute: "2-digit",
                                  second: "2-digit",
                                });
                                return (
                                  <th key={job.job_id} className="py-2 px-3">
                                    Run #{runNum} @ {formattedTime}
                                  </th>
                                );
                              })}
                            </tr>
                          </thead>
                          <tbody>
                            {allConfigKeys.map((key) => {
                              const vals = comparedJobs.map((j) => j.config[key]);
                              const isDiff = new Set(vals.map((v) => JSON.stringify(v))).size > 1;
                              return (
                                <tr
                                  key={key}
                                  className={`border-b border-white/5 hover:bg-white/[0.01] transition-colors ${isDiff ? "bg-amber-500/[0.04] text-amber-200" : "text-gray-300"}`}
                                >
                                  <td className="py-2 px-3 font-semibold flex items-center gap-1.5">
                                    {isDiff && (
                                      <span
                                        title="Value varies across runs"
                                        className="flex items-center"
                                      >
                                        <AlertTriangle className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
                                      </span>
                                    )}
                                    {key}
                                  </td>
                                  {vals.map((v, vIdx) => (
                                    <td key={vIdx} className="py-2 px-3 font-mono">
                                      {v !== undefined ? String(v) : "-"}
                                    </td>
                                  ))}
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Plot comparison */}
                  {plots.length > 0 && (
                    <div className="space-y-4">
                      <span className="text-xs text-gray-400 uppercase tracking-widest font-semibold block">
                        Visual Results
                      </span>
                      <div className="space-y-6">
                        {plots.map((plotName) => {
                          const cleanPlotName = plotName
                            .replace(".png", "")
                            .replace(/_/g, " ")
                            .replace(/\b\w/g, (c) => c.toUpperCase());

                          return (
                            <div key={plotName} className="space-y-2">
                              <h5 className="text-[11px] font-bold text-gray-500 uppercase tracking-wide">
                                {cleanPlotName}
                              </h5>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {comparedJobs.map((job) => {
                                  const jobIdx = recentJobs.findIndex(
                                    (j) => j.job_id === job.job_id
                                  );
                                  const runNum = jobIdx !== -1 ? recentJobs.length - jobIdx : 0;
                                  const formattedTime = new Date(
                                    job.created_at * 1000
                                  ).toLocaleTimeString([], {
                                    hour: "2-digit",
                                    minute: "2-digit",
                                    second: "2-digit",
                                  });

                                  const hpSummary = Object.entries(job.config)
                                    .map(([k, v]) => `${k}=${v}`)
                                    .join(", ");
                                  return (
                                    <div
                                      key={job.job_id}
                                      className="border border-white/5 bg-black/40 rounded-xl p-3 flex flex-col items-center space-y-2 hover:border-white/10 transition-colors"
                                    >
                                      <div className="flex justify-between items-center w-full text-[10px] font-semibold text-gray-400">
                                        <span className="text-indigo-400 font-bold">
                                          Run #{runNum} @ {formattedTime}
                                        </span>
                                        <span
                                          className="truncate max-w-[150px] font-normal"
                                          title={hpSummary}
                                        >
                                          {hpSummary || "default"}
                                        </span>
                                      </div>
                                      {/* eslint-disable-next-line @next/next/no-img-element */}
                                      <img
                                        src={`${API_BASE}/plots/${job.job_id}/${plotName}`}
                                        alt={`Run #${runNum} ${plotName}`}
                                        onClick={() => {
                                          setZoomedJobId(job.job_id);
                                          setSelectedPlot(plotName);
                                          setIsZoomed(true);
                                        }}
                                        className="max-h-[180px] object-contain rounded-lg bg-white/[0.02] cursor-zoom-in hover:scale-[1.01] transition-transform"
                                      />
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              );
            })()
          ) : selectedJobIds.length === 1 ? (
            <div className="text-center py-6 bg-white/[0.01] border border-white/5 rounded-xl text-gray-500 text-xs italic">
              Select at least one more run to view comparison charts and hyperparameter differences.
            </div>
          ) : null}
        </div>
      )}

      {/* Error message */}
      {status === "FAILED" && error && (
        <div className="p-4 bg-red-900/40 border border-red-700 rounded text-sm text-red-300 font-mono whitespace-pre-wrap">
          {error}
        </div>
      )}

      {/* Cancellation message */}
      {status === "CANCELLED" && (
        <div className="p-3 bg-gray-900/40 border border-gray-700 rounded text-sm text-gray-300 flex items-center gap-2">
          <Square className="w-3.5 h-3.5 fill-current text-gray-400" />
          <span>Task was cancelled by the user.</span>
        </div>
      )}

      {/* Completion message */}
      {status === "COMPLETED" && (
        <div className="p-3 bg-green-900/30 border border-green-700 rounded text-sm text-green-300">
          ✓ Task completed successfully.
        </div>
      )}

      {/* Lightbox / Zoom Modal */}
      {isZoomed && zoomedJobId && selectedPlot && (
        <div
          onClick={() => setIsZoomed(false)}
          className="fixed inset-0 bg-black/95 backdrop-blur-md z-[100] flex flex-col items-center justify-center cursor-zoom-out p-6 animate-fade-in"
        >
          <button
            type="button"
            onClick={() => setIsZoomed(false)}
            className="absolute top-6 right-6 p-2 rounded-full bg-white/5 hover:bg-white/10 text-white transition-all cursor-pointer border border-white/10"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>

          <div className="max-w-5xl max-h-[85vh] flex items-center justify-center p-2 rounded-xl border border-white/10 bg-[#0f0f10] shadow-2xl">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${API_BASE}/plots/${zoomedJobId}/${selectedPlot}`}
              alt={selectedPlot}
              className="max-w-full max-h-[80vh] object-contain rounded-lg select-none"
            />
          </div>

          <p className="text-xs text-gray-400 mt-4 uppercase tracking-widest font-semibold">
            {selectedPlot.replace(".png", "").replace(/_/g, " ")}
          </p>
        </div>
      )}
    </div>
  );
}
