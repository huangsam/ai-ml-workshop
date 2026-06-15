"use client";

import { JobState } from "../api";
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
            <span className="text-gray-300 font-medium">{entry.name?.toString()}:</span>
            <span className="text-white font-semibold">
              {typeof entry.value === "number" ? entry.value.toFixed(4) : String(entry.value ?? "")}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default function ProgressPanel({ jobState, stages, onCancel }: ProgressPanelProps) {
  if (!jobState) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        Select a task and click <strong className="mx-1">Run</strong> to begin.
      </div>
    );
  }

  const { status, stage, percentage, metrics, error } = jobState;

  // Determine which metrics keys exist across all snapshots
  const metricKeys = Array.from(
    new Set(metrics.flatMap((m) => Object.keys(m).filter((k) => k !== "percentage")))
  );

  // Use the first metric key as the X-axis (e.g. "epoch", "step"), or fall back to index.
  const hasEpoch = metricKeys.includes("epoch");
  const xAxisKey = hasEpoch
    ? "epoch"
    : (metricKeys.find((k) => k !== "loss" && k !== "accuracy") ?? "step");
  const lineKeys = metricKeys.filter((k) => k !== xAxisKey);

  // Map metrics to ensure every point has an X-axis value (falls back to index if missing)
  const chartData = metrics.map((m, idx) => {
    const pt = { ...m };
    if (pt[xAxisKey] === undefined) {
      pt[xAxisKey] = idx + 1;
    }
    return pt;
  });

  // Build stage list dynamically – include current stage if not yet in the list
  const allStages = stages.length > 0 ? [...stages] : [];
  if (stage && !allStages.includes(stage)) {
    allStages.push(stage);
  }
  const currentStageIdx = allStages.indexOf(stage);

  return (
    <div className="space-y-6">
      {/* Status badge and Cancel button */}
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
        </div>
        {onCancel && (status === "RUNNING" || status === "PENDING") && (
          <button
            onClick={onCancel}
            className="px-3 py-1.5 bg-red-900/40 hover:bg-red-800/60 border border-red-700/50 text-xs font-semibold text-red-200 rounded-lg transition-all duration-300 cursor-pointer shadow-lg shadow-red-900/10"
          >
            ⏹ Cancel Task
          </button>
        )}
      </div>

      {/* Linear progress bar */}
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

      {/* Connected Stepper Timeline */}
      {allStages.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-widest mb-4">Training Pipeline</p>
          <ol className="flex items-center justify-between w-full overflow-x-auto pb-4 custom-scrollbar">
            {allStages.map((s, idx) => {
              const isPast = currentStageIdx > idx;
              const isCurrent = currentStageIdx === idx;

              return (
                <li
                  key={s}
                  className="flex flex-col items-center gap-2 flex-1 min-w-[80px] relative"
                >
                  {/* Connector line (before the dot) */}
                  {idx < allStages.length - 1 && (
                    <div
                      className={`absolute top-2 left-1/2 w-full h-0.5 -translate-y-1/2 transition-colors duration-300 z-0 ${
                        isPast ? "bg-green-500" : isCurrent ? "bg-indigo-500" : "bg-gray-800"
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
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </div>

                  {/* Stage Label */}
                  <span
                    className={`text-[10px] font-medium px-2 py-1 rounded transition-colors duration-300 ${
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
      {chartData.length > 0 && lineKeys.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 uppercase tracking-widest mb-3">Live Metrics</p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
              {/* Custom Grid with subtle styling */}
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

              {/* Custom Glassmorphic Tooltip */}
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
      )}

      {/* Error message */}
      {status === "FAILED" && error && (
        <div className="p-4 bg-red-900/40 border border-red-700 rounded text-sm text-red-300 font-mono whitespace-pre-wrap">
          {error}
        </div>
      )}

      {/* Cancellation message */}
      {status === "CANCELLED" && (
        <div className="p-3 bg-gray-900/40 border border-gray-700 rounded text-sm text-gray-300">
          ⏹ Task was cancelled by the user.
        </div>
      )}

      {/* Completion message */}
      {status === "COMPLETED" && (
        <div className="p-3 bg-green-900/30 border border-green-700 rounded text-sm text-green-300">
          ✓ Task completed successfully.
        </div>
      )}
    </div>
  );
}
