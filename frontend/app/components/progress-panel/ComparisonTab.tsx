"use client";

import { useState } from "react";
import { Scale, AlertTriangle, RefreshCw } from "lucide-react";
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
import { JobInfo, API_BASE } from "../../api";
import CustomTooltip from "./CustomTooltip";

interface ComparisonTabProps {
  recentJobs: JobInfo[];
  plots: string[];
  onLoadConfig?: (config: Record<string, number>) => void;
  onZoom: (jobId: string, plotName: string) => void;
}

export default function ComparisonTab({
  recentJobs,
  plots,
  onLoadConfig,
  onZoom,
}: ComparisonTabProps) {
  const [selectedJobIds, setSelectedJobIds] = useState<string[]>([]);
  const [compareMetric, setCompareMetric] = useState<string>("");

  if (recentJobs.length === 0) {
    return (
      <div className="text-center py-8 bg-white/[0.01] border border-white/5 rounded-xl text-gray-500 text-sm italic animate-fade-in">
        No runs recorded yet. Start a run on the left!
      </div>
    );
  }

  // Determine if epoch exists in any metric in the history
  const hasEpoch = recentJobs.some((j) => j.metrics.some((m) => "epoch" in m));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Run History */}
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest">Run History</h4>
          <span className="text-[11px] text-gray-500 font-medium">
            {recentJobs.length} {recentJobs.length === 1 ? "run" : "runs"} in session
          </span>
        </div>

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
                              setSelectedJobIds((prev) => prev.filter((id) => id !== job.job_id));
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
                          const formattedTime = new Date(job.created_at * 1000).toLocaleTimeString(
                            [],
                            {
                              hour: "2-digit",
                              minute: "2-digit",
                              second: "2-digit",
                            }
                          );

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
                              const jobIdx = recentJobs.findIndex((j) => j.job_id === job.job_id);
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
                                    onClick={() => onZoom(job.job_id, plotName)}
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
  );
}
