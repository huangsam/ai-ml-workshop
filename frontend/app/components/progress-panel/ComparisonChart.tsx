"use client";

import { useState } from "react";
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
import { JobInfo } from "../../api";
import CustomTooltip from "./CustomTooltip";
import { formatRunTime } from "../../utils/formatters";

interface ComparisonChartProps {
  comparedJobs: JobInfo[];
  recentJobs: JobInfo[];
  compXAxisKey: string;
  compLineKeys: string[];
}

export default function ComparisonChart({
  comparedJobs,
  recentJobs,
  compXAxisKey,
  compLineKeys,
}: ComparisonChartProps) {
  const [compareMetric, setCompareMetric] = useState<string>("");

  if (compLineKeys.length === 0) return null;

  let activeMetric = compareMetric;
  if (!activeMetric && compLineKeys.length > 0) {
    activeMetric = compLineKeys.includes("loss") ? "loss" : compLineKeys[0];
  }

  // Construct combined chart data
  const combinedDataMap: Record<number, Record<string, number>> = {};
  comparedJobs.forEach((job) => {
    const jobIdx = recentJobs.findIndex((j) => j.job_id === job.job_id);
    const runNum = jobIdx !== -1 ? recentJobs.length - jobIdx : 0;
    const formattedTime = formatRunTime(job.created_at);
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

  return (
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
          <LineChart data={compareChartData} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} opacity={0.3} />
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
              const formattedTime = formatRunTime(job.created_at);
              const runLabel = `Run #${runNum} @ ${formattedTime}`;

              const colors = ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#EC4899", "#8B5CF6"];
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
  );
}
