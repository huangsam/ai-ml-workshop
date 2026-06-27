"use client";

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
import CustomTooltip from "./CustomTooltip";

interface MetricsChartProps {
  metrics: Record<string, number>[];
}

export default function MetricsChart({ metrics }: MetricsChartProps) {
  if (metrics.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm italic">
        Running task initialization...
      </div>
    );
  }

  // Determine which metrics keys exist across all snapshots for active run
  const metricKeys = Array.from(
    new Set(metrics.flatMap((m) => Object.keys(m).filter((k) => k !== "percentage")))
  );

  // Use the first metric key as the X-axis (e.g. "epoch", "step"), or fall back to index.
  const hasEpoch = metricKeys.includes("epoch");
  const xAxisKey = hasEpoch
    ? "epoch"
    : (metricKeys.find((k) => k !== "loss" && k !== "accuracy") ?? "step");
  const lineKeys = metricKeys.filter(
    (key) => key !== xAxisKey && metrics.some((m) => typeof m[key] === "number")
  );

  // Map metrics to ensure every point has an X-axis value
  const chartData = metrics.map((m, idx) => {
    const pt = { ...m };
    if (pt[xAxisKey] === undefined) {
      pt[xAxisKey] = idx + 1;
    }
    return pt;
  });

  if (lineKeys.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-500 text-sm italic">
        Waiting for initial metrics...
      </div>
    );
  }

  return (
    <div>
      <p className="text-xs text-gray-400 uppercase tracking-widest mb-3">Live Metrics</p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} opacity={0.3} />
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
  );
}
