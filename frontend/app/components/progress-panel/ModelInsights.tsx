"use client";

import { JobState } from "../../api";

interface ModelInsightsProps {
  jobState: JobState | null;
}

export default function ModelInsights({ jobState }: ModelInsightsProps) {
  if (!jobState) return null;

  const lastMetric = jobState.metrics && jobState.metrics[jobState.metrics.length - 1];
  if (!lastMetric) return null;

  const textEntries = Object.entries(lastMetric).filter(
    ([key, val]) => key !== "stage" && key !== "epoch" && key !== "step" && typeof val === "string"
  );
  if (textEntries.length === 0) return null;

  return (
    <div className="mt-4 p-4 rounded-xl border border-white/5 bg-white/[0.01] space-y-3 animate-fade-in">
      <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest block">
        Model Insights & Text Telemetry
      </span>
      <div className="grid grid-cols-1 gap-3.5">
        {textEntries.map(([key, val]) => {
          const cleanKey = key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
          return (
            <div key={key} className="space-y-1">
              <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">
                {cleanKey}
              </span>
              <div className="p-3 bg-black/35 rounded-lg border border-white/5 font-mono text-[11px] text-gray-200 whitespace-pre-wrap leading-relaxed select-text">
                {val}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
