"use client";

import { useState } from "react";
import { Scale } from "lucide-react";
import { JobInfo } from "../../api";

// Decomposed Sub-components
import RunHistoryTable from "./RunHistoryTable";
import ComparisonChart from "./ComparisonChart";
import ParameterDiffTable from "./ParameterDiffTable";
import PlotComparisonMatrix from "./PlotComparisonMatrix";

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

  if (recentJobs.length === 0) {
    return (
      <div className="text-center py-8 bg-white/[0.01] border border-white/5 rounded-xl text-gray-500 text-sm italic animate-fade-in">
        No runs recorded yet. Start a run on the left!
      </div>
    );
  }

  const comparedJobs = recentJobs.filter((j) => selectedJobIds.includes(j.job_id));

  // Determine if epoch exists in any metric in the history
  const hasEpoch = recentJobs.some((j) => j.metrics.some((m) => "epoch" in m));

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

        <RunHistoryTable
          recentJobs={recentJobs}
          selectedJobIds={selectedJobIds}
          setSelectedJobIds={setSelectedJobIds}
          onLoadConfig={onLoadConfig}
        />
      </div>

      {/* Comparison Details */}
      {selectedJobIds.length >= 2 ? (
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

          {/* Metric line chart */}
          <ComparisonChart
            comparedJobs={comparedJobs}
            recentJobs={recentJobs}
            compXAxisKey={compXAxisKey}
            compLineKeys={compLineKeys}
          />

          {/* Config variance table */}
          <ParameterDiffTable comparedJobs={comparedJobs} recentJobs={recentJobs} />

          {/* Plots comparison matrix */}
          <PlotComparisonMatrix
            comparedJobs={comparedJobs}
            recentJobs={recentJobs}
            plots={plots}
            onZoom={onZoom}
          />
        </div>
      ) : selectedJobIds.length === 1 ? (
        <div className="text-center py-6 bg-white/[0.01] border border-white/5 rounded-xl text-gray-500 text-xs italic">
          Select at least one more run to view comparison charts and hyperparameter differences.
        </div>
      ) : null}
    </div>
  );
}
