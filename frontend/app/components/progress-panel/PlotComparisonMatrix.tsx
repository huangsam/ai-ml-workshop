"use client";

import { JobInfo, API_BASE } from "../../api";
import { formatPlotName, formatRunTime, formatRunConfigCompact } from "../../utils/formatters";

interface PlotComparisonMatrixProps {
  comparedJobs: JobInfo[];
  recentJobs: JobInfo[];
  plots: string[];
  onZoom: (jobId: string, plotName: string) => void;
}

export default function PlotComparisonMatrix({
  comparedJobs,
  recentJobs,
  plots,
  onZoom,
}: PlotComparisonMatrixProps) {
  if (plots.length === 0) return null;

  return (
    <div className="space-y-4">
      <span className="text-xs text-gray-400 uppercase tracking-widest font-semibold block">
        Visual Results
      </span>
      <div className="space-y-6">
        {plots.map((plotName) => {
          const cleanPlotName = formatPlotName(plotName);

          return (
            <div key={plotName} className="space-y-2">
              <h5 className="text-[11px] font-bold text-gray-500 uppercase tracking-wide">
                {cleanPlotName}
              </h5>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {comparedJobs.map((job) => {
                  const jobIdx = recentJobs.findIndex((j) => j.job_id === job.job_id);
                  const runNum = jobIdx !== -1 ? recentJobs.length - jobIdx : 0;
                  const formattedTime = formatRunTime(job.created_at);
                  const hpSummary = formatRunConfigCompact(job.config);

                  return (
                    <div
                      key={job.job_id}
                      className="border border-white/5 bg-black/40 rounded-xl p-3 flex flex-col items-center space-y-2 hover:border-white/10 transition-colors"
                    >
                      <div className="flex justify-between items-center w-full text-[10px] font-semibold text-gray-400">
                        <span className="text-indigo-400 font-bold">
                          Run #{runNum} @ {formattedTime}
                        </span>
                        <span className="truncate max-w-[150px] font-normal" title={hpSummary}>
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
  );
}
