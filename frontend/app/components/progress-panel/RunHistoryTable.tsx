"use client";

import { RefreshCw } from "lucide-react";
import { JobInfo } from "../../api";
import StatusBadge from "./StatusBadge";
import { formatRunTime, formatRunConfig } from "../../utils/formatters";

interface RunHistoryTableProps {
  recentJobs: JobInfo[];
  selectedJobIds: string[];
  setSelectedJobIds: React.Dispatch<React.SetStateAction<string[]>>;
  onLoadConfig?: (config: Record<string, number>) => void;
}

export default function RunHistoryTable({
  recentJobs,
  selectedJobIds,
  setSelectedJobIds,
  onLoadConfig,
}: RunHistoryTableProps) {
  return (
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
            const formattedTime = formatRunTime(job.created_at);
            const hpSummary = formatRunConfig(job.config);

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
                  <StatusBadge status={job.status} size="sm" />
                </td>
                <td className="py-2.5 px-3 text-gray-400 truncate max-w-xs" title={hpSummary}>
                  {hpSummary ? hpSummary : <span className="italic text-gray-600">none</span>}
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
  );
}
