"use client";

import { AlertTriangle } from "lucide-react";
import { JobInfo } from "../../api";
import { formatRunTime } from "../../utils/formatters";

interface ParameterDiffTableProps {
  comparedJobs: JobInfo[];
  recentJobs: JobInfo[];
}

export default function ParameterDiffTable({ comparedJobs, recentJobs }: ParameterDiffTableProps) {
  // Get union of all configuration keys
  const allConfigKeys = Array.from(new Set(comparedJobs.flatMap((j) => Object.keys(j.config))));

  if (allConfigKeys.length === 0) return null;

  return (
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
                const formattedTime = formatRunTime(job.created_at);
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
                  className={`border-b border-white/5 hover:bg-white/[0.01] transition-colors ${
                    isDiff ? "bg-amber-500/[0.04] text-amber-200" : "text-gray-300"
                  }`}
                >
                  <td className="py-2 px-3 font-semibold flex items-center gap-1.5">
                    {isDiff && (
                      <span title="Value varies across runs" className="flex items-center">
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
  );
}
