"use client";

import { Square } from "lucide-react";
import { JobState } from "../../api";
import StatusBadge from "./StatusBadge";

interface ActiveRunCardProps {
  jobState: JobState | null;
  onCancel?: () => void;
  apiConnected: "connected" | "disconnected" | "checking";
}

export default function ActiveRunCard({ jobState, onCancel, apiConnected }: ActiveRunCardProps) {
  if (!jobState) {
    return (
      <div className="p-4 bg-white/[0.02] border border-white/5 rounded-xl text-center text-xs text-gray-400">
        No active training job. Click{" "}
        <strong className="text-indigo-400 font-semibold">Run Task</strong> on the left to start a
        new run, or view history and compare results below.
      </div>
    );
  }

  const { status, stage, percentage = 0.0 } = jobState;

  return (
    <div className="space-y-4 pb-4 border-b border-white/5">
      <div className="flex items-center justify-between w-full">
        <div className="flex items-center gap-3">
          <StatusBadge status={status} />
          {stage && (
            <span className="text-sm text-gray-400 italic bg-white/5 px-2 py-1 rounded">
              {stage}
            </span>
          )}
          {apiConnected === "disconnected" && (status === "RUNNING" || status === "PENDING") && (
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
  );
}
