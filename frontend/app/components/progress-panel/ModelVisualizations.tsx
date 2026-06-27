"use client";

import { useState } from "react";
import { Maximize2 } from "lucide-react";
import { JobState, API_BASE } from "../../api";
import { formatPlotName } from "../../utils/formatters";

interface ModelVisualizationsProps {
  jobState: JobState | null;
  jobId?: string | null;
  plots: string[];
  onZoom: (jobId: string, plotName: string) => void;
}

export default function ModelVisualizations({
  jobState,
  jobId,
  plots,
  onZoom,
}: ModelVisualizationsProps) {
  const [prevPlots, setPrevPlots] = useState<string[]>(plots);
  const [selectedPlot, setSelectedPlot] = useState<string>((plots && plots[0]) || "");

  // Sync state with props during render to avoid useEffect warning and double rendering
  if (plots !== prevPlots) {
    setPrevPlots(plots);
    setSelectedPlot((plots && plots[0]) || "");
  }

  if (!jobState) {
    return (
      <div className="flex flex-col items-center justify-center py-12 px-6 text-center space-y-4 bg-white/[0.01] rounded-xl border border-white/5">
        <div className="w-16 h-16 rounded-full bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 shadow-[0_0_20px_rgba(168,85,247,0.05)]">
          <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
        </div>
        <div className="space-y-1 max-w-sm">
          <h4 className="text-sm font-semibold text-white">No Visualizations Yet</h4>
          <p className="text-xs text-gray-400 leading-relaxed">
            Model decision boundaries, loss surfaces, or feature plots will be rendered here once a
            training job completes successfully.
          </p>
        </div>
      </div>
    );
  }

  const { status } = jobState;

  if (status === "PENDING" || status === "RUNNING") {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-400 space-y-4 bg-white/[0.01] rounded-xl border border-white/5">
        <div className="relative w-8 h-8">
          <div className="absolute inset-0 border-2 border-indigo-500/20 rounded-full" />
          <div className="absolute inset-0 border-t-2 border-indigo-500 rounded-full animate-spin" />
        </div>
        <p className="text-xs font-medium animate-pulse text-gray-300">
          Training model and generating plots. Waiting for completion...
        </p>
      </div>
    );
  }

  if (status === "FAILED" || status === "CANCELLED") {
    return (
      <div className="flex flex-col items-center justify-center py-12 px-6 text-center space-y-4 bg-white/[0.01] rounded-xl border border-white/5">
        <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400">
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
        </div>
        <div className="space-y-1 max-w-sm">
          <h4 className="text-sm font-semibold text-red-300">Visualization Failed</h4>
          <p className="text-xs text-gray-400 leading-relaxed">
            The training run {status === "CANCELLED" ? "was cancelled" : "encountered an error"}. No
            new plots could be generated.
          </p>
        </div>
      </div>
    );
  }

  if (!jobId || !selectedPlot) return null;

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Sub-tabs if multiple plots exist */}
      {plots && plots.length > 1 && (
        <div className="flex gap-2">
          {plots.map((plot) => {
            const cleanName = formatPlotName(plot);
            return (
              <button
                key={plot}
                type="button"
                onClick={() => setSelectedPlot(plot)}
                className={`px-3 py-1.5 rounded-lg border text-xs font-semibold transition-all cursor-pointer ${
                  selectedPlot === plot
                    ? "bg-indigo-600/20 border-indigo-500 text-indigo-200"
                    : "bg-white/[0.02] border-white/5 text-gray-400 hover:text-gray-200 hover:border-white/10"
                }`}
              >
                {cleanName}
              </button>
            );
          })}
        </div>
      )}

      {/* Plot Image Container */}
      <div className="relative group rounded-xl overflow-hidden border border-white/5 bg-black/40 flex items-center justify-center p-3 hover:border-indigo-500/20 transition-all duration-300">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`${API_BASE}/plots/${jobId}/${selectedPlot}`}
          alt={selectedPlot}
          onClick={() => onZoom(jobId, selectedPlot)}
          className="max-h-[300px] object-contain rounded-lg cursor-zoom-in group-hover:scale-[1.01] transition-transform duration-300 select-none bg-white/[0.02]"
        />
        <div className="absolute bottom-4 right-4 bg-black/75 backdrop-blur-md border border-white/10 px-3 py-1.5 rounded-lg text-[10px] font-semibold text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center gap-1.5 pointer-events-none shadow-lg">
          <Maximize2 className="w-3 h-3 text-indigo-400" />
          <span>Click to expand</span>
        </div>
      </div>
    </div>
  );
}
