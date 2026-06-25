"use client";

import { useEffect, useState } from "react";
import { TrendingUp, Sparkles, History, Square, Activity, BarChart3 } from "lucide-react";
import { JobState, JobInfo, fetchJobs, Task } from "../api";

// Import our new refactored sub-components
import ActiveRunCard from "./progress-panel/ActiveRunCard";
import PipelineTimeline from "./progress-panel/PipelineTimeline";
import MetricsChart from "./progress-panel/MetricsChart";
import ModelVisualizations from "./progress-panel/ModelVisualizations";
import ComparisonTab from "./progress-panel/ComparisonTab";
import ZoomModal from "./progress-panel/ZoomModal";

interface ProgressPanelProps {
  jobState: JobState | null;
  stages: string[];
  onCancel?: () => void;
  jobId?: string | null;
  plots: string[];
  selectedTask?: Task | null;
  onLoadConfig?: (config: Record<string, number>) => void;
  apiConnected?: "connected" | "disconnected" | "checking";
}

export default function ProgressPanel({
  jobState,
  stages,
  onCancel,
  jobId,
  plots,
  selectedTask,
  onLoadConfig,
  apiConnected = "connected",
}: ProgressPanelProps) {
  const availablePlots = plots ?? [];

  const [activeTab, setActiveTab] = useState<"metrics" | "visualizations" | "compare">("metrics");

  // Zoom Lightbox State
  const [isZoomed, setIsZoomed] = useState<boolean>(false);
  const [zoomedJobId, setZoomedJobId] = useState<string | null>(null);
  const [zoomedPlotName, setZoomedPlotName] = useState<string | null>(null);

  // History state
  const [recentJobs, setRecentJobs] = useState<JobInfo[]>([]);

  const status = jobState?.status;
  const error = jobState?.error;

  // Fetch job history for this task
  useEffect(() => {
    if (!selectedTask || apiConnected !== "connected") return;
    fetchJobs(selectedTask.module, selectedTask.task)
      .then((data) => {
        setRecentJobs(data);
      })
      .catch((err) => {
        console.error("Failed to load run history", err);
      });
  }, [selectedTask, status, apiConnected]);

  // Handler to open zoom modal
  const handleZoom = (id: string, name: string) => {
    setZoomedJobId(id);
    setZoomedPlotName(name);
    setIsZoomed(true);
  };

  return (
    <div className="space-y-6">
      {/* Active Run Status/Progress Card */}
      <ActiveRunCard jobState={jobState} onCancel={onCancel} apiConnected={apiConnected} />

      {/* Tab Switcher */}
      <div className="flex border-b border-white/5 pb-1 gap-5">
        <button
          type="button"
          onClick={() => setActiveTab("metrics")}
          className={`pb-2 text-sm font-semibold transition-all border-b-2 cursor-pointer flex items-center gap-2 ${
            activeTab === "metrics"
              ? "text-indigo-400 border-indigo-500"
              : "text-gray-400 border-transparent hover:text-gray-200"
          }`}
        >
          <TrendingUp className="w-4 h-4" />
          <span>Metrics & Timeline</span>
        </button>
        {availablePlots.length > 0 && (
          <button
            type="button"
            onClick={() => setActiveTab("visualizations")}
            className={`pb-2 text-sm font-semibold transition-all border-b-2 cursor-pointer flex items-center gap-2 ${
              activeTab === "visualizations"
                ? "text-indigo-400 border-indigo-500"
                : "text-gray-400 border-transparent hover:text-gray-200"
            }`}
          >
            <Sparkles className="w-4 h-4" />
            <span>Model Visualizations</span>
          </button>
        )}
        <button
          type="button"
          onClick={() => setActiveTab("compare")}
          className={`pb-2 text-sm font-semibold transition-all border-b-2 cursor-pointer flex items-center gap-2 ${
            activeTab === "compare"
              ? "text-indigo-400 border-indigo-500"
              : "text-gray-400 border-transparent hover:text-gray-200"
          }`}
        >
          <History className="w-4 h-4" />
          <span>Compare & History</span>
        </button>
      </div>

      {/* Tab Contents */}
      {activeTab === "metrics" && (
        <div className="space-y-6">
          {!jobState ? (
            <div className="flex flex-col items-center justify-center py-12 px-6 text-center space-y-5 bg-white/[0.01] rounded-xl border border-white/5 animate-fade-in">
              <div className="w-16 h-16 rounded-full bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shadow-[0_0_20px_rgba(99,102,241,0.05)]">
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2m0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>
              <div className="space-y-1.5 max-w-sm">
                <h4 className="text-sm font-semibold text-white">Ready for Training Metrics</h4>
                <p className="text-xs text-gray-400 leading-relaxed">
                  Configure your hyperparameters on the left and click{" "}
                  <strong className="text-indigo-400 font-medium">Run Task</strong> to start
                  training. Live telemetry metrics and accuracy curves will appear here.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3 w-full max-w-xs pt-2 text-[10px] text-gray-500 font-medium justify-center">
                <div className="flex items-center gap-1.5 justify-center bg-white/[0.02] border border-white/5 py-1.5 px-2.5 rounded-lg">
                  <Activity className="w-3.5 h-3.5 text-amber-400" />
                  <span>Live Telemetry</span>
                </div>
                <div className="flex items-center gap-1.5 justify-center bg-white/[0.02] border border-white/5 py-1.5 px-2.5 rounded-lg">
                  <BarChart3 className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Epoch Tracking</span>
                </div>
              </div>
            </div>
          ) : (
            <>
              <PipelineTimeline stages={stages} currentStage={jobState.stage} />
              <MetricsChart metrics={jobState.metrics ?? []} />
            </>
          )}
        </div>
      )}

      {activeTab === "visualizations" && (
        <ModelVisualizations
          jobState={jobState}
          jobId={jobId}
          plots={availablePlots}
          onZoom={handleZoom}
        />
      )}

      {activeTab === "compare" && (
        <ComparisonTab
          recentJobs={recentJobs}
          plots={availablePlots}
          onLoadConfig={onLoadConfig}
          onZoom={handleZoom}
        />
      )}

      {/* Error message */}
      {status === "FAILED" && error && (
        <div className="p-4 bg-red-900/40 border border-red-700 rounded text-sm text-red-300 font-mono whitespace-pre-wrap">
          {error}
        </div>
      )}

      {/* Cancellation message */}
      {status === "CANCELLED" && (
        <div className="p-3 bg-gray-900/40 border border-gray-700 rounded text-sm text-gray-300 flex items-center gap-2">
          <Square className="w-3.5 h-3.5 fill-current text-gray-400" />
          <span>Task was cancelled by the user.</span>
        </div>
      )}

      {/* Completion message */}
      {status === "COMPLETED" && (
        <div className="p-3 bg-green-900/30 border border-green-700 rounded text-sm text-green-300">
          ✓ Task completed successfully.
        </div>
      )}

      {/* Lightbox / Zoom Modal */}
      {isZoomed && zoomedJobId && zoomedPlotName && (
        <ZoomModal
          isOpen={isZoomed}
          onClose={() => setIsZoomed(false)}
          jobId={zoomedJobId}
          plotName={zoomedPlotName}
        />
      )}
    </div>
  );
}
