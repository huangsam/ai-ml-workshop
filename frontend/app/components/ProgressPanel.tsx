"use client";

import { useEffect, useState } from "react";
import {
  TrendingUp,
  Sparkles,
  History,
  Square,
  Activity,
  BarChart3,
  Code2,
  Terminal,
} from "lucide-react";
import { JobState, JobInfo, fetchJobs, Task } from "../api";

// Import our sub-components
import ActiveRunCard from "./progress-panel/ActiveRunCard";
import PipelineTimeline from "./progress-panel/PipelineTimeline";
import MetricsChart from "./progress-panel/MetricsChart";
import ModelVisualizations from "./progress-panel/ModelVisualizations";
import ComparisonTab from "./progress-panel/ComparisonTab";
import ZoomModal from "./progress-panel/ZoomModal";
import CodeViewer from "./progress-panel/CodeViewer";
import ConsoleTerminal from "./progress-panel/ConsoleTerminal";
import ModelInsights from "./progress-panel/ModelInsights";

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

  const [activeTab, setActiveTab] = useState<
    "metrics" | "visualizations" | "compare" | "code" | "terminal"
  >("metrics");

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
      <div className="inline-flex flex-wrap items-center bg-white/[0.02] border border-white/5 p-1.5 rounded-xl gap-1 shadow-inner select-none">
        <button
          type="button"
          onClick={() => setActiveTab("metrics")}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 select-none ${
            activeTab === "metrics"
              ? "bg-indigo-500/15 border border-indigo-500/20 text-indigo-400 shadow-sm"
              : "border border-transparent text-gray-400 hover:bg-white/[0.03] hover:text-gray-200"
          }`}
        >
          <TrendingUp className="w-3.5 h-3.5" />
          <span>Metrics</span>
        </button>
        {availablePlots.length > 0 && (
          <button
            type="button"
            onClick={() => setActiveTab("visualizations")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 select-none ${
              activeTab === "visualizations"
                ? "bg-indigo-500/15 border border-indigo-500/20 text-indigo-400 shadow-sm"
                : "border border-transparent text-gray-400 hover:bg-white/[0.03] hover:text-gray-200"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Visualizations</span>
          </button>
        )}
        {selectedTask && (
          <button
            type="button"
            onClick={() => setActiveTab("code")}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 select-none ${
              activeTab === "code"
                ? "bg-indigo-500/15 border border-indigo-500/20 text-indigo-400 shadow-sm"
                : "border border-transparent text-gray-400 hover:bg-white/[0.03] hover:text-gray-200"
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            <span>Code</span>
          </button>
        )}
        <button
          type="button"
          onClick={() => setActiveTab("terminal")}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 select-none ${
            activeTab === "terminal"
              ? "bg-indigo-500/15 border border-indigo-500/20 text-indigo-400 shadow-sm"
              : "border border-transparent text-gray-400 hover:bg-white/[0.03] hover:text-gray-200"
          }`}
        >
          <Terminal className="w-3.5 h-3.5" />
          <span>Logs</span>
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("compare")}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 select-none ${
            activeTab === "compare"
              ? "bg-indigo-500/15 border border-indigo-500/20 text-indigo-400 shadow-sm"
              : "border border-transparent text-gray-400 hover:bg-white/[0.03] hover:text-gray-200"
          }`}
        >
          <History className="w-3.5 h-3.5" />
          <span>History</span>
        </button>
      </div>

      {/* Tab Contents */}
      {activeTab === "metrics" && (
        <div className="space-y-6">
          {!jobState ? (
            <div className="flex flex-col items-center justify-center py-20 px-6 text-center space-y-5 bg-white/[0.01] rounded-xl border border-white/5 min-h-[290px] animate-fade-in">
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
              <ModelInsights jobState={jobState} />
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

      {activeTab === "code" && selectedTask && <CodeViewer task={selectedTask} />}

      {activeTab === "terminal" && (
        <ConsoleTerminal logs={jobState?.logs ?? ""} status={jobState?.status} />
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
