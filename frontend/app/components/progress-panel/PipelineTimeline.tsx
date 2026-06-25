"use client";

interface PipelineTimelineProps {
  stages: string[];
  currentStage?: string;
}

export default function PipelineTimeline({ stages, currentStage }: PipelineTimelineProps) {
  const allStages = stages.length > 0 ? [...stages] : [];
  if (currentStage && !allStages.includes(currentStage)) {
    allStages.push(currentStage);
  }

  if (allStages.length === 0) return null;

  const currentStageIdx = currentStage ? allStages.indexOf(currentStage) : -1;

  return (
    <div>
      <p className="text-xs text-gray-400 uppercase tracking-widest mb-4">Training Pipeline</p>
      <ol className="flex items-start justify-between w-full overflow-x-auto py-4 custom-scrollbar">
        {allStages.map((s, idx) => {
          const isPast = currentStageIdx > idx;
          const isCurrent = currentStageIdx === idx;

          return (
            <li
              key={s}
              className="flex flex-col items-center gap-2 flex-1 min-w-[85px] px-1.5 relative"
            >
              {/* Connector line */}
              {idx < allStages.length - 1 && (
                <div
                  className={`absolute top-2 left-1/2 w-full h-0.5 -translate-y-1/2 transition-colors duration-300 z-0 ${
                    isPast ? "bg-green-500" : isCurrent ? "bg-indigo-500" : "bg-gray-800"
                  }`}
                />
              )}

              {/* Status Dot */}
              <div
                className={`relative w-4 h-4 rounded-full border-2 transition-all duration-300 z-10 ${
                  isPast
                    ? "bg-green-500 border-green-600"
                    : isCurrent
                      ? "bg-indigo-500 border-[#0a0a0a] shadow-[0_0_10px_rgba(99,102,241,0.8)] pulse-glow"
                      : "bg-gray-700 border-gray-600"
                }`}
              >
                {isPast && (
                  <svg
                    className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-3 text-white"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={3}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </div>

              {/* Stage Label */}
              <span
                className={`text-[10px] font-medium px-2 py-1 rounded transition-colors duration-300 min-h-[36px] flex items-center justify-center text-center w-full ${
                  isPast
                    ? "bg-green-900/20 text-green-300"
                    : isCurrent
                      ? "bg-indigo-900/40 text-white border border-indigo-500/30"
                      : "text-gray-500"
                }`}
              >
                {s}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
