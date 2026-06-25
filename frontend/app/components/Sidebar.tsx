"use client";

import { Cpu } from "lucide-react";
import { MODULE_LABELS, TASK_LABELS } from "../constants";
import { Task } from "../api";

/** Group tasks by module name */
function groupByModule(tasks: Task[]): Record<string, Task[]> {
  return tasks.reduce<Record<string, Task[]>>((acc, t) => {
    (acc[t.module] ??= []).push(t);
    return acc;
  }, {});
}

interface SidebarProps {
  tasks: Task[];
  selected: Task | null;
  onSelect: (task: Task | null) => void;
  apiConnected: "connected" | "disconnected" | "checking";
}

export default function Sidebar({ tasks, selected, onSelect, apiConnected }: SidebarProps) {
  const groups = groupByModule(tasks);

  return (
    <aside className="glass-sidebar w-64 text-gray-100 min-h-screen flex flex-col justify-between">
      <div className="flex flex-col flex-1 overflow-y-hidden">
        <button
          onClick={() => onSelect(null)}
          className="px-5 py-6 border-b border-white/5 text-left hover:bg-white/[0.02] active:bg-white/[0.04] transition-all duration-300 cursor-pointer group w-full flex-shrink-0"
        >
          <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent group-hover:from-indigo-300 group-hover:to-purple-300 transition-all duration-300 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400 group-hover:text-indigo-300 transition-colors flex-shrink-0" />
            <span>ML Workshop</span>
          </h1>
          <p className="text-xs text-gray-400 mt-1">Interactive learning platform</p>
        </button>

        <nav className="flex-1 overflow-y-auto py-4 custom-scrollbar">
          {Object.entries(groups).map(([module, moduleTasks]) => (
            <div key={module} className="mb-4 first:mt-0">
              <p className="px-5 mb-2 text-[10px] font-bold uppercase tracking-widest text-indigo-400/70">
                {MODULE_LABELS[module] ?? module}
              </p>
              {moduleTasks.map((t) => {
                const isActive = selected?.module === t.module && selected?.task === t.task;
                return (
                  <button
                    key={`${t.module}/${t.task}`}
                    onClick={() => onSelect(t)}
                    className={`relative w-full text-left pl-9 pr-5 py-2 text-sm transition-all duration-300 group ${
                      isActive ? "text-white" : "text-gray-300 hover:bg-white/5 hover:text-gray-100"
                    }`}
                  >
                    {isActive && (
                      <>
                        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-full h-full bg-gradient-to-r from-indigo-600/20 to-purple-600/20 rounded animate-active-glow" />
                        <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-gradient-to-b from-indigo-500 to-purple-600 rounded-r shadow-[0_0_10px_rgba(99,102,241,0.5)]" />
                      </>
                    )}
                    <span
                      className={`${
                        isActive ? "font-medium" : "font-normal"
                      } transition-all duration-300 group-hover:translate-x-1 inline-block`}
                    >
                      {TASK_LABELS[t.task] ?? t.task}
                    </span>
                  </button>
                );
              })}
            </div>
          ))}
        </nav>
      </div>

      {/* API Connection Status Footer */}
      <div className="p-4 border-t border-white/5 bg-black/25 flex items-center justify-between text-xs text-gray-400 flex-shrink-0">
        <span className="font-semibold tracking-wider uppercase text-[10px] text-gray-500">
          API Status
        </span>
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full transition-all duration-500 ${
              apiConnected === "connected"
                ? "bg-emerald-500 shadow-[0_0_8px_#10b981]"
                : apiConnected === "disconnected"
                  ? "bg-red-500 shadow-[0_0_8px_#ef4444] animate-pulse"
                  : "bg-amber-500 shadow-[0_0_8px_#f59e0b] animate-pulse"
            }`}
          />
          <span className="font-medium text-gray-300">
            {apiConnected === "connected"
              ? "Online"
              : apiConnected === "disconnected"
                ? "Offline"
                : "Checking..."}
          </span>
        </div>
      </div>
    </aside>
  );
}
