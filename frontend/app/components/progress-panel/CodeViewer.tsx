"use client";

import { useEffect, useState } from "react";
import { Copy, Check, FileCode } from "lucide-react";
import { fetchTaskCode, Task } from "../../api";
import { highlightPython } from "../../utils/highlighter";

interface CodeViewerProps {
  task: Task;
}

export default function CodeViewer({ task }: CodeViewerProps) {
  const [prevTask, setPrevTask] = useState<Task>(task);
  const [code, setCode] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  // Sync state with props during render to avoid useEffect warning
  if (task !== prevTask) {
    setPrevTask(task);
    setLoading(true);
    setError(null);
  }

  useEffect(() => {
    let active = true;

    fetchTaskCode(task.module, task.task)
      .then((srcCode) => {
        if (!active) return;
        setCode(srcCode);
        setLoading(false);
      })
      .catch((err) => {
        if (!active) return;
        console.error(err);
        setError("Failed to load source code for this task.");
        setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [task]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy code:", err);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-gray-400 space-y-3 bg-white/[0.01] rounded-xl border border-white/5">
        <div className="relative w-8 h-8">
          <div className="absolute inset-0 border-2 border-indigo-500/20 rounded-full" />
          <div className="absolute inset-0 border-t-2 border-indigo-500 rounded-full animate-spin" />
        </div>
        <p className="text-xs animate-pulse text-gray-400">Loading task code...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-red-400 bg-red-950/5 border border-red-900/30 rounded-xl p-6">
        <p className="text-xs font-semibold">{error}</p>
      </div>
    );
  }

  const codeLines = code.split("\n");
  const highlightedHtml = highlightPython(code);

  return (
    <div className="flex flex-col h-[340px] rounded-xl border border-white/5 bg-black/40 overflow-hidden shadow-2xl animate-fade-in group">
      {/* Code Header Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-black/60 border-b border-white/5 select-none">
        <div className="flex items-center gap-2 text-gray-400 text-xs font-semibold">
          <FileCode className="w-3.5 h-3.5 text-indigo-400" />
          <span>
            workshop/core/{task.module}/
            {task.module === "numpy" && task.task === "fundamentals"
              ? "main.py"
              : `${task.task}.py`}
          </span>
        </div>

        <button
          onClick={handleCopy}
          className="p-1 rounded bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-colors cursor-pointer border border-white/5 flex items-center gap-1 text-[10px] font-semibold"
          title="Copy file code to clipboard"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-green-400" />
              <span className="text-green-400">Copied</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span>Copy Code</span>
            </>
          )}
        </button>
      </div>

      {/* Code Area */}
      <div className="flex-1 flex overflow-auto bg-[#050505]/40 text-xs custom-scrollbar">
        {/* Line Numbers Column */}
        <div className="sticky left-0 z-10 py-4 select-none text-right pr-3 pl-4 border-r border-white/5 bg-[#0d0d0f] font-mono text-[10px] text-gray-600 w-12 text-balance leading-normal">
          {codeLines.map((_, i) => (
            <div key={i} className="h-5">
              {i + 1}
            </div>
          ))}
        </div>

        {/* Code Content Column */}
        <pre className="p-4 font-mono text-[11.5px] leading-normal select-text text-gray-300">
          <code
            dangerouslySetInnerHTML={{ __html: highlightedHtml }}
            className="block whitespace-pre text-left leading-5"
          />
        </pre>
      </div>
    </div>
  );
}
