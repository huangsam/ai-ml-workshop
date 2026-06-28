"use client";

import { useEffect, useRef, useState } from "react";
import { Terminal, Copy, Check, Trash2, ArrowDown } from "lucide-react";

interface ConsoleTerminalProps {
  logs: string;
  status?: string;
  onClear?: () => void;
}

export default function ConsoleTerminal({ logs, status, onClear }: ConsoleTerminalProps) {
  const terminalEndRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [copied, setCopied] = useState(false);
  const [userScrolledUp, setUserScrolledUp] = useState(false);

  const [prevStatus, setPrevStatus] = useState<string | undefined>(status);

  // Sync state with props during render to avoid useEffect set-state warning
  if (status !== prevStatus) {
    setPrevStatus(status);
    if (status === "PENDING") {
      setUserScrolledUp(false);
    }
  }

  // Auto-scroll logic (scroll container only, avoids moving the browser window)
  useEffect(() => {
    if (!userScrolledUp && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs, userScrolledUp]);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    // If the user scrolls up by more than 40px from the bottom, lock scrolling
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 40;
    setUserScrolledUp(!isAtBottom);
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(logs);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy logs:", err);
    }
  };

  // Helper to format logs with colorized keywords (Error, Warning, Success checkmarks)
  const formatLogLine = (line: string, index: number) => {
    if (!line) return <div key={index} className="h-4" />;

    let className = "text-gray-300";
    if (
      line.toLowerCase().includes("error") ||
      line.toLowerCase().includes("exception") ||
      line.toLowerCase().includes("failed")
    ) {
      className = "text-red-400 font-semibold";
    } else if (line.toLowerCase().includes("warning")) {
      className = "text-amber-400";
    } else if (
      line.includes("✓") ||
      line.toLowerCase().includes("successfully") ||
      line.toLowerCase().includes("complete")
    ) {
      className = "text-emerald-400 font-medium";
    } else if (
      line.startsWith("Training") ||
      line.startsWith("Evaluating") ||
      line.startsWith("Testing")
    ) {
      className = "text-indigo-300";
    }

    return (
      <div key={index} className={`whitespace-pre-wrap break-all leading-relaxed ${className}`}>
        {line}
      </div>
    );
  };

  // Process carriage returns (\r) to simulate standard terminal overwriting (e.g. for progress bars)
  const logLines = logs.split("\n").map((line) => {
    if (line.includes("\r")) {
      const parts = line.split("\r");
      for (let i = parts.length - 1; i >= 0; i--) {
        if (parts[i]) return parts[i];
      }
      return "";
    }
    return line;
  });

  return (
    <div className="flex flex-col h-[340px] rounded-xl border border-white/5 bg-black/40 overflow-hidden shadow-2xl animate-fade-in group relative">
      {/* Terminal Title Bar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-black/60 border-b border-white/5 select-none">
        <div className="flex items-center gap-6">
          {/* OS Terminal Buttons */}
          <div className="flex gap-1.5">
            <span className="w-3 h-3 rounded-full bg-[#ff5f56] border border-[#e0443e]" />
            <span className="w-3 h-3 rounded-full bg-[#ffbd2e] border border-[#dea123]" />
            <span className="w-3 h-3 rounded-full bg-[#27c93f] border border-[#1aab29]" />
          </div>
          <div className="flex items-center gap-1.5 text-gray-400 text-xs font-semibold">
            <Terminal className="w-3.5 h-3.5 text-indigo-400" />
            <span>bash — stdout logs</span>
          </div>
        </div>

        {/* Toolbar Actions */}
        {logs.trim() && (
          <div className="flex items-center gap-1">
            <button
              onClick={handleCopy}
              className="p-1 rounded bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-colors cursor-pointer border border-white/5 flex items-center gap-1 text-[10px] font-semibold"
              title="Copy output to clipboard"
            >
              {copied ? (
                <>
                  <Check className="w-3 h-3 text-green-400" />
                  <span className="text-green-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy</span>
                </>
              )}
            </button>
            {onClear && (
              <button
                onClick={onClear}
                className="p-1 rounded bg-white/5 hover:bg-white/10 text-gray-400 hover:text-red-400 transition-colors cursor-pointer border border-white/5 flex items-center gap-1 text-[10px] font-semibold"
                title="Clear console"
              >
                <Trash2 className="w-3 h-3" />
                <span>Clear</span>
              </button>
            )}
          </div>
        )}
      </div>

      {/* Terminal Viewport */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 font-mono text-[11px] custom-scrollbar bg-[#050505]/60 relative"
      >
        {logs.trim() ? (
          <div className="space-y-0.5">
            {logLines.map((line, idx) => formatLogLine(line, idx))}
            {/* Blinking cursor if task is active */}
            {status === "RUNNING" && (
              <div className="inline-flex items-center gap-1 mt-1 text-indigo-400">
                <span className="w-1.5 h-3.5 bg-indigo-500 animate-[pulse_1s_infinite]" />
                <span className="text-[10px] italic text-gray-500 font-semibold uppercase tracking-wider ml-1">
                  Task Running...
                </span>
              </div>
            )}
            <div ref={terminalEndRef} />
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-4">
            <div className="w-12 h-12 rounded-full bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 shadow-[0_0_15px_rgba(99,102,241,0.05)]">
              <Terminal className="w-6 h-6 animate-pulse" />
            </div>
            <div className="space-y-1 max-w-xs">
              <h5 className="text-xs font-semibold text-white">Console Terminal Ready</h5>
              <p className="text-[10px] text-gray-400 leading-relaxed">
                Click <strong className="text-indigo-400 font-medium">Run Task</strong> to capture
                and inspect live Python print statements and training diagnostics.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Scroll to bottom floating button wrapper for smooth transition */}
      <div
        className={`absolute bottom-4 right-4 z-10 transition-all duration-300 ${
          userScrolledUp && logs.trim()
            ? "opacity-100 translate-y-0 pointer-events-auto"
            : "opacity-0 translate-y-2 pointer-events-none"
        }`}
      >
        <button
          onClick={(e) => {
            e.stopPropagation();
            setUserScrolledUp(false);
            if (containerRef.current) {
              containerRef.current.scrollTop = containerRef.current.scrollHeight;
            }
          }}
          className="bg-indigo-600 hover:bg-indigo-500 border border-indigo-400/30 text-white p-1.5 rounded-full shadow-lg transition-all duration-300 hover:scale-105 active:scale-95 cursor-pointer flex items-center gap-1 text-[9px] font-bold uppercase tracking-wider pr-2.5 animate-bounce"
        >
          <ArrowDown className="w-3.5 h-3.5" />
          <span>Scroll Down</span>
        </button>
      </div>
    </div>
  );
}
