"use client";

interface StatusBadgeProps {
  status?: string;
  size?: "sm" | "md";
}

export default function StatusBadge({ status, size = "md" }: StatusBadgeProps) {
  if (!status) return null;

  const sizeClass = size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-xs";

  let colorClass = "bg-gray-800/40 text-gray-300 border border-gray-600/50";
  if (status === "PENDING") {
    colorClass = "bg-yellow-900/40 text-yellow-300 border border-yellow-700/50";
  } else if (status === "RUNNING") {
    colorClass = "bg-blue-900/40 text-blue-300 border border-blue-700/50 pulse-glow";
  } else if (status === "COMPLETED") {
    colorClass = "bg-green-900/40 text-green-300 border border-green-700/50";
  } else if (status === "FAILED") {
    colorClass = "bg-red-900/40 text-red-300 border border-red-700/50";
  }

  return (
    <span className={`font-semibold rounded-full border ${sizeClass} ${colorClass}`}>{status}</span>
  );
}
