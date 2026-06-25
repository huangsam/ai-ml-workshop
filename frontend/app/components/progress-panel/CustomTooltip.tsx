"use client";

interface CustomTooltipProps {
  active?: boolean;
  payload?: ReadonlyArray<{
    name?: string | number | symbol;
    value?: unknown;
    color?: string;
    stroke?: string;
  }>;
  label?: unknown;
}

export default function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (active && payload && payload.length) {
    return (
      <div className="glass-tooltip rounded-lg px-3 py-2 text-xs space-y-1">
        <p className="text-gray-400 font-medium mb-1 border-b border-white/5 pb-1">
          {label !== undefined ? `Step/Epoch: ${String(label)}` : "Metrics"}
        </p>
        {payload.map((entry, index) => (
          <div key={entry.name?.toString() ?? index} className="flex items-center gap-2">
            <span
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: entry.stroke || entry.color }}
            />
            <span className="text-gray-300 font-medium text-left truncate max-w-[180px]">
              {entry.name?.toString()}:
            </span>
            <span className="text-white font-semibold ml-auto">
              {typeof entry.value === "number" ? entry.value.toFixed(4) : String(entry.value ?? "")}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
}
