"use client";

import { useEffect, useState } from "react";
import { Task, fetchTaskSchema } from "../api";

interface ConfigFormProps {
  task: Task;
  disabled: boolean;
  onSubmit: (config: Record<string, number>) => void;
  initialValues?: Record<string, number> | null;
}

interface SchemaProperty {
  type: string;
  title?: string;
  description?: string;
  default?: number;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
}

export default function ConfigForm({ task, disabled, onSubmit, initialValues }: ConfigFormProps) {
  const [schema, setSchema] = useState<Record<string, SchemaProperty> | null>(null);
  const [configValues, setConfigValues] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);

  const [prevInitialValues, setPrevInitialValues] = useState<
    Record<string, number> | null | undefined
  >(initialValues);

  if (initialValues !== prevInitialValues) {
    setPrevInitialValues(initialValues);
    if (initialValues) {
      setConfigValues(initialValues);
    }
  }

  useEffect(() => {
    let active = true;
    fetchTaskSchema(task.module, task.task)
      .then((data) => {
        if (!active) return;
        const properties = (data.properties as Record<string, SchemaProperty>) ?? {};
        setSchema(properties);

        // Initialize values with defaults
        const initialValues: Record<string, number> = {};
        Object.entries(properties).forEach(([key, prop]) => {
          if (prop.default !== undefined) {
            initialValues[key] = prop.default;
          } else if (prop.minimum !== undefined && prop.maximum !== undefined) {
            initialValues[key] = (prop.minimum + prop.maximum) / 2;
          } else {
            initialValues[key] = 0;
          }
        });
        setConfigValues(initialValues);
      })
      .catch((err) => {
        if (!active) return;
        console.error(err);
        setError("Failed to load configuration schema.");
      });
    return () => {
      active = false;
    };
  }, [task]);

  const handleValueChange = (key: string, val: number) => {
    setConfigValues((prev) => ({ ...prev, [key]: val }));
  };

  const applyPreset = (presetType: "standard" | "quick" | "thorough") => {
    if (!schema) return;
    const newValues: Record<string, number> = {};
    Object.entries(schema).forEach(([key, prop]) => {
      const min = prop.minimum ?? 0;
      const max = prop.maximum ?? 100;
      const def = prop.default !== undefined ? prop.default : (min + max) / 2;
      const isInteger = prop.type === "integer";

      if (presetType === "standard") {
        newValues[key] = def;
      } else {
        const name = key.toLowerCase();
        const isCostly =
          name.includes("epoch") ||
          name.includes("iter") ||
          name.includes("estimator") ||
          name.includes("sample") ||
          name.includes("fold") ||
          name.includes("length");

        if (isCostly) {
          if (presetType === "quick") {
            if (prop.minimum !== undefined) {
              newValues[key] = prop.minimum;
            } else {
              newValues[key] = isInteger ? Math.max(1, Math.round(def * 0.2)) : def * 0.2;
            }
          } else {
            if (prop.maximum !== undefined) {
              newValues[key] = prop.maximum;
            } else {
              newValues[key] = isInteger ? Math.round(def * 2.0) : def * 2.0;
            }
          }
        } else {
          if (name.includes("learning_rate") || name.includes("lr")) {
            newValues[key] = presetType === "quick" ? Math.min(1.0, def * 2) : def;
          } else {
            newValues[key] = def;
          }
        }
      }

      if (isInteger) {
        newValues[key] = Math.round(newValues[key]);
      }
      if (prop.minimum !== undefined) {
        newValues[key] = Math.max(prop.minimum, newValues[key]);
      }
      if (prop.maximum !== undefined) {
        newValues[key] = Math.min(prop.maximum, newValues[key]);
      }
    });

    setConfigValues(newValues);
  };

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    onSubmit(configValues);
  }

  if (error) {
    return <p className="text-sm text-red-400 italic">{error}</p>;
  }

  if (!schema) {
    return <p className="text-sm text-gray-400 italic animate-pulse">Loading schema...</p>;
  }

  const fields = Object.entries(schema);
  const showPresets = fields.length > 0;

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {showPresets && (
        <div className="space-y-2 pb-2.5 border-b border-white/5 mb-4">
          <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest block">
            Execution Presets
          </span>
          <div className="flex gap-2.5">
            {[
              {
                id: "standard",
                label: "Base",
                icon: "⚙️",
                color: "hover:border-blue-500/50 hover:text-blue-300 hover:bg-blue-500/5",
              },
              {
                id: "quick",
                label: "Quick",
                icon: "⚡",
                color: "hover:border-amber-500/50 hover:text-amber-300 hover:bg-amber-500/5",
              },
              {
                id: "thorough",
                label: "Deep",
                icon: "🎯",
                color: "hover:border-emerald-500/50 hover:text-emerald-300 hover:bg-emerald-500/5",
              },
            ].map((preset) => (
              <button
                key={preset.id}
                type="button"
                disabled={disabled}
                onClick={() => applyPreset(preset.id as "standard" | "quick" | "thorough")}
                className={`flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg border border-white/10 bg-white/[0.02] text-xs font-semibold text-gray-300 transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] cursor-pointer ${preset.color} disabled:opacity-50 disabled:pointer-events-none`}
              >
                <span>{preset.icon}</span>
                <span>{preset.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {fields.length === 0 ? (
        <p className="text-sm text-gray-400 italic">No configurable parameters for this task.</p>
      ) : (
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          {fields.map(([key, prop]) => {
            const isInteger = prop.type === "integer";
            const min = prop.minimum;
            const max = prop.maximum;
            const hasRange = min !== undefined && max !== undefined;
            const val =
              configValues[key] !== undefined ? configValues[key] : (prop.default ?? min ?? 0);

            return (
              <div key={key} className="flex flex-col space-y-2">
                <label
                  className="block text-xs font-medium text-gray-300 animate-fade-in"
                  htmlFor={key}
                >
                  {prop.title ?? key}
                </label>

                {hasRange ? (
                  /* Dual-input range slider with bidirectional sync */
                  <div className="range-slider-container">
                    <input
                      id={`${key}-slider`}
                      name={`${key}-slider`}
                      type="range"
                      min={min}
                      max={max}
                      step={isInteger ? "1" : "any"}
                      value={val}
                      disabled={disabled}
                      className="flex-1 range-slider-input"
                      onChange={(e) => handleValueChange(key, parseFloat(e.target.value))}
                    />
                    <input
                      id={key}
                      name={key}
                      type="number"
                      step={isInteger ? "1" : "any"}
                      min={min}
                      max={max}
                      value={val}
                      disabled={disabled}
                      className="w-24 bg-gray-700/50 text-gray-100 text-sm rounded px-3 py-1.5 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all disabled:opacity-50"
                      onChange={(e) => handleValueChange(key, parseFloat(e.target.value) || 0)}
                    />
                  </div>
                ) : (
                  /* Standard number input */
                  <input
                    id={key}
                    name={key}
                    type="number"
                    step={isInteger ? "1" : "any"}
                    min={min !== undefined ? min : undefined}
                    max={max !== undefined ? max : undefined}
                    value={val}
                    disabled={disabled}
                    className="w-full bg-gray-700/50 text-gray-100 text-sm rounded px-3 py-2 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all disabled:opacity-50"
                    onChange={(e) => handleValueChange(key, parseFloat(e.target.value) || 0)}
                  />
                )}

                {prop.description && (
                  <span className="text-[10px] text-gray-400 mt-1 italic leading-relaxed">
                    {prop.description}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
      <button
        type="submit"
        disabled={disabled}
        className={`mt-3 w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 text-white text-sm font-medium py-2.5 rounded-lg transition-all duration-300 shadow-lg shadow-indigo-900/20 ${
          disabled
            ? "cursor-wait"
            : "hover:scale-[1.01] hover:shadow-indigo-500/10 active:scale-[0.99]"
        }`}
      >
        {disabled && (
          <span className="inline-flex items-center justify-center gap-2">
            <svg
              className="animate-spin h-4 w-4 text-white"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
            Running...
          </span>
        )}
        {!disabled && "▶  Run Task"}
      </button>
    </form>
  );
}
