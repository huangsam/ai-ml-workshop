"use client";

import { useEffect, useState } from "react";
import { Task, fetchTaskSchema } from "../api";

interface ConfigFormProps {
  task: Task;
  disabled: boolean;
  onSubmit: (config: Record<string, number>) => void;
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

export default function ConfigForm({ task, disabled, onSubmit }: ConfigFormProps) {
  const [schema, setSchema] = useState<Record<string, SchemaProperty> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchTaskSchema(task.module, task.task)
      .then((data) => {
        if (!active) return;
        const properties = (data.properties as Record<string, SchemaProperty>) ?? {};
        setSchema(properties);
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

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!schema) return;
    const form = e.currentTarget;
    const config: Record<string, number> = {};
    Object.keys(schema).forEach((key) => {
      const el = form.elements.namedItem(key) as HTMLInputElement | null;
      if (el && el.value !== "") {
        config[key] = parseFloat(el.value);
      }
    });
    onSubmit(config);
  }

  if (error) {
    return <p className="text-sm text-red-400 italic">{error}</p>;
  }

  if (!schema) {
    return <p className="text-sm text-gray-400 italic animate-pulse">Loading schema...</p>;
  }

  const fields = Object.entries(schema);

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {fields.length === 0 ? (
        <p className="text-sm text-gray-400 italic">No configurable parameters for this task.</p>
      ) : (
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          {fields.map(([key, prop]) => {
            const isInteger = prop.type === "integer";
            const min = prop.minimum;
            const max = prop.maximum;
            const hasRange = min !== undefined && max !== undefined;

            return (
              <div key={key} className="flex flex-col space-y-2">
                <label className="block text-xs font-medium text-gray-300" htmlFor={key}>
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
                      defaultValue={prop.default !== undefined ? prop.default : (min + max) / 2}
                      disabled={disabled}
                      className="flex-1 range-slider-input"
                      onChange={(e) => {
                        const slider = e.target;
                        const numberInput = document.getElementById(key) as HTMLInputElement | null;
                        if (numberInput) {
                          numberInput.value = slider.value;
                        }
                      }}
                    />
                    <input
                      id={key}
                      name={key}
                      type="number"
                      step={isInteger ? "1" : "any"}
                      min={min}
                      max={max}
                      defaultValue={prop.default !== undefined ? prop.default : (min + max) / 2}
                      disabled={disabled}
                      className="w-24 bg-gray-700/50 text-gray-100 text-sm rounded px-3 py-1.5 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all disabled:opacity-50"
                      onChange={(e) => {
                        const numberInput = e.target;
                        const slider = document.getElementById(
                          `${key}-slider`
                        ) as HTMLInputElement | null;
                        if (slider && numberInput.value !== "") {
                          slider.value = numberInput.value;
                        }
                      }}
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
                    defaultValue={prop.default !== undefined ? prop.default : ""}
                    disabled={disabled}
                    className="w-full bg-gray-700/50 text-gray-100 text-sm rounded px-3 py-2 border border-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all disabled:opacity-50"
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
