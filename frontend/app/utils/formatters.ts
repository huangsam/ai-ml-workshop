/**
 * Utility functions for formatting and presentation
 */

/**
 * Clean up a plot filename (e.g. "loss_curve.png" -> "Loss Curve")
 */
export function formatPlotName(plotName: string): string {
  if (!plotName) return "";
  return plotName
    .replace(".png", "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Format timestamp into standard local time string
 */
export function formatRunTime(createdAt: number): string {
  return new Date(createdAt * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/**
 * Format configuration options into a human-readable list
 */
export function formatRunConfig(config: Record<string, number | string | boolean>): string {
  if (!config || Object.keys(config).length === 0) return "";
  return Object.entries(config)
    .map(([k, v]) => `${k}: ${v}`)
    .join(", ");
}

/**
 * Format configuration options in a compact style (e.g. "lr=0.01, epochs=10")
 */
export function formatRunConfigCompact(config: Record<string, number | string | boolean>): string {
  if (!config || Object.keys(config).length === 0) return "";
  return Object.entries(config)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");
}
