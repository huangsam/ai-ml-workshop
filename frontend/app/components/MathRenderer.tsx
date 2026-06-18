"use client";

import React, { useMemo } from "react";
import katex from "katex";

interface MathRendererProps {
  text: string;
  className?: string;
}

interface MathSegment {
  type: "text" | "inline" | "block";
  content: string;
  raw: string;
}

/**
 * Splits a text into segments of plain text, inline math ($...$), and block math ($$...$$).
 */
function parseMathText(text: string): MathSegment[] {
  // First, split by `$$` to separate block math
  const blockParts = text.split(/(\$\$[\s\S]*?\$\$)/g);
  const rawSegments: MathSegment[] = [];

  for (const part of blockParts) {
    if (part.startsWith("$$") && part.endsWith("$$")) {
      const math = part.slice(2, -2).trim();
      rawSegments.push({ type: "block", content: math, raw: part });
    } else {
      // For non-block parts, split by `$` to separate inline math
      const inlineParts = part.split(/(\$.*?\$)/g);
      for (const subPart of inlineParts) {
        if (subPart.startsWith("$") && subPart.endsWith("$")) {
          const math = subPart.slice(1, -1).trim();
          rawSegments.push({ type: "inline", content: math, raw: subPart });
        } else if (subPart) {
          rawSegments.push({ type: "text", content: subPart, raw: subPart });
        }
      }
    }
  }

  // Clean up any extra newlines immediately adjacent to block math segments
  const cleanedSegments: MathSegment[] = [];
  for (let i = 0; i < rawSegments.length; i++) {
    const current = rawSegments[i];
    if (current.type === "text") {
      let content = current.content;

      // If preceded by a block math segment, trim all leading newlines
      const precededByBlock = i > 0 && rawSegments[i - 1].type === "block";
      if (precededByBlock) {
        content = content.replace(/^[\r\n]+/, "");
      }

      // If followed by a block math segment, trim all trailing newlines
      const followedByBlock = i < rawSegments.length - 1 && rawSegments[i + 1].type === "block";
      if (followedByBlock) {
        content = content.replace(/[\r\n]+$/, "");
      }

      // Only add non-empty text segments
      if (content !== "") {
        cleanedSegments.push({ ...current, content });
      }
    } else {
      cleanedSegments.push(current);
    }
  }

  return cleanedSegments;
}

export default function MathRenderer({ text, className = "" }: MathRendererProps) {
  const segments = useMemo(() => parseMathText(text), [text]);

  return (
    <div className={`prose-math whitespace-pre-wrap ${className}`}>
      {segments.map((segment, index) => {
        if (segment.type === "text") {
          return <span key={index}>{segment.content}</span>;
        }

        const isBlock = segment.type === "block";
        try {
          const html = katex.renderToString(segment.content, {
            displayMode: isBlock,
            throwOnError: false,
            trust: true,
          });

          if (isBlock) {
            return (
              <div
                key={index}
                className="my-3 overflow-x-auto overflow-y-hidden py-1 text-center"
                dangerouslySetInnerHTML={{ __html: html }}
              />
            );
          } else {
            return (
              <span
                key={index}
                className="inline-block px-0.5 align-middle"
                dangerouslySetInnerHTML={{ __html: html }}
              />
            );
          }
        } catch (error) {
          console.error("KaTeX rendering error for:", segment.content, error);
          return (
            <code key={index} className="text-red-400 bg-red-950/20 px-1 py-0.5 rounded font-mono">
              {segment.raw}
            </code>
          );
        }
      })}
    </div>
  );
}
