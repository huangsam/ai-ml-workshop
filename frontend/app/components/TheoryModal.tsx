"use client";

import { useEffect } from "react";
import { TheoryContent } from "../theory";

interface TheoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  content?: TheoryContent;
}

export default function TheoryModal({ isOpen, onClose, content }: TheoryModalProps) {
  // Listen for Escape key to close modal
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, onClose]);

  // Prevent background scrolling when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  if (!isOpen || !content) return null;

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 bg-black/75 backdrop-blur-md z-[100] flex items-center justify-center p-4 md:p-6 animate-fade-in"
      aria-modal="true"
      role="dialog"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="glass-panel rounded-2xl w-full max-w-3xl max-h-[85vh] flex flex-col shadow-2xl border border-white/10 overflow-hidden transform-gpu transition-all duration-300 scale-100 animate-slide-up"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/5 bg-white/[0.01]">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center shadow-inner">
              <svg
                className="w-4.5 h-4.5 text-indigo-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
                />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight leading-tight">
                {content.title}
              </h2>
              <p className="text-[10px] text-gray-500 font-medium uppercase tracking-wider mt-0.5">
                Theory & Concept Wiki
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-gray-400 hover:text-white transition-colors cursor-pointer border border-white/5"
            aria-label="Close modal"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Scrollable Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar bg-black/10">
          {/* Overview */}
          <section className="space-y-2">
            <h3 className="text-xs uppercase tracking-widest text-indigo-400 font-bold">
              Overview
            </h3>
            <p className="text-gray-300 text-sm leading-relaxed text-balance">{content.overview}</p>
          </section>

          {/* Key Terminology */}
          <section className="space-y-3">
            <h3 className="text-xs uppercase tracking-widest text-purple-400 font-bold">
              Key Concepts
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {content.concepts.map((concept) => (
                <div
                  key={concept.term}
                  className="p-4 rounded-xl bg-white/[0.02] border border-white/5 space-y-1.5 hover:border-white/10 transition-colors"
                >
                  <h4 className="text-sm font-semibold text-white tracking-tight">
                    {concept.term}
                  </h4>
                  <p className="text-xs text-gray-400 leading-relaxed">{concept.definition}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Mathematical Insights */}
          {content.math && (
            <section className="space-y-2.5">
              <h3 className="text-xs uppercase tracking-widest text-pink-400 font-bold">
                Mathematical Insights
              </h3>
              <pre className="p-4 rounded-xl bg-black/45 border border-white/5 text-xs text-gray-300 font-mono overflow-x-auto whitespace-pre leading-relaxed select-all">
                {content.math}
              </pre>
            </section>
          )}

          {/* What to Observe */}
          <section className="space-y-3">
            <h3 className="text-xs uppercase tracking-widest text-emerald-400 font-bold">
              What to Observe
            </h3>
            <ul className="space-y-2.5">
              {content.whatToObserve.map((item, index) => (
                <li key={index} className="flex gap-2.5 text-sm text-gray-300 leading-relaxed">
                  <span className="text-emerald-500 font-bold select-none">•</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </section>
        </div>

        {/* Footer */}
        <div className="p-4 bg-white/[0.01] border-t border-white/5 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-5 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-xs font-bold text-white rounded-xl transition-all duration-300 shadow-md active:scale-97 cursor-pointer hover:shadow-indigo-500/15"
          >
            Got It, Let&apos;s Train!
          </button>
        </div>
      </div>
    </div>
  );
}
