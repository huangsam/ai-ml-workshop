"use client";

import { API_BASE } from "../../api";

interface ZoomModalProps {
  isOpen: boolean;
  onClose: () => void;
  jobId: string;
  plotName: string;
}

export default function ZoomModal({ isOpen, onClose, jobId, plotName }: ZoomModalProps) {
  if (!isOpen) return null;

  return (
    <div
      onClick={onClose}
      className="fixed inset-0 bg-black/95 backdrop-blur-md z-[100] flex flex-col items-center justify-center cursor-zoom-out p-6 animate-fade-in"
    >
      <button
        type="button"
        onClick={onClose}
        className="absolute top-6 right-6 p-2 rounded-full bg-white/5 hover:bg-white/10 text-white transition-all cursor-pointer border border-white/10"
      >
        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </button>

      <div className="max-w-5xl max-h-[85vh] flex items-center justify-center p-2 rounded-xl border border-white/10 bg-[#0f0f10] shadow-2xl">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={`${API_BASE}/plots/${jobId}/${plotName}`}
          alt={plotName}
          className="max-w-full max-h-[80vh] object-contain rounded-lg select-none"
        />
      </div>

      <p className="text-xs text-gray-400 mt-4 uppercase tracking-widest font-semibold">
        {plotName.replace(".png", "").replace(/_/g, " ")}
      </p>
    </div>
  );
}
