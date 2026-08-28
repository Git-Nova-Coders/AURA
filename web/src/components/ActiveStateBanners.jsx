import React from 'react';

/**
 * ActiveStateBanners — Renders instant cybernetic visual overlays on the Live Feed
 * for real-time hand gestures, state toggles (SAHI, Voice, OCR, Freeze), and toasts.
 */
export default function ActiveStateBanners({ telemetry, activeToast }) {
  if (!telemetry && !activeToast) return null;

  const isSAHI = telemetry?.sahi_enabled;
  const isVoiceActive = telemetry?.voice_listening || telemetry?.voice_status === 'LISTENING';
  const isFrozen = telemetry?.gesture_mode === 'FROZEN';
  const isCleanView = telemetry?.gesture_mode === 'HIDE_BOXES';
  const pointedTarget = telemetry?.pointed_target;
  const isTargeting = telemetry?.gesture_mode === 'FOCUS_OBJECT' || telemetry?.gesture_mode === 'INSPECT_OBJECT';

  return (
    <>
      {/* ── Global Animated Action Toast Banner ── */}
      {activeToast && (
        <div className="action-toast-banner" id="hud-action-toast">
          <span className="toast-icon">⚡</span>
          <span className="toast-text">{activeToast}</span>
        </div>
      )}

      {/* ── Top-Left Floating State Badges ── */}
      <div className="feed-status-stack">
        {/* SAHI High-Res Active Badge */}
        {isSAHI && (
          <div className="status-pill status-pill-sahi animate-slide-in">
            <span className="pill-dot emerald-pulse"></span>
            <span className="pill-icon">🤘</span>
            <span className="pill-label">SAHI HIGH-RES (320px Slices)</span>
          </div>
        )}

        {/* Voice Assistant Listening Waveform */}
        {isVoiceActive && (
          <div className="status-pill status-pill-voice animate-slide-in">
            <div className="audio-visualizer">
              <div className="audio-bar"></div>
              <div className="audio-bar"></div>
              <div className="audio-bar"></div>
              <div className="audio-bar"></div>
              <div className="audio-bar"></div>
            </div>
            <span className="pill-label">VOICE LISTENING...</span>
          </div>
        )}

        {/* Clean Camera View Active */}
        {isCleanView && (
          <div className="status-pill status-pill-clean animate-slide-in">
            <span className="pill-icon">🖐️</span>
            <span className="pill-label">CLEAN VIEW (BOXES HIDDEN)</span>
          </div>
        )}

        {/* Overlay Frozen */}
        {isFrozen && (
          <div className="status-pill status-pill-frozen animate-slide-in">
            <span className="pill-icon">✊</span>
            <span className="pill-label">OVERLAY FROZEN</span>
          </div>
        )}

        {/* Active Laser Target Lock */}
        {isTargeting && pointedTarget && (
          <div className="status-pill status-pill-target animate-slide-in">
            <span className="pill-dot cyan-pulse"></span>
            <span className="pill-icon">👉</span>
            <span className="pill-label">TARGET LOCKED: {pointedTarget.toUpperCase()}</span>
          </div>
        )}
      </div>
    </>
  );
}
