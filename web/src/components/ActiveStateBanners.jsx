import React from 'react';

/**
 * ActiveStateBanners — Renders instant cybernetic visual overlays on the Live Feed
 * for real-time hand gestures, state toggles (SAHI, Voice, OCR, Freeze), and toasts.
 */
export default function ActiveStateBanners({ telemetry, activeToast }) {
  if (!telemetry && !activeToast) return null;

  const isGestures = Boolean(telemetry?.gestures_enabled);
  const isSAHI = telemetry?.sahi_enabled;
  const isVoiceActive = telemetry?.voice_listening || telemetry?.voice_status === 'LISTENING';
  const isFrozen = isGestures && telemetry?.gesture_mode === 'FROZEN';
  const isCleanView = isGestures && telemetry?.gesture_mode === 'HIDE_BOXES';
  const pointedTarget = isGestures ? telemetry?.pointed_target : null;
  const isTargeting = isGestures && (telemetry?.gesture_mode === 'FOCUS_OBJECT' || telemetry?.gesture_mode === 'INSPECT_OBJECT');

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

        {/* HAR Active Motion Pill */}
        {telemetry?.har_enabled && telemetry?.har_activity && (
          <div className="status-pill status-pill-har animate-slide-in">
            <span className="pill-dot emerald-pulse"></span>
            <span className="pill-icon">🏃</span>
            <span className="pill-label">HAR: {telemetry.har_activity.toUpperCase()} ({(telemetry.har_confidence * 100).toFixed(0)}%)</span>
          </div>
        )}

        {/* HAD Anomaly Monitor Pill */}
        {telemetry?.had_enabled && telemetry?.had_status && (
          <div className={`status-pill ${telemetry.had_status.includes('ANOMALY') ? 'status-pill-anomaly animate-pulse' : 'status-pill-had'} animate-slide-in`}>
            <span className={`pill-dot ${telemetry.had_status.includes('ANOMALY') ? 'rose-pulse' : 'emerald-pulse'}`}></span>
            <span className="pill-icon">{telemetry.had_status.includes('ANOMALY') ? '🚨' : '🛡️'}</span>
            <span className="pill-label">HAD: {telemetry.had_status.toUpperCase()} ({(telemetry.had_confidence * 100).toFixed(0)}%)</span>
          </div>
        )}
      </div>
    </>
  );
}
