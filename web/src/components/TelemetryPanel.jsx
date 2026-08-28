import React from 'react';

/**
 * ControlDeck (TelemetryPanel) — Cybernetic control deck with real-time performance gauges,
 * interactive system toggle switches (SAHI, Tracking, OCR, Voice), and pipeline status indicators.
 */
export default function TelemetryPanel({
  telemetry,
  onToggleSAHI,
  onToggleTracking,
  onToggleOCR,
  onToggleVoice,
}) {
  if (!telemetry) {
    return (
      <div className="control-deck glass-card">
        <div className="deck-loading">
          <span className="loading-text">Synchronizing AURA Telemetry Stream...</span>
          <div className="skeleton" style={{ width: '240px', height: '8px' }} />
        </div>
      </div>
    );
  }

  const fps = telemetry.fps || 0;
  const latency = telemetry.inference_latency_ms || 0;
  const isSAHI = telemetry.sahi_enabled ?? false;
  const isTracking = telemetry.tracking_enabled ?? true;
  const isOCR = telemetry.ocr_enabled ?? true;
  const isVoice = telemetry.voice_listening || telemetry.voice_status === 'LISTENING';

  return (
    <div className="control-deck glass-card">
      <div className="deck-content">
        {/* ── Left Metrics Group ── */}
        <div className="deck-metrics-group">
          {/* Radial FPS Meter */}
          <div className="deck-metric-gauge">
            <svg width="42" height="42" viewBox="0 0 42 42">
              <circle cx="21" cy="21" r="17" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
              <circle
                cx="21" cy="21" r="17" fill="none"
                stroke={fps >= 25 ? 'var(--accent-emerald)' : fps >= 15 ? 'var(--accent-amber)' : 'var(--accent-red)'}
                strokeWidth="3.2"
                strokeDasharray={`${Math.min(100, (fps / 60) * 107)} 107`}
                strokeLinecap="round"
                transform="rotate(-90 21 21)"
                style={{
                  filter: `drop-shadow(0 0 6px ${fps >= 25 ? 'var(--accent-emerald)' : 'var(--accent-amber)'})`,
                  transition: 'stroke-dasharray 0.4s ease',
                }}
              />
              <text x="21" y="25" textAnchor="middle" fill="#fff" style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', fontWeight: 700 }}>
                {Math.round(fps)}
              </text>
            </svg>
            <div className="metric-meta">
              <span className="metric-title">FPS</span>
              <span className="metric-subtitle">TARGET 60</span>
            </div>
          </div>

          {/* Inference Latency */}
          <div className="deck-metric-card">
            <span className="metric-val metric-val-cyan">{latency.toFixed(1)} <small>ms</small></span>
            <span className="metric-title">GPU LATENCY</span>
          </div>

          {/* Detections & Tracks */}
          <div className="deck-metric-card">
            <span className="metric-val metric-val-emerald">{telemetry.detection_count}</span>
            <span className="metric-title">OBJECTS</span>
          </div>

          <div className="deck-metric-card">
            <span className="metric-val metric-val-violet">{telemetry.active_tracks}</span>
            <span className="metric-title">TRACKS</span>
          </div>

          <div className="deck-metric-card">
            <span className="metric-val metric-val-amber">{telemetry.ocr_text_count}</span>
            <span className="metric-title">OCR TEXTS</span>
          </div>
        </div>

        {/* ── Center Divider ── */}
        <div className="deck-divider" />

        {/* ── Right Interactive Switches ── */}
        <div className="deck-switches-group">
          {/* SAHI Switch */}
          <label className="cyber-switch" title="Toggle SAHI Sliced High-Resolution Inference (🤘 Rock On)">
            <input type="checkbox" checked={isSAHI} onChange={onToggleSAHI} id="switch-sahi" />
            <span className="cyber-switch-slider" />
            <div className="switch-info">
              <span className="switch-name">🤘 SAHI High-Res</span>
              <span className="switch-status">{isSAHI ? 'ENABLED' : 'OFF'}</span>
            </div>
          </label>

          {/* Tracking Switch */}
          <label className="cyber-switch" title="Toggle Spatial Object Tracking">
            <input type="checkbox" checked={isTracking} onChange={onToggleTracking} id="switch-tracking" />
            <span className="cyber-switch-slider" />
            <div className="switch-info">
              <span className="switch-name">🎯 Tracker</span>
              <span className="switch-status">{isTracking ? 'ENABLED' : 'OFF'}</span>
            </div>
          </label>

          {/* OCR Switch */}
          <label className="cyber-switch" title="Toggle Asynchronous OCR Text Scanner">
            <input type="checkbox" checked={isOCR} onChange={onToggleOCR} id="switch-ocr" />
            <span className="cyber-switch-slider" />
            <div className="switch-info">
              <span className="switch-name">🔍 OCR Scanner</span>
              <span className="switch-status">{isOCR ? 'ACTIVE' : 'OFF'}</span>
            </div>
          </label>

          {/* Voice Switch */}
          <label className="cyber-switch" title="Toggle Voice Assistant Speech Querying (🤙 Call Me)">
            <input type="checkbox" checked={isVoice} onChange={onToggleVoice} id="switch-voice" />
            <span className="cyber-switch-slider" />
            <div className="switch-info">
              <span className="switch-name">🤙 Voice Assistant</span>
              <span className="switch-status">{isVoice ? 'LISTENING' : 'PAUSED'}</span>
            </div>
          </label>
        </div>
      </div>
    </div>
  );
}
