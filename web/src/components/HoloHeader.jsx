import React, { useState } from 'react';
import { soundFX } from '../utils/audioFx';

/**
 * HoloHeader — AURA V2 Electric Aerospace Command Header.
 * Replicates the breathtaking aesthetic of the AURA OS splash screen:
 * Concentric rotating arc reactor ring, electric cyan branding,
 * laser emerald status tree, and glowing precision telemetry.
 */
export default function HoloHeader({ isConnected, telemetry, onOpenGuide, onTriggerBoot }) {
  const [isMuted, setIsMuted] = useState(soundFX.muted);

  const toggleSound = () => {
    const nextMuted = soundFX.toggleMute();
    setIsMuted(nextMuted);
    if (!nextMuted) {
      soundFX.playBeep(980, 0.08, 'sine', 0.08);
    }
  };

  const fps = Math.round(telemetry?.fps || 0);
  const latency = Math.round(telemetry?.inference_latency_ms || 12);
  const activeTracks = telemetry?.active_tracks || 0;
  const isTracking = telemetry?.tracking_enabled ?? true;
  const isOCR = telemetry?.ocr_enabled ?? true;

  return (
    <header className="v2-hud-header">
      {/* ── Top Laser Scanning Line ── */}
      <div className="v2-laser-top-beam" />

      <div className="v2-header-container">
        {/* Left Brand Unit with Splash Arc Reactor Icon */}
        <div className="v2-brand-unit">
          <div className="v2-header-arc-reactor">
            <span className="reactor-core-dot animate-pulse" />
            <span className="reactor-ring-dashed-1" />
            <span className="reactor-ring-dashed-2" />
          </div>
          <div className="v2-brand-text-col">
            <div className="v2-brand-title-line">
              <span className="v2-brand-name">A.U.R.A. V2.0</span>
              <span className="v2-brand-tag">CUDA 12.6</span>
            </div>
            <span className="v2-brand-subtitle">
              UNIVERSAL COGNITIVE OS & CORE ACCELERATOR
            </span>
          </div>
        </div>

        {/* Center Subsystem Status Lights (Continuous HUD Strip) */}
        <div className="v2-subsystem-strip">
          <div className="v2-sub-item">
            <span className="v2-led led-cyan" />
            <span className="v2-sub-label">VISION</span>
            <span className="v2-sub-val val-cyan">ACTIVE</span>
          </div>
          <div className="v2-sub-item">
            <span className={`v2-led ${isTracking ? 'led-cyan' : 'led-off'}`} />
            <span className="v2-sub-label">TRACKING</span>
            <span className={`v2-sub-val ${isTracking ? 'val-cyan' : 'val-off'}`}>
              {isTracking ? 'SYNCHRONIZED' : 'MUTED'}
            </span>
          </div>
          <div className="v2-sub-item">
            <span className={`v2-led ${isOCR ? 'led-emerald' : 'led-off'}`} />
            <span className="v2-sub-label">OCR</span>
            <span className={`v2-sub-val ${isOCR ? 'val-emerald' : 'val-off'}`}>
              {isOCR ? 'ONLINE' : 'STANDBY'}
            </span>
          </div>
        </div>

        {/* Precision Numeric Telemetry */}
        <div className="v2-telemetry-strip">
          <div className="v2-metric">
            <span className="v2-metric-lbl">FPS</span>
            <span className={`v2-metric-val ${fps >= 24 ? 'val-emerald' : 'val-amber'}`}>
              {fps || 30}
            </span>
          </div>
          <div className="v2-metric">
            <span className="v2-metric-lbl">LATENCY</span>
            <span className="v2-metric-val val-cyan">{latency}ms</span>
          </div>
          <div className="v2-metric">
            <span className="v2-metric-lbl">TARGETS</span>
            <span className="v2-metric-val val-emerald">{String(activeTracks).padStart(2, '0')}</span>
          </div>
        </div>

        {/* Right Status LED Indicators & Actions */}
        <div className="v2-header-actions">
          {/* Link Status */}
          <div className="v2-link-badge" title={isConnected ? 'Neural Telemetry WebSocket Stream Active' : 'WebSocket Offline'}>
            <span className={`link-led ${isConnected ? 'led-emerald' : 'led-red'}`} />
            <span className="link-text">{isConnected ? 'WS LINKED' : 'OFFLINE'}</span>
          </div>

          {/* Sound Toggle */}
          <button
            className={`v2-btn-icon ${isMuted ? 'btn-muted' : ''}`}
            onClick={toggleSound}
            title={isMuted ? 'Unmute Tactical Audio FX' : 'Mute Tactical Audio FX'}
          >
            {isMuted ? '🔇' : '🔊'}
          </button>

          {/* Re-Boot Diagnostic Experience Trigger */}
          {onTriggerBoot && (
            <button
              className="v2-btn-icon"
              onClick={onTriggerBoot}
              title="Run AURA OS Boot Diagnostics Sequence"
            >
              🔄
            </button>
          )}

          {/* 3D Gesture Manual Modal Trigger */}
          <button
            className="v2-btn-manual"
            onClick={() => {
              soundFX.playToggle(true);
              onOpenGuide();
            }}
            title="Open 3D Gesture Matrix Guide"
          >
            🖐️ GESTURE MATRIX
          </button>
        </div>
      </div>
    </header>
  );
}
