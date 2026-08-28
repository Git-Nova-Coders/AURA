import React, { useState } from 'react';
import OscilloscopeCanvas from './OscilloscopeCanvas';
import { soundFX } from '../utils/audioFx';

/**
 * HoloHeader — Cinematic Aerospace HUD command bar with live oscilloscope,
 * GPU acceleration matrix, active gesture mode badge, and interactive audio toggle.
 */
export default function HoloHeader({ isConnected, telemetry, onOpenGuide }) {
  const [isMuted, setIsMuted] = useState(soundFX.muted);

  const toggleSound = () => {
    const nextMuted = soundFX.toggleMute();
    setIsMuted(nextMuted);
    if (!nextMuted) {
      soundFX.playBeep(980, 0.08, 'sine', 0.08);
    }
  };

  const fps = telemetry?.fps || 0;
  const latency = telemetry?.inference_latency_ms || 12.0;
  const gestureMode = telemetry?.gesture_mode || 'ALL_OBJECTS';

  return (
    <header className="holo-header glass-panel">
      {/* ── Top Laser Line ── */}
      <div className="holo-laser-top" />

      <div className="holo-header-row">
        {/* ── Brand & Architecture ── */}
        <div className="holo-brand-group">
          <div className="holo-reactor-icon">
            <div className="reactor-core"></div>
            <div className="reactor-ring"></div>
          </div>
          <div className="holo-brand-meta">
            <div className="holo-brand-title-wrap">
              <span className="holo-brand-title">AURA</span>
              <span className="holo-badge-matrix">NEURAL HUD v1.0</span>
            </div>
            <span className="holo-brand-desc">Adaptive Understanding & Reasoning Architecture</span>
          </div>
        </div>

        {/* ── Aerospace Telemetry Oscilloscope Matrix ── */}
        <div className="holo-telemetry-matrix">
          {/* GPU Hardware Card */}
          <div className="holo-chip holo-chip-gpu">
            <span className="chip-icon">⚡</span>
            <div className="chip-text">
              <span className="chip-header">GPU ACCELERATOR</span>
              <span className="chip-sub">RTX 3050 Ti · CUDA 12.6</span>
            </div>
          </div>

          {/* Live Latency Oscilloscope */}
          <div className="holo-chip holo-chip-latency">
            <div className="chip-text">
              <span className="chip-header">INFERENCE LATENCY</span>
              <span className="chip-val chip-val-cyan">{latency.toFixed(1)} ms</span>
            </div>
            <OscilloscopeCanvas value={latency} color="#00f0ff" />
          </div>

          {/* FPS Gauge */}
          <div className="holo-chip holo-chip-fps">
            <div className="chip-text">
              <span className="chip-header">PERCEPTION FPS</span>
              <span className={`chip-val ${fps >= 25 ? 'chip-val-emerald' : 'chip-val-amber'}`}>
                {Math.round(fps)} FPS
              </span>
            </div>
          </div>

          {/* Gesture Mode Badge */}
          <div className="holo-chip holo-chip-mode">
            <span className="mode-led cyan-pulse"></span>
            <span className="mode-text">MODE: {gestureMode}</span>
          </div>
        </div>

        {/* ── Right Actions ── */}
        <div className="holo-actions-group">
          {/* Audio FX Toggle */}
          <button
            className={`btn-holo-icon ${isMuted ? 'btn-holo-muted' : ''}`}
            onClick={toggleSound}
            title={isMuted ? 'Unmute Tactical Audio FX' : 'Mute Tactical Audio FX'}
          >
            {isMuted ? '🔇' : '🔊'}
          </button>

          {/* Holo Gesture Guide */}
          <button
            className="btn-holo-guide"
            onClick={() => {
              soundFX.playToggle(true);
              onOpenGuide();
            }}
            id="btn-open-holo-guide"
          >
            <span className="guide-icon">🖐️</span>
            <span>GESTURE MANUAL</span>
          </button>

          {/* Connection Status */}
          <div className="holo-connection-badge">
            <div className={`conn-dot ${isConnected ? 'conn-live' : 'conn-offline'}`} />
            <span>{isConnected ? 'ONLINE' : 'LINK LOST'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
