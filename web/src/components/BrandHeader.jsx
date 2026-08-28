import { useState, useEffect } from 'react';

/**
 * AURA Brand Header — Futuristic Cybernetic HUD status bar
 * with GPU hardware status, FPS/Latency telemetry, gesture mode indicator, and guide trigger.
 */
export default function BrandHeader({ isConnected, telemetry, onOpenGuide }) {
  const [glowPhase, setGlowPhase] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => setGlowPhase((p) => (p + 1) % 360), 50);
    return () => clearInterval(interval);
  }, []);

  const glowColor = `hsl(${180 + Math.sin(glowPhase * 0.02) * 20}, 100%, 60%)`;
  const gestureMode = telemetry?.gesture_mode || 'ALL_OBJECTS';
  const fps = telemetry?.fps || 0;
  const latency = telemetry?.inference_latency_ms || 0;

  return (
    <header className="brand-header glass-card">
      {/* ── Top Neon Gradient Accent ── */}
      <div className="header-top-border" />

      <div className="header-content">
        {/* ── Brand Logo & Architecture Subtitle ── */}
        <div className="logo-group">
          <div className="logo-core" style={{ textShadow: `0 0 24px ${glowColor}` }}>
            <span className="logo-symbol">⬢</span>
          </div>
          <div>
            <h1 className="brand-title">
              <span style={{ color: glowColor }}>AURA</span>
              <span className="brand-version-tag">CYBERNETIC 3D</span>
            </h1>
            <p className="brand-subtitle">Adaptive Understanding & Reasoning Architecture</p>
          </div>
        </div>

        {/* ── Center Telemetry Gauges ── */}
        <div className="header-telemetry-center">
          {/* GPU Hardware Acceleration Pill */}
          <div className="hardware-pill glass-card">
            <span className="hw-icon">⚡</span>
            <span className="hw-label">RTX 3050 Ti</span>
            <span className="badge badge-cyan">CUDA 12.6</span>
          </div>

          {/* Latency Gauge */}
          <div className="telemetry-gauge glass-card">
            <span className="gauge-label">LATENCY</span>
            <span className="gauge-val gauge-val-cyan">{latency > 0 ? latency.toFixed(1) : '--'} ms</span>
          </div>

          {/* FPS Gauge */}
          <div className="telemetry-gauge glass-card">
            <span className="gauge-label">FPS</span>
            <span className={`gauge-val ${fps >= 25 ? 'gauge-val-emerald' : fps >= 15 ? 'gauge-val-amber' : 'gauge-val-red'}`}>
              {fps > 0 ? fps.toFixed(1) : '--'}
            </span>
          </div>

          {/* Active Gesture Mode */}
          <div className="gesture-mode-pill glass-card">
            <span className="mode-dot cyan-pulse"></span>
            <span className="mode-label">{gestureMode}</span>
          </div>
        </div>

        {/* ── Right Actions & Connection Status ── */}
        <div className="header-actions-right">
          {onOpenGuide && (
            <button
              className="btn-header-guide glass-card"
              onClick={onOpenGuide}
              title="Open 3D Gesture Guide"
              id="btn-header-gesture-guide"
            >
              <span>🖐️</span>
              <span>Guide</span>
            </button>
          )}

          <div className="connection-status-pill glass-card">
            <div className={`status-dot ${isConnected ? 'dot-live' : 'dot-offline'}`} />
            <span className="status-text">{isConnected ? 'LIVE FEED' : 'RECONNECTING'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
