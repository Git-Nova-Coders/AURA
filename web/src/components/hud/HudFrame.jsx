import React from 'react';

/**
 * HudFrame — Signature AURA Aerospace Corner-Reticle HUD Frame.
 * Replaces standard SaaS card borders with 4 sharp corner brackets,
 * optional sensor telemetry stamps, active scanline, and technical glow.
 */
export default function HudFrame({
  children,
  className = '',
  sensorId = null,
  status = null,
  coords = null,
  showTelemetryTop = false,
  showTelemetryBottom = false,
  showScanline = false,
  accentColor = 'var(--aura-cyan)',
}) {
  return (
    <div className={`aura-hud-frame ${className}`}>
      {/* ── 4 Signature Corner Reticles ── */}
      <span className="reticle-corner reticle-top-left" style={{ borderColor: accentColor }} />
      <span className="reticle-corner reticle-top-right" style={{ borderColor: accentColor }} />
      <span className="reticle-corner reticle-bottom-left" style={{ borderColor: accentColor }} />
      <span className="reticle-corner reticle-bottom-right" style={{ borderColor: accentColor }} />

      {/* ── Optional Micro Technical Header Stamp (only if child has no header) ── */}
      {showTelemetryTop && sensorId && (
        <div className="hud-frame-telemetry-top">
          <span className="hud-stamp-sensor">
            <span className="stamp-led" style={{ background: accentColor }} />
            {sensorId}
          </span>
          {coords && <span className="hud-stamp-coords">{coords}</span>}
          {status && <span className="hud-stamp-status">{status}</span>}
        </div>
      )}

      {/* ── Optional Ambient Scanline ── */}
      {showScanline && <div className="hud-ambient-scanline" />}

      {/* ── Frame Body Content ── */}
      <div className="hud-frame-content">
        {children}
      </div>

      {/* ── Optional Micro Technical Footer Stamp ── */}
      {showTelemetryBottom && (
        <div className="hud-frame-telemetry-bottom">
          <span className="hud-stamp-code">SYS::AURA_V2</span>
          <div className="hud-stamp-dashes">
            <span /><span /><span /><span />
          </div>
        </div>
      )}
    </div>
  );
}
