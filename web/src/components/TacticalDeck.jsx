import React from 'react';
import { soundFX } from '../utils/audioFx';

/**
 * TacticalDeck — Bottom deck tactical control matrix with glowing LED switches
 * for SAHI, Object Tracker, OCR Scanner, Voice Assistant, and Quick Gesture Overrides.
 */
export default function TacticalDeck({
  telemetry,
  onToggleSAHI,
  onToggleTracking,
  onToggleOCR,
  onToggleVoice,
}) {
  const isSAHI = telemetry?.sahi_enabled ?? false;
  const isTracking = telemetry?.tracking_enabled ?? true;
  const isOCR = telemetry?.ocr_enabled ?? true;
  const isVoice = telemetry?.voice_listening || telemetry?.voice_status === 'LISTENING';

  const handleSAHI = () => {
    soundFX.playToggle(!isSAHI);
    if (onToggleSAHI) onToggleSAHI();
  };

  const handleTracking = () => {
    soundFX.playToggle(!isTracking);
    if (onToggleTracking) onToggleTracking();
  };

  const handleOCR = () => {
    soundFX.playToggle(!isOCR);
    if (onToggleOCR) onToggleOCR();
  };

  const handleVoice = () => {
    soundFX.playToggle(!isVoice);
    if (onToggleVoice) onToggleVoice();
  };

  return (
    <div className="tactical-deck glass-panel">
      <div className="deck-wrapper">
        {/* ── Left Status & Core Metrics ── */}
        <div className="deck-core-telemetry">
          <div className="deck-telemetry-item">
            <span className="deck-lbl">SYSTEM STATUS</span>
            <span className="deck-val val-emerald">ONLINE // NOMINAL</span>
          </div>
          <div className="deck-telemetry-item">
            <span className="deck-lbl">AI CORE</span>
            <span className="deck-val val-cyan">YOLO-World + MediaPipe 3D</span>
          </div>
          <div className="deck-telemetry-item">
            <span className="deck-lbl">TRACKS</span>
            <span className="deck-val">{telemetry?.active_tracks || 0} ACTIVE</span>
          </div>
        </div>

        <div className="deck-vertical-divider" />

        {/* ── Right Interactive Tactical Switches ── */}
        <div className="deck-switches-matrix">
          {/* SAHI Switch */}
          <div
            className={`tactical-switch-card ${isSAHI ? 'switch-card-active' : ''}`}
            onClick={handleSAHI}
            title="Toggle SAHI Sliced High-Res Inference (Gesture: 🤘 Rock On)"
          >
            <div className="switch-top">
              <span className="switch-icon">🤘</span>
              <span className={`switch-led ${isSAHI ? 'led-active-emerald' : ''}`} />
            </div>
            <span className="switch-title">SAHI MATRIX</span>
            <span className="switch-status-text">{isSAHI ? 'ENABLED' : 'DISABLED'}</span>
          </div>

          {/* Tracking Switch */}
          <div
            className={`tactical-switch-card ${isTracking ? 'switch-card-active' : ''}`}
            onClick={handleTracking}
            title="Toggle Spatial Object Tracking"
          >
            <div className="switch-top">
              <span className="switch-icon">🎯</span>
              <span className={`switch-led ${isTracking ? 'led-active-cyan' : ''}`} />
            </div>
            <span className="switch-title">SPATIAL TRACKER</span>
            <span className="switch-status-text">{isTracking ? 'TRACKING' : 'OFF'}</span>
          </div>

          {/* OCR Scanner Switch */}
          <div
            className={`tactical-switch-card ${isOCR ? 'switch-card-active' : ''}`}
            onClick={handleOCR}
            title="Toggle Asynchronous Text Scanner"
          >
            <div className="switch-top">
              <span className="switch-icon">🔍</span>
              <span className={`switch-led ${isOCR ? 'led-active-cyan' : ''}`} />
            </div>
            <span className="switch-title">OCR SCANNER</span>
            <span className="switch-status-text">{isOCR ? 'ACTIVE' : 'MUTED'}</span>
          </div>

          {/* Voice Assistant Switch */}
          <div
            className={`tactical-switch-card ${isVoice ? 'switch-card-active' : ''}`}
            onClick={handleVoice}
            title="Toggle Voice Assistant (Gesture: 🤙 Call Me)"
          >
            <div className="switch-top">
              <span className="switch-icon">🤙</span>
              <span className={`switch-led ${isVoice ? 'led-active-emerald' : ''}`} />
            </div>
            <span className="switch-title">NEURAL VOICE</span>
            <span className="switch-status-text">{isVoice ? 'LISTENING' : 'STANDBY'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
