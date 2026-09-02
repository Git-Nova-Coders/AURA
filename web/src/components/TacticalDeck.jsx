import React, { useState, useEffect } from 'react';
import { soundFX } from '../utils/audioFx';

/**
 * TacticalDeck — Bottom deck tactical control matrix with glowing LED switches
 * for Perception Filter, 3D Gestures, SAHI, Object Tracker, OCR Scanner, and Voice Assistant.
 */
export default function TacticalDeck({
  telemetry,
  filterMode: propFilterMode,
  onToggleSAHI,
  onToggleTracking,
  onToggleOCR,
  onToggleVoice,
  onToggleGestures,
  onCycleTargetFilter,
  onSetTargetFilter,
}) {
  const isSAHI = telemetry?.sahi_enabled ?? false;
  const isTracking = telemetry?.tracking_enabled ?? true;
  const isOCR = telemetry?.ocr_enabled ?? true;
  const isGestures = telemetry?.gestures_enabled ?? false;
  const isVoice = telemetry?.voice_listening || telemetry?.voice_status === 'LISTENING';

  // Optimistic local state for instantaneous click responsiveness
  const [localFilterMode, setLocalFilterMode] = useState(
    propFilterMode || telemetry?.target_filter_mode || 'ALL'
  );

  // Sync with incoming telemetry / prop updates
  useEffect(() => {
    if (telemetry?.target_filter_mode) {
      setLocalFilterMode(telemetry.target_filter_mode);
    }
  }, [telemetry?.target_filter_mode]);

  useEffect(() => {
    if (propFilterMode) {
      setLocalFilterMode(propFilterMode);
    }
  }, [propFilterMode]);

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

  const handleGestures = () => {
    soundFX.playToggle(!isGestures);
    if (onToggleGestures) onToggleGestures();
  };

  const handleFilter = () => {
    soundFX.playClick();
    const cycleOrder = ['ALL', 'OBJECTS_ONLY', 'HUMANS_ONLY', 'OFF'];
    const currentIdx = cycleOrder.indexOf(localFilterMode);
    const nextMode = cycleOrder[(currentIdx + 1) % cycleOrder.length];
    
    // Immediate optimistic local update
    setLocalFilterMode(nextMode);

    // Dispatch to backend
    if (onSetTargetFilter) {
      onSetTargetFilter(nextMode);
    } else if (onCycleTargetFilter) {
      onCycleTargetFilter();
    }
  };

  const getFilterIcon = () => {
    if (localFilterMode === 'OBJECTS_ONLY') return '📦';
    if (localFilterMode === 'HUMANS_ONLY') return '👤';
    if (localFilterMode === 'OFF') return '🛑';
    return '🌐';
  };

  const getFilterLabel = () => {
    if (localFilterMode === 'OBJECTS_ONLY') return 'OBJECTS ONLY';
    if (localFilterMode === 'HUMANS_ONLY') return 'HUMANS ONLY';
    if (localFilterMode === 'OFF') return 'MUTED (OFF)';
    return 'OMNI VIEW';
  };

  const isPerceptionActive = localFilterMode !== 'OFF';

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
          {/* Perception Filter Switch (4 Functions: Omni View -> Objects Only -> Humans Only -> Muted Off) */}
          <div
            className={`tactical-switch-card ${isPerceptionActive ? 'switch-card-active' : 'switch-card-off'}`}
            onClick={handleFilter}
            title="Cycle Perception: Omni View ➔ Objects Only ➔ Humans Only ➔ Muted (Off)"
          >
            <div className="switch-top">
              <span className="switch-icon">{getFilterIcon()}</span>
              <span className={`switch-led ${isPerceptionActive ? 'led-active-cyan' : ''}`} />
            </div>
            <span className="switch-title">PERCEPTION</span>
            <span className="switch-status-text">{getFilterLabel()}</span>
          </div>

          {/* 3D Gestures Master Armed Switch */}
          <div
            className={`tactical-switch-card ${isGestures ? 'switch-card-active' : ''}`}
            onClick={handleGestures}
            title="Arm / Disarm 3D Hand Gestures (Raycast, Air-Click Pinch & Skeleton)"
          >
            <div className="switch-top">
              <span className="switch-icon">🖐️</span>
              <span className={`switch-led ${isGestures ? 'led-active-emerald' : ''}`} />
            </div>
            <span className="switch-title">3D GESTURES</span>
            <span className="switch-status-text">{isGestures ? 'ARMED' : 'STANDBY'}</span>
          </div>

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
