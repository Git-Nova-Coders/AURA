import React, { useState, useEffect, useRef } from 'react';
import { soundFX } from '../utils/audioFx';

/**
 * TacticalDeck — AURA V2 Holographic Aerospace Command Dock.
 * Collapsed into a sleek, floating command pill.
 * Pops up into 6 aerospace control pods with radial rings and system LEDs.
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

  const [localFilterMode, setLocalFilterMode] = useState(
    propFilterMode || telemetry?.target_filter_mode || 'ALL'
  );

  const [isExpanded, setIsExpanded] = useState(false);
  const hoverTimeoutRef = useRef(null);

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

  const handleMouseEnter = () => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    setIsExpanded(true);
  };

  const handleMouseLeave = () => {
    hoverTimeoutRef.current = setTimeout(() => {
      setIsExpanded(false);
    }, 450);
  };

  const handleToggleExpand = (e) => {
    e.stopPropagation();
    soundFX.playClick();
    setIsExpanded((prev) => !prev);
  };

  const handleFilter = (e) => {
    e?.stopPropagation();
    soundFX.playClick();
    const cycleOrder = ['ALL', 'OBJECTS_ONLY', 'HUMANS_ONLY', 'OFF'];
    const currentIdx = cycleOrder.indexOf(localFilterMode);
    const nextMode = cycleOrder[(currentIdx + 1) % cycleOrder.length];

    setLocalFilterMode(nextMode);
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
    if (localFilterMode === 'OBJECTS_ONLY') return 'OBJECTS';
    if (localFilterMode === 'HUMANS_ONLY') return 'HUMANS';
    if (localFilterMode === 'OFF') return 'MUTED';
    return 'OMNI VIEW';
  };

  const isPerceptionActive = localFilterMode !== 'OFF';

  return (
    <div
      className={`v2-command-dock-container ${isExpanded ? 'v2-dock-open' : 'v2-dock-closed'}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* ── 1. Collapsed Floating Trigger Bar ── */}
      <div
        className="v2-dock-trigger-tab"
        onClick={handleToggleExpand}
        title={isExpanded ? 'Click to minimize dock' : 'Hover or click to open command matrix'}
      >
        <span className="v2-dock-core-dot animate-pulse" />
        <span className="v2-dock-trigger-title">⚡ AURA COMMAND ORBIT</span>

        {/* Live Mini Subsystem LEDs */}
        <div className="v2-dock-mini-leds">
          <span className={`mini-led ${isPerceptionActive ? 'led-cyan' : 'led-off'}`} title="Perception" />
          <span className={`mini-led ${isGestures ? 'led-emerald' : 'led-off'}`} title="Gestures" />
          <span className={`mini-led ${isSAHI ? 'led-emerald' : 'led-off'}`} title="SAHI" />
          <span className={`mini-led ${isTracking ? 'led-cyan' : 'led-off'}`} title="Tracking" />
          <span className={`mini-led ${isOCR ? 'led-cyan' : 'led-off'}`} title="OCR" />
          <span className={`mini-led ${isVoice ? 'led-emerald' : 'led-off'}`} title="Voice" />
        </div>

        <span className="v2-dock-arrow">{isExpanded ? '▼' : '▲'}</span>
      </div>

      {/* ── 2. Expanded Aerospace Control Pods ── */}
      <div className="v2-dock-panel">
        <div className="v2-pods-grid">
          {/* Pod 1: Perception Filter */}
          <div
            className={`v2-control-pod ${isPerceptionActive ? 'pod-active' : 'pod-off'}`}
            onClick={handleFilter}
            title="Cycle Perception: Omni View ➔ Objects Only ➔ Humans Only ➔ Muted"
          >
            <div className="pod-radial-ring ring-cyan">
              <span className="pod-icon">{getFilterIcon()}</span>
            </div>
            <div className="pod-meta">
              <span className="pod-title">PERCEPTION</span>
              <span className="pod-status">{getFilterLabel()}</span>
            </div>
          </div>

          {/* Pod 2: 3D Hand Gestures */}
          <div
            className={`v2-control-pod ${isGestures ? 'pod-active' : ''}`}
            onClick={(e) => {
              e.stopPropagation();
              soundFX.playToggle(!isGestures);
              if (onToggleGestures) onToggleGestures();
            }}
            title="Arm / Disarm 3D Hand Gestures (MediaPipe 21 Landmarks)"
          >
            <div className="pod-radial-ring ring-emerald">
              <span className="pod-icon">🖐️</span>
            </div>
            <div className="pod-meta">
              <span className="pod-title">3D GESTURES</span>
              <span className="pod-status">{isGestures ? 'ARMED' : 'STANDBY'}</span>
            </div>
          </div>

          {/* Pod 3: SAHI Matrix */}
          <div
            className={`v2-control-pod ${isSAHI ? 'pod-active' : ''}`}
            onClick={(e) => {
              e.stopPropagation();
              soundFX.playToggle(!isSAHI);
              if (onToggleSAHI) onToggleSAHI();
            }}
            title="Toggle SAHI Sliced High-Res Inference"
          >
            <div className="pod-radial-ring ring-emerald">
              <span className="pod-icon">🤘</span>
            </div>
            <div className="pod-meta">
              <span className="pod-title">SAHI MATRIX</span>
              <span className="pod-status">{isSAHI ? 'SLICING' : 'OFF'}</span>
            </div>
          </div>

          {/* Pod 4: Spatial Tracker */}
          <div
            className={`v2-control-pod ${isTracking ? 'pod-active' : ''}`}
            onClick={(e) => {
              e.stopPropagation();
              soundFX.playToggle(!isTracking);
              if (onToggleTracking) onToggleTracking();
            }}
            title="Toggle Spatial Tracking (Auto-hides Radar when OFF)"
          >
            <div className="pod-radial-ring ring-cyan">
              <span className="pod-icon">🎯</span>
            </div>
            <div className="pod-meta">
              <span className="pod-title">SPATIAL TRACKER</span>
              <span className="pod-status">{isTracking ? 'TRACKING' : 'MUTED'}</span>
            </div>
          </div>

          {/* Pod 5: OCR Scanner */}
          <div
            className={`v2-control-pod ${isOCR ? 'pod-active' : ''}`}
            onClick={(e) => {
              e.stopPropagation();
              soundFX.playToggle(!isOCR);
              if (onToggleOCR) onToggleOCR();
            }}
            title="Toggle Machine Vision OCR Text Scanner"
          >
            <div className="pod-radial-ring ring-cyan">
              <span className="pod-icon">🔍</span>
            </div>
            <div className="pod-meta">
              <span className="pod-title">OCR SCANNER</span>
              <span className="pod-status">{isOCR ? 'SCANNING' : 'MUTED'}</span>
            </div>
          </div>

          {/* Pod 6: Neural Voice */}
          <div
            className={`v2-control-pod ${isVoice ? 'pod-active' : ''}`}
            onClick={(e) => {
              e.stopPropagation();
              soundFX.playToggle(!isVoice);
              if (onToggleVoice) onToggleVoice();
            }}
            title="Toggle Voice Assistant (Call Me 🤙 Gesture)"
          >
            <div className="pod-radial-ring ring-emerald">
              <span className="pod-icon">🤙</span>
            </div>
            <div className="pod-meta">
              <span className="pod-title">NEURAL VOICE</span>
              <span className="pod-status">{isVoice ? 'LISTENING' : 'STANDBY'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
