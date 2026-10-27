import { useRef, useEffect, useState } from 'react';
import { soundFX } from '../utils/audioFx';
import HudFrame from './hud/HudFrame';
import HudGrid from './hud/HudGrid';

/**
 * TacticalViewport — AURA V2 Hero Camera Viewport (75-85% Presence)
 * Cinematic aerospace sensor feed with signature corner reticles,
 * perspective grid overlay (3-8%), multi-stage target locking reticle,
 * and machine vision perception badges.
 */
export default function TacticalViewport({
  frame,
  telemetry,
  scene,
  activeToast,
  filterMode: propFilterMode,
  onObjectClick,
  onOpenGuide,
  onToggleGestures,
  onSetTargetFilter,
}) {
  const canvasRef = useRef(null);
  const imgRef = useRef(new Image());
  const [hoveredEntity, setHoveredEntity] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ w: 640, h: 480 });
  const [isFlashing, setIsFlashing] = useState(false);
  const prevToastRef = useRef(null);

  const isSAHI = telemetry?.sahi_enabled ?? false;
  const isGestures = telemetry?.gestures_enabled ?? false;
  const isVoice = telemetry?.voice_listening || telemetry?.voice_status === 'LISTENING';
  const isClean = isGestures && telemetry?.gesture_mode === 'HIDE_BOXES';
  const isFrozen = isGestures && telemetry?.gesture_mode === 'FROZEN';
  const activeGesture = isGestures ? (telemetry?.active_gesture || 'none') : 'none';
  const pointedTarget = isGestures ? telemetry?.pointed_target : null;
  const filterMode = propFilterMode || telemetry?.target_filter_mode || 'ALL';

  // Sound and flash triggers
  useEffect(() => {
    if (activeToast && activeToast !== prevToastRef.current) {
      prevToastRef.current = activeToast;
      if (activeToast.includes('SNAPSHOT')) {
        soundFX.playShutter();
        setIsFlashing(true);
        setTimeout(() => setIsFlashing(false), 350);
      } else if (activeToast.includes('LOCKED') || activeToast.includes('INSPECT')) {
        soundFX.playLockOn();
      } else if (activeToast.includes('VOICE')) {
        soundFX.playVoiceChime();
      } else if (activeToast.includes('SAHI')) {
        soundFX.playToggle(true);
      }
    }
  }, [activeToast]);

  // Render incoming JPEG stream to Canvas
  useEffect(() => {
    if (!frame || !canvasRef.current) return;
    const img = imgRef.current;
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (canvas.width !== img.width || canvas.height !== img.height) {
        canvas.width = img.width;
        canvas.height = img.height;
        setCanvasSize({ w: img.width, h: img.height });
      }
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${frame}`;
  }, [frame]);

  const getCanvasCoords = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvasSize.w / rect.width;
    const scaleY = canvasSize.h / rect.height;
    let clientX = e.clientX;
    let clientY = e.clientY;
    if (clientX === undefined && e.changedTouches && e.changedTouches.length > 0) {
      clientX = e.changedTouches[0].clientX;
      clientY = e.changedTouches[0].clientY;
    }
    if (clientX === undefined && e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    }
    if (clientX === undefined || clientY === undefined) return null;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  };

  const handleCanvasClick = (e) => {
    if (!scene?.entities?.length || !onObjectClick) return;
    const coords = getCanvasCoords(e);
    if (!coords) return;
    const { x, y } = coords;

    const candidates = [];
    for (const entity of scene.entities) {
      const [x1, y1, x2, y2] = entity.bbox;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        const area = Math.abs((x2 - x1) * (y2 - y1));
        candidates.push({ entity, area });
      }
    }

    if (candidates.length > 0) {
      // Sort ascending by area so smaller nested objects (e.g. bottle inside person) are picked first
      candidates.sort((a, b) => a.area - b.area);
      const chosen = candidates[0].entity;
      soundFX.playLockOn();
      onObjectClick(chosen);
    }
  };

  const handleCanvasMove = (e) => {
    if (!scene?.entities?.length) { setHoveredEntity(null); return; }
    const coords = getCanvasCoords(e);
    if (!coords) { setHoveredEntity(null); return; }
    const { x, y } = coords;

    const candidates = [];
    for (const entity of scene.entities) {
      const [x1, y1, x2, y2] = entity.bbox;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        const area = Math.abs((x2 - x1) * (y2 - y1));
        candidates.push({ entity, area });
      }
    }

    if (candidates.length > 0) {
      candidates.sort((a, b) => a.area - b.area);
      setHoveredEntity(candidates[0].entity);
    } else {
      setHoveredEntity(null);
    }
  };

  const gestureIcons = {
    open_palm: '🖐️',
    peace_sign: '✌️',
    pointing: '👉',
    pinch: '👌',
    thumbs_up: '👍',
    thumbs_down: '👎',
    fist: '✊',
    rock_on: '🤘',
    call_me: '🤙',
    none: '✋',
  };

  return (
    <div className={`tactical-viewport-v2-container ${isFrozen ? 'viewport-frozen' : ''}`}>
      <HudFrame
        sensorId="OPTICAL_CAM_01"
        coords="AZ: 042.8° // ELEV: -12.4°"
        status={filterMode === 'OFF' ? 'MUTED' : 'PERCEPTION ACTIVE'}
        showScanline={true}
        className="tactical-viewport-hud-frame"
      >
        {/* ── Layer 5: Perspective Depth Grid (3–8% Opacity) ── */}
        <HudGrid opacity={0.06} />

        {/* ── Top Floating Sensor Telemetry Bar ── */}
        <div className="v2-viewport-header">
          <div className="v2-title-left">
            <span className="v2-beacon-dot animate-pulse" />
            <span className="v2-viewport-label">PRIMARY SENSOR HUD</span>
            <span className="v2-viewport-meta">YOLO26-OV // 640×480 @ {Math.round(telemetry?.fps || 30)} FPS</span>
          </div>

          <div className="v2-title-right">
            {/* Quick 4-Mode Perception Filter Chips */}
            <div className="v2-filter-chips">
              <button
                className={`v2-chip ${filterMode === 'ALL' ? 'v2-chip-active-cyan' : ''}`}
                onClick={() => onSetTargetFilter && onSetTargetFilter('ALL')}
                title="Omni Perception: Physical Objects & Biological Humans"
              >
                🌐 OMNI
              </button>
              <button
                className={`v2-chip ${filterMode === 'OBJECTS_ONLY' ? 'v2-chip-active-cyan' : ''}`}
                onClick={() => onSetTargetFilter && onSetTargetFilter('OBJECTS_ONLY')}
                title="Objects Only: Suppress Humans, Faces & Hand Skeletons"
              >
                📦 OBJECTS
              </button>
              <button
                className={`v2-chip ${filterMode === 'HUMANS_ONLY' ? 'v2-chip-active-cyan' : ''}`}
                onClick={() => onSetTargetFilter && onSetTargetFilter('HUMANS_ONLY')}
                title="Humans Only: Pure Biometric Focus"
              >
                👤 HUMANS
              </button>
              <button
                className={`v2-chip ${filterMode === 'OFF' ? 'v2-chip-active-red' : ''}`}
                onClick={() => onSetTargetFilter && onSetTargetFilter('OFF')}
                title="Mute Perception"
              >
                🛑 OFF
              </button>
            </div>

            {isSAHI && <span className="v2-tag-sahi">🤘 SAHI</span>}
            <span className="v2-tag-count">
              {filterMode === 'OFF' ? '0 TARGETS' : `${scene?.entity_count || 0} TARGETS`}
            </span>
          </div>
        </div>

        {/* ── Hero Camera Sensor Stream Canvas ── */}
        <div className="v2-viewport-canvas-wrapper">
          {/* Shutter snapshot flash */}
          {isFlashing && <div className="snapshot-flash-screen" />}

          {/* Action HUD Toast Alert Banner */}
          {activeToast && (
            <div className="v2-holo-action-toast animate-slide-down">
              <span className="toast-glow-icon">⚡</span>
              <span>{activeToast}</span>
            </div>
          )}

          {/* Dynamic Status Stack Pills */}
          <div className="v2-status-pill-stack">
            {filterMode === 'OFF' && (
              <div className="v2-pill v2-pill-red animate-slide-in">
                <span>🛑 PERCEPTION MUTED</span>
              </div>
            )}

            {!isGestures && (
              <div
                className="v2-pill v2-pill-amber animate-slide-in cursor-pointer"
                onClick={() => onToggleGestures && onToggleGestures()}
                title="Click to Arm 3D Hand Gestures"
              >
                <span>🖐️ GESTURES STANDBY (CLICK TO ARM)</span>
              </div>
            )}

            {isSAHI && (
              <div className="v2-pill v2-pill-emerald animate-slide-in">
                <span className="v2-pulse-dot dot-emerald" />
                <span>🤘 SAHI HIGH-RES MATRIX</span>
              </div>
            )}

            {isVoice && (
              <div className="v2-pill v2-pill-cyan animate-slide-in">
                <span className="v2-pulse-dot dot-cyan" />
                <span>🤙 NEURAL VOICE LISTENING...</span>
              </div>
            )}

            {isClean && (
              <div className="v2-pill v2-pill-cyan animate-slide-in">
                <span>🖐️ CLEAN VIEW ACTIVATED</span>
              </div>
            )}

            {isFrozen && (
              <div className="v2-pill v2-pill-cyan animate-slide-in">
                <span>❄️ OVERLAY FROZEN</span>
              </div>
            )}

            {pointedTarget && (
              <div className="v2-pill v2-pill-target animate-slide-in">
                <span className="v2-pulse-dot dot-amber" />
                <span>🎯 TARGET LOCKED: {pointedTarget.toUpperCase()}</span>
              </div>
            )}

            {telemetry?.har_activity && (
              <div className="v2-pill v2-pill-emerald animate-slide-in" title={`Confidence: ${Math.round((telemetry.har_confidence || 0) * 100)}%`}>
                <span className="v2-pulse-dot dot-cyan animate-pulse" />
                <span>🏃 ACTION: {telemetry.har_activity.toUpperCase()} ({Math.round((telemetry.har_confidence || 0) * 100)}%)</span>
              </div>
            )}

            {telemetry?.had_status && (
              <div
                className={`v2-pill ${telemetry.had_status.includes('ANOMALY') ? 'v2-pill-rose animate-pulse' : 'v2-pill-emerald'} animate-slide-in`}
                title={`HAD Confidence: ${Math.round((telemetry.had_confidence || 0) * 100)}%`}
              >
                <span className={`v2-pulse-dot ${telemetry.had_status.includes('ANOMALY') ? 'dot-rose animate-pulse' : 'dot-emerald'}`} />
                <span>🛡️ HAD: {telemetry.had_status.toUpperCase()} ({Math.round((telemetry.had_confidence || 0) * 100)}%)</span>
              </div>
            )}

            {telemetry?.camera_source === 'remote' && telemetry?.remote_device_connected && (
              <div className="v2-pill v2-pill-emerald animate-slide-in" title="Streaming from external device camera">
                <span className="v2-pulse-dot dot-emerald animate-pulse" />
                <span>📱 REMOTE DEVICE CAMERA ACTIVE</span>
              </div>
            )}
          </div>

          {/* Main Video Stream Canvas */}
          {frame ? (
            <canvas
              ref={canvasRef}
              className="v2-viewport-canvas"
              onClick={handleCanvasClick}
              onTouchEnd={handleCanvasClick}
              onMouseMove={handleCanvasMove}
              onMouseLeave={() => setHoveredEntity(null)}
            />
          ) : (
            <div className="v2-viewport-loading-state">
              <div className="v2-loading-ring animate-pulse">⬢</div>
              <p className="v2-loading-text">SYNCHRONIZING SENSOR STREAM...</p>
            </div>
          )}

          {/* ── Target Lock Multi-Stage Overlay (When Target is Pointed/Locked) ── */}
          {pointedTarget && (
            <div className="v2-target-lock-indicator animate-pulse">
              <div className="lock-crosshair-ring" />
              <div className="lock-target-label">
                <span className="lock-sub">TARGET ACQUIRED</span>
                <span className="lock-name">{pointedTarget.toUpperCase()}</span>
                <span className="lock-status">TRACKED // READY TO PINCH 👌</span>
              </div>
            </div>
          )}

          {/* Hover Tooltip Overlay */}
          {hoveredEntity && (
            <div className="v2-entity-tooltip animate-fade-in">
              <span className="v2-tooltip-tag">{hoveredEntity.class_name.toUpperCase()}</span>
              {hoveredEntity.track_id != null && (
                <span className="v2-tooltip-id">#ID {hoveredEntity.track_id}</span>
              )}
              <span className="v2-tooltip-conf">{Math.round((hoveredEntity.confidence || 0.9) * 100)}% Match</span>
              <span className="v2-tooltip-action">Click / Pinch 👌 to Inspect</span>
            </div>
          )}
        </div>
      </HudFrame>
    </div>
  );
}
