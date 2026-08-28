import { useRef, useEffect, useState } from 'react';
import { soundFX } from '../utils/audioFx';

/**
 * TacticalViewport — Centerpiece Vision Perception screen with holographic corner HUD brackets,
 * scanline shaders, live 3D gesture ring hologram, real-time status banners, and target reticles.
 */
export default function TacticalViewport({
  frame,
  scene,
  telemetry,
  activeToast,
  onObjectClick,
  onOpenGuide,
}) {
  const canvasRef = useRef(null);
  const imgRef = useRef(new Image());
  const [hoveredEntity, setHoveredEntity] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ w: 640, h: 480 });
  const [isFlashing, setIsFlashing] = useState(false);
  const prevToastRef = useRef(null);

  const isFrozen = telemetry?.gesture_mode === 'FROZEN';
  const isSAHI = telemetry?.sahi_enabled;
  const isVoice = telemetry?.voice_listening || telemetry?.voice_status === 'LISTENING';
  const isClean = telemetry?.gesture_mode === 'HIDE_BOXES';
  const pointedTarget = telemetry?.pointed_target;
  const activeGesture = telemetry?.active_gesture || 'none';

  // Trigger sound and flash on snapshot toast
  useEffect(() => {
    if (activeToast && activeToast !== prevToastRef.current) {
      prevToastRef.current = activeToast;
      if (activeToast.includes('SNAPSHOT')) {
        soundFX.playShutter();
        setIsFlashing(true);
        setTimeout(() => setIsFlashing(false), 350);
      } else if (activeToast.includes('LOCKED')) {
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
      canvas.width = img.width;
      canvas.height = img.height;
      setCanvasSize({ w: img.width, h: img.height });
      ctx.drawImage(img, 0, 0);
    };
    img.src = `data:image/jpeg;base64,${frame}`;
  }, [frame]);

  const handleCanvasClick = (e) => {
    if (!scene?.entities?.length || !onObjectClick) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvasSize.w / rect.width;
    const scaleY = canvasSize.h / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    for (const entity of scene.entities) {
      const [x1, y1, x2, y2] = entity.bbox;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        soundFX.playLockOn();
        onObjectClick(entity);
        return;
      }
    }
  };

  const handleCanvasMove = (e) => {
    if (!scene?.entities?.length) { setHoveredEntity(null); return; }
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvasSize.w / rect.width;
    const scaleY = canvasSize.h / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    for (const entity of scene.entities) {
      const [x1, y1, x2, y2] = entity.bbox;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        setHoveredEntity(entity);
        return;
      }
    }
    setHoveredEntity(null);
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
    <div className={`tactical-viewport glass-panel ${isFrozen ? 'viewport-frozen' : ''}`}>
      {/* ── Top Viewport Bar ── */}
      <div className="viewport-header">
        <div className="viewport-title-left">
          <span className="crosshair-icon">✛</span>
          <span className="viewport-label">TACTICAL PRIMARY SENSOR FEED</span>
          <span className="viewport-coords">CAM_01 // 640×480 @ 60FPS</span>
        </div>
        <div className="viewport-title-right">
          {isSAHI && <span className="tag-sahi">🤘 SAHI SLICED (320px)</span>}
          {scene && <span className="tag-count">{scene.entity_count || 0} TARGETS</span>}
        </div>
      </div>

      {/* ── Viewport Canvas Display ── */}
      <div className="viewport-display">
        {/* Holographic Corner Brackets */}
        <div className="hud-corner hud-corner-tl" />
        <div className="hud-corner hud-corner-tr" />
        <div className="hud-corner hud-corner-bl" />
        <div className="hud-corner hud-corner-br" />

        {/* Scanlines and Vignette */}
        <div className="scanline-layer" />
        <div className="vignette-layer" />

        {/* Snapshot Flash Overlay */}
        {isFlashing && <div className="snapshot-flash-screen" />}

        {/* Dynamic Action Toast Banner */}
        {activeToast && (
          <div className="holo-action-toast animate-slide-down">
            <span className="toast-glow-icon">⚡</span>
            <span>{activeToast}</span>
          </div>
        )}

        {/* Dynamic State Overlay Banners */}
        <div className="viewport-status-stack">
          {isSAHI && (
            <div className="holo-state-pill pill-sahi animate-slide-in">
              <span className="pill-pulse-dot dot-emerald"></span>
              <span>🤘 SAHI HIGH-RES MATRIX ACTIVE</span>
            </div>
          )}

          {isVoice && (
            <div className="holo-state-pill pill-voice animate-slide-in">
              <span className="pill-pulse-dot dot-cyan"></span>
              <span>🤙 NEURAL VOICE LISTENING...</span>
            </div>
          )}

          {isClean && (
            <div className="holo-state-pill pill-clean animate-slide-in">
              <span>🖐️ CLEAN VIEW ACTIVATED</span>
            </div>
          )}

          {isFrozen && (
            <div className="holo-state-pill pill-frozen animate-slide-in">
              <span>❄️ OVERLAY FROZEN</span>
            </div>
          )}

          {pointedTarget && (
            <div className="holo-state-pill pill-target animate-slide-in">
              <span className="pill-pulse-dot dot-cyan"></span>
              <span>👉 LOCKED TARGET: {pointedTarget.toUpperCase()}</span>
            </div>
          )}
        </div>

        {/* Main Stream Canvas */}
        {frame ? (
          <canvas
            ref={canvasRef}
            className="viewport-canvas"
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMove}
            onMouseLeave={() => setHoveredEntity(null)}
          />
        ) : (
          <div className="viewport-loading-state">
            <div className="loading-radar-ring animate-pulse">⬢</div>
            <p className="loading-text">SYNCHRONIZING NEURAL SENSOR FEED...</p>
          </div>
        )}

        {/* Floating 3D Gesture Hologram Pill (Bottom Left) */}
        <div className="holo-gesture-dock">
          <div className={`holo-gesture-ring ${activeGesture !== 'none' ? 'ring-active' : ''}`}>
            <span className="gesture-holo-icon">{gestureIcons[activeGesture] || '✋'}</span>
            <div className="gesture-holo-meta">
              <span className="gesture-holo-name">
                {activeGesture === 'none' ? 'AWAITING GESTURE' : activeGesture.replace('_', ' ').toUpperCase()}
              </span>
              <span className="gesture-holo-sub">21-LANDMARK 3D KINEMATICS</span>
            </div>
          </div>

          <button
            className="btn-holo-manual"
            onClick={() => {
              soundFX.playToggle(true);
              onOpenGuide();
            }}
            title="Open Gesture Guide"
          >
            📖 Manual
          </button>
        </div>

        {/* Hover Tooltip */}
        {hoveredEntity && (
          <div className="holo-entity-tooltip animate-fade-in">
            <span className="tooltip-tag">{hoveredEntity.class_name.toUpperCase()}</span>
            {hoveredEntity.track_id != null && (
              <span className="tooltip-id">#ID {hoveredEntity.track_id}</span>
            )}
            <span className="tooltip-conf">{Math.round((hoveredEntity.confidence || 0.9) * 100)}% Match</span>
            <span className="tooltip-action">Click / Pinch 👌 to Inspect</span>
          </div>
        )}
      </div>
    </div>
  );
}
