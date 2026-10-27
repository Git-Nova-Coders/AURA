import { useRef, useEffect, useState } from 'react';
import ActiveStateBanners from './ActiveStateBanners';
import GestureHUD from './GestureHUD';

/**
 * Live Video Feed — Renders real-time video stream with holographic HUD scanlines,
 * interactive 3D gesture HUD overlays, state toggle banners, and target entity inspection.
 */
export default function LiveFeed({ frame, scene, telemetry, activeToast, onObjectClick, onOpenGuide }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(new Image());
  const [hoveredEntity, setHoveredEntity] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ w: 640, h: 480 });

  const isFrozen = telemetry?.gesture_mode === 'FROZEN';

  // Draw frame onto canvas
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
      // Sort ascending by area so smaller nested objects (e.g. water bottle) take precedence
      candidates.sort((a, b) => a.area - b.area);
      onObjectClick(candidates[0].entity);
    }
  };

  const handleCanvasMove = (e) => {
    if (!scene?.entities?.length) {
      setHoveredEntity(null);
      return;
    }
    const coords = getCanvasCoords(e);
    if (!coords) {
      setHoveredEntity(null);
      return;
    }
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

  return (
    <div className={`live-feed-card glass-card ${isFrozen ? 'frozen-overlay' : ''}`}>
      {/* ── Title Bar ── */}
      <div className="feed-title-bar">
        <div className="feed-title-left">
          <span className="feed-icon">📹</span>
          <span className="panel-title">Neural Perception Feed</span>
          {isFrozen && <span className="badge badge-cyan">❄️ FROZEN</span>}
        </div>
        <div className="feed-title-right">
          {scene && (
            <span className="badge badge-cyan">{scene.entity_count || 0} Entities</span>
          )}
          {telemetry?.har_activity && (
            <span className="badge badge-emerald animate-pulse" title={`Confidence: ${Math.round((telemetry.har_confidence || 0) * 100)}%`}>
              🏃 {telemetry.har_activity.toUpperCase()}
            </span>
          )}
          {telemetry?.had_status && (
            <span
              className={`badge ${telemetry.had_status.includes('ANOMALY') ? 'badge-rose animate-pulse' : 'badge-emerald'}`}
              title={`HAD Confidence: ${Math.round((telemetry.had_confidence || 0) * 100)}%`}
            >
              🛡️ {telemetry.had_status.toUpperCase()}
            </span>
          )}
          {telemetry?.ocr_text_count > 0 && (
            <span className="badge badge-emerald">{telemetry.ocr_text_count} OCR Texts</span>
          )}
        </div>
      </div>

      {/* ── Video Viewport & Overlays ── */}
      <div className="feed-viewport">
        {/* Real-time State Banners (SAHI, Voice, Freeze, Toasts) */}
        <ActiveStateBanners telemetry={telemetry} activeToast={activeToast} />

        {/* Scanlines Effect */}
        <div className="scanlines" />

        {frame ? (
          <canvas
            ref={canvasRef}
            className="feed-canvas"
            onClick={handleCanvasClick}
            onTouchEnd={handleCanvasClick}
            onMouseMove={handleCanvasMove}
            onMouseLeave={() => setHoveredEntity(null)}
          />
        ) : (
          <div className="feed-placeholder">
            <div className="placeholder-beacon animate-pulse">⬢</div>
            <p className="placeholder-text">Establishing neural link to AURA pipeline...</p>
            <div className="skeleton" style={{ width: '220px', height: '8px', marginTop: '14px' }} />
          </div>
        )}

        {/* Floating 3D Gesture HUD Pill */}
        <GestureHUD telemetry={telemetry} onOpenGuide={onOpenGuide} />

        {/* Hover Tooltip */}
        {hoveredEntity && (
          <div className="entity-hover-tooltip animate-fade-in">
            <span className="tooltip-title">{hoveredEntity.class_name.toUpperCase()}</span>
            {hoveredEntity.track_id != null && (
              <span className="badge badge-cyan">ID #{hoveredEntity.track_id}</span>
            )}
            <span className="tooltip-conf">{Math.round((hoveredEntity.confidence || 0.9) * 100)}%</span>
            <span className="tooltip-hint">Click or Pinch 👌 to Inspect</span>
          </div>
        )}
      </div>
    </div>
  );
}
