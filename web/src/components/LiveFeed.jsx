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

  const handleCanvasClick = (e) => {
    if (!scene?.entities?.length || !onObjectClick) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvasSize.w / rect.width;
    const scaleY = canvasSize.h / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Find clicked entity by bbox
    for (const entity of scene.entities) {
      const [x1, y1, x2, y2] = entity.bbox;
      if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
        onObjectClick(entity);
        return;
      }
    }
  };

  const handleCanvasMove = (e) => {
    if (!scene?.entities?.length) {
      setHoveredEntity(null);
      return;
    }

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

  return (
    <div className={`live-feed-card glass-card ${isFrozen ? 'frozen-overlay' : ''}`}>
      {/* ── Title Bar ── */}
      <div className="feed-title-bar">
        <div className="feed-title-left">
          <span className="feed-icon">📹</span>
          <span className="panel-title">Cybernetic Perception Feed</span>
          {isFrozen && <span className="badge badge-cyan">❄️ FROZEN</span>}
        </div>
        <div className="feed-title-right">
          {scene && (
            <span className="badge badge-cyan">{scene.entity_count || 0} Entities</span>
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
