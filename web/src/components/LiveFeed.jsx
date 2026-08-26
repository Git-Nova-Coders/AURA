import { useRef, useEffect, useState } from 'react';

/**
 * Live Video Feed — Renders MJPEG stream with interactive detection overlays.
 */
export default function LiveFeed({ frame, scene, onObjectClick }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(new Image());
  const [hoveredEntity, setHoveredEntity] = useState(null);
  const [canvasSize, setCanvasSize] = useState({ w: 640, h: 480 });

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

  return (
    <div style={styles.container} className="glass-card">
      <div style={styles.titleBar}>
        <span className="panel-title"><span className="icon">📹</span> Live Visual Perception Feed</span>
        {scene && (
          <span className="badge badge-cyan">{scene.entity_count || 0} objects</span>
        )}
      </div>

      <div style={styles.feedWrapper}>
        {frame ? (
          <canvas
            ref={canvasRef}
            style={styles.canvas}
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMove}
            onMouseLeave={() => setHoveredEntity(null)}
          />
        ) : (
          <div style={styles.placeholder}>
            <div style={styles.placeholderIcon}>◉</div>
            <p style={styles.placeholderText}>Connecting to AURA vision pipeline...</p>
            <div className="skeleton" style={{ width: '200px', height: '8px', marginTop: '12px' }} />
          </div>
        )}

        {/* Hover tooltip */}
        {hoveredEntity && (
          <div style={styles.tooltip} className="animate-fade-in">
            <span style={styles.tooltipName}>{hoveredEntity.class_name}</span>
            {hoveredEntity.track_id != null && (
              <span className="badge badge-cyan">#{hoveredEntity.track_id}</span>
            )}
            <span className="badge badge-emerald">{(hoveredEntity.confidence * 100).toFixed(0)}%</span>
            {hoveredEntity.reliability_label && (
              <span className={`badge ${hoveredEntity.reliability_label === 'reliable' ? 'badge-emerald' : 'badge-amber'}`}>
                {hoveredEntity.reliability_label}
              </span>
            )}
            <span style={styles.tooltipRegion}>📍 {hoveredEntity.spatial_pos}</span>
          </div>
        )}
      </div>

      {/* Click instruction */}
      <div style={styles.hint}>
        💡 Click any detected object for Knowledge Lookup
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    flex: 1,
    minHeight: 0,
    overflow: 'hidden',
  },
  titleBar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '4px 8px 4px 0',
  },
  feedWrapper: {
    position: 'relative',
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    borderRadius: 'var(--radius-md)',
    margin: '0 var(--space-md)',
    background: '#080b10',
  },
  canvas: {
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    cursor: 'crosshair',
    borderRadius: 'var(--radius-md)',
  },
  placeholder: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '300px',
  },
  placeholderIcon: {
    fontSize: '3rem',
    color: 'var(--accent-cyan)',
    opacity: 0.3,
    animation: 'pulse-glow 2s ease-in-out infinite',
  },
  placeholderText: {
    fontSize: '0.85rem',
    color: 'var(--text-muted)',
    marginTop: '12px',
  },
  tooltip: {
    position: 'absolute',
    top: '12px',
    left: '12px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 14px',
    background: 'rgba(10, 13, 20, 0.92)',
    backdropFilter: 'blur(8px)',
    border: '1px solid var(--accent-cyan)',
    borderRadius: 'var(--radius-md)',
    boxShadow: '0 0 16px rgba(0, 240, 255, 0.2)',
    zIndex: 10,
  },
  tooltipName: {
    fontFamily: 'var(--font-display)',
    fontWeight: 600,
    fontSize: '0.85rem',
    color: 'var(--text-primary)',
    textTransform: 'capitalize',
  },
  tooltipRegion: {
    fontSize: '0.72rem',
    color: 'var(--text-muted)',
  },
  hint: {
    fontSize: '0.7rem',
    color: 'var(--text-muted)',
    textAlign: 'center',
    padding: '6px 0 10px',
    opacity: 0.6,
  },
};
