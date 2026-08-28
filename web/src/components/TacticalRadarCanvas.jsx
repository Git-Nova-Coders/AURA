import { useEffect, useRef } from 'react';

/**
 * TacticalRadarCanvas — Canvas-driven sci-fi rotating holographic radar with
 * sweep beam gradient, concentric range rings, azimuth degrees, and interactive entity pings.
 */
export default function TacticalRadarCanvas({ scene, pointedTarget, onSelectEntity }) {
  const canvasRef = useRef(null);
  const entitiesRef = useRef([]);

  useEffect(() => {
    entitiesRef.current = scene?.entities || [];
  }, [scene]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let angle = 0;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const radius = Math.min(cx, cy) - 10;

      ctx.clearRect(0, 0, w, h);

      // Radar dark background
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(6, 10, 20, 0.85)';
      ctx.fill();

      // Concentric Range Rings
      ctx.lineWidth = 1;
      const rings = [0.33, 0.66, 1.0];
      rings.forEach((rScale, idx) => {
        ctx.beginPath();
        ctx.arc(cx, cy, radius * rScale, 0, Math.PI * 2);
        ctx.strokeStyle = idx === 2 ? 'rgba(0, 240, 255, 0.4)' : 'rgba(0, 240, 255, 0.12)';
        ctx.stroke();

        // Distance Labels
        ctx.fillStyle = 'rgba(0, 240, 255, 0.4)';
        ctx.font = '8px monospace';
        ctx.fillText(`${(idx + 1) * 1.5}m`, cx + 4, cy - radius * rScale + 10);
      });

      // Crosshair Axes
      ctx.beginPath();
      ctx.moveTo(cx - radius, cy);
      ctx.lineTo(cx + radius, cy);
      ctx.moveTo(cx, cy - radius);
      ctx.lineTo(cx, cy + radius);
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.12)';
      ctx.stroke();

      // Camera FOV Sector Cone (60 degree camera view)
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, radius, -Math.PI / 2 - Math.PI / 6, -Math.PI / 2 + Math.PI / 6);
      ctx.closePath();
      ctx.fillStyle = 'rgba(0, 240, 255, 0.04)';
      ctx.fill();

      // Rotating Sweep Beam
      angle += 0.035;
      const sweepGrad = ctx.createConicGradient(angle, cx, cy);
      sweepGrad.addColorStop(0, 'rgba(0, 240, 255, 0)');
      sweepGrad.addColorStop(0.85, 'rgba(0, 240, 255, 0)');
      sweepGrad.addColorStop(1, 'rgba(0, 240, 255, 0.35)');

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = sweepGrad;
      ctx.fill();

      // Draw Entity Blips
      const entities = entitiesRef.current;
      entities.forEach((entity) => {
        // Map bounding box center to radar polar coordinates
        const bbox = entity.bbox || [320, 240, 320, 240];
        const normX = ((bbox[0] + bbox[2]) / 2 / 640 - 0.5) * 1.6; // -0.8 to 0.8
        const normY = ((bbox[1] + bbox[3]) / 2 / 480 - 0.5) * 1.4;

        const bx = cx + normX * (radius * 0.85);
        const by = cy + normY * (radius * 0.85);

        const isLocked = pointedTarget && entity.class_name?.toLowerCase() === pointedTarget.toLowerCase();

        ctx.shadowColor = isLocked ? '#00f0ff' : '#00ff9d';
        ctx.shadowBlur = isLocked ? 12 : 6;

        ctx.beginPath();
        ctx.arc(bx, by, isLocked ? 5.5 : 3.5, 0, Math.PI * 2);
        ctx.fillStyle = isLocked ? '#00f0ff' : '#00ff9d';
        ctx.fill();

        if (isLocked) {
          // Animated lock ring
          ctx.beginPath();
          ctx.arc(bx, by, 9, 0, Math.PI * 2);
          ctx.strokeStyle = '#00f0ff';
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }

        ctx.shadowBlur = 0;

        // Label
        ctx.fillStyle = isLocked ? '#00f0ff' : '#cbd5e1';
        ctx.font = isLocked ? 'bold 9px monospace' : '8px monospace';
        ctx.fillText(entity.class_name.toUpperCase(), bx + 6, by - 4);
      });

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [pointedTarget]);

  return (
    <div className="tactical-radar-box">
      <canvas
        ref={canvasRef}
        width={270}
        height={220}
        className="tactical-radar-canvas"
      />
    </div>
  );
}
