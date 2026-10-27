import { useEffect, useRef } from 'react';

/**
 * TacticalRadarCanvas — AURA V2 Miniature Spatial Model.
 * Interactive canvas with concentric distance range rings, dynamic spatial vector rays,
 * rotating surveillance beam, target lock expanding rings, and entity blips.
 */
export default function TacticalRadarCanvas({ scene, pointedTarget, onSelectEntity, onHoverEntity }) {
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
    let pulseScale = 0;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const radius = Math.min(cx, cy) - 14;

      ctx.clearRect(0, 0, w, h);

      // Radar deep void background
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(4, 9, 18, 0.9)';
      ctx.fill();

      // ── Concentric Range Rings (1.5m, 3.0m, 4.5m) ──
      const rings = [0.33, 0.66, 1.0];
      rings.forEach((rScale, idx) => {
        ctx.beginPath();
        ctx.arc(cx, cy, radius * rScale, 0, Math.PI * 2);
        ctx.strokeStyle = idx === 2 ? 'rgba(0, 240, 255, 0.35)' : 'rgba(0, 240, 255, 0.1)';
        ctx.lineWidth = idx === 2 ? 1.2 : 0.8;
        ctx.stroke();

        // Distance Range Labels (High contrast cyan)
        ctx.fillStyle = idx === 2 ? '#00f0ff' : 'rgba(0, 240, 255, 0.75)';
        ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
        ctx.fillText(`${((idx + 1) * 1.5).toFixed(1)}m`, cx + 6, cy - radius * rScale + 10);
      });

      // ── Cardinal Axes & Angle Ticks ──
      ctx.beginPath();
      ctx.moveTo(cx - radius, cy);
      ctx.lineTo(cx + radius, cy);
      ctx.moveTo(cx, cy - radius);
      ctx.lineTo(cx, cy + radius);
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.22)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // ── Camera FOV Sector Cone ──
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, radius, -Math.PI / 2 - Math.PI / 5, -Math.PI / 2 + Math.PI / 5);
      ctx.closePath();
      ctx.fillStyle = 'rgba(0, 240, 255, 0.06)';
      ctx.fill();

      // ── Rotating Holographic Sweep Beam ──
      angle += 0.035;
      const sweepGrad = ctx.createConicGradient(angle, cx, cy);
      sweepGrad.addColorStop(0, 'rgba(0, 240, 255, 0)');
      sweepGrad.addColorStop(0.82, 'rgba(0, 240, 255, 0)');
      sweepGrad.addColorStop(1, 'rgba(0, 255, 157, 0.45)');

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = sweepGrad;
      ctx.fill();

      // Pulse wave counter for lock animation
      pulseScale = (pulseScale + 0.04) % 1;

      // ── Draw Entities & Spatial Vector Lines ──
      const entities = entitiesRef.current;
      const positions = [];

      entities.forEach((entity) => {
        const bbox = entity.bbox || [320, 240, 320, 240];
        const normX = ((bbox[0] + bbox[2]) / 2 / 640 - 0.5) * 1.6;
        const normY = ((bbox[1] + bbox[3]) / 2 / 480 - 0.5) * 1.4;

        const bx = cx + normX * (radius * 0.85);
        const by = cy + normY * (radius * 0.85);
        positions.push({ entity, bx, by });

        const isLocked = pointedTarget && entity.class_name?.toLowerCase() === pointedTarget.toLowerCase();

        // ── Spatial Vector Line from Center to Entity ──
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(bx, by);
        ctx.strokeStyle = isLocked ? 'rgba(255, 183, 0, 0.7)' : 'rgba(0, 255, 157, 0.35)';
        ctx.lineWidth = isLocked ? 1.5 : 0.8;
        if (!isLocked) ctx.setLineDash([3, 3]);
        else ctx.setLineDash([]);
        ctx.stroke();
        ctx.setLineDash([]);

        // ── Entity Blip Point ──
        ctx.shadowColor = isLocked ? '#ffb700' : '#00ff9d';
        ctx.shadowBlur = isLocked ? 14 : 8;

        ctx.beginPath();
        ctx.arc(bx, by, isLocked ? 5.5 : 4, 0, Math.PI * 2);
        ctx.fillStyle = isLocked ? '#ffb700' : '#00ff9d';
        ctx.fill();

        if (isLocked) {
          // Expanding Target Ring
          ctx.beginPath();
          ctx.arc(bx, by, 7 + pulseScale * 12, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(255, 183, 0, ${1 - pulseScale})`;
          ctx.lineWidth = 1.4;
          ctx.stroke();
        }

        ctx.shadowBlur = 0;

        // High-Contrast Luminous Label
        ctx.fillStyle = isLocked ? '#ffb700' : '#ffffff';
        ctx.font = 'bold 8.5px "JetBrains Mono", monospace';
        ctx.fillText(entity.class_name.toUpperCase(), bx + 7, by - 4);
      });

      // ── Spatial Vector Inter-Links (Connect nearby entities) ──
      if (positions.length > 1) {
        for (let i = 0; i < positions.length - 1; i++) {
          const p1 = positions[i];
          const p2 = positions[i + 1];
          const dist = Math.hypot(p1.bx - p2.bx, p1.by - p2.by);
          if (dist < radius * 0.7) {
            ctx.beginPath();
            ctx.moveTo(p1.bx, p1.by);
            ctx.lineTo(p2.bx, p2.by);
            ctx.strokeStyle = 'rgba(0, 255, 157, 0.12)';
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [pointedTarget]);

  return (
    <div className="v2-tactical-radar-box">
      <canvas
        ref={canvasRef}
        width={250}
        height={210}
        className="v2-radar-canvas"
      />
    </div>
  );
}
