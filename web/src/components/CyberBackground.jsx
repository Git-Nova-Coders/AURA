import { useEffect, useRef } from 'react';

/**
 * CyberBackground — AURA V3.0 Deep Purple Holographic Grid & Neural Particles.
 * Renders a deep purple-blue space grid with floating constellation particles,
 * crosshair coordinate nodes, and neural link connections.
 * JARVIS Iron Man inspired aesthetic with deep purple + electric blue palette.
 */
export default function CyberBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const particles = [];
    const count = 55;
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        radius: Math.random() * 1.8 + 0.8,
        // Purple-blue particle color palette
        color: Math.random() > 0.5 ? 'rgba(129, 140, 248, ' : 'rgba(168, 85, 247, ',
        alpha: Math.random() * 0.5 + 0.2,
      });
    }

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const w = canvas.width;
      const h = canvas.height;

      // 1. Deep Purple Space Vignette
      const bgGrad = ctx.createRadialGradient(w / 2, h / 2, 50, w / 2, h / 2, Math.max(w, h));
      bgGrad.addColorStop(0, '#0c0020');
      bgGrad.addColorStop(0.4, '#080014');
      bgGrad.addColorStop(0.7, '#06000f');
      bgGrad.addColorStop(1, '#030008');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // 2. Subtle purple glow in center
      const centerGlow = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w * 0.45);
      centerGlow.addColorStop(0, 'rgba(99, 102, 241, 0.04)');
      centerGlow.addColorStop(0.5, 'rgba(168, 85, 247, 0.02)');
      centerGlow.addColorStop(1, 'transparent');
      ctx.fillStyle = centerGlow;
      ctx.fillRect(0, 0, w, h);

      // 3. Aerospace Tactical Grid (Purple-tinted)
      const gridSize = 60;
      ctx.lineWidth = 0.7;

      for (let x = 0; x < w; x += gridSize) {
        ctx.strokeStyle = x % (gridSize * 4) === 0 ? 'rgba(129, 140, 248, 0.10)' : 'rgba(129, 140, 248, 0.03)';
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }

      for (let y = 0; y < h; y += gridSize) {
        ctx.strokeStyle = y % (gridSize * 4) === 0 ? 'rgba(129, 140, 248, 0.10)' : 'rgba(129, 140, 248, 0.03)';
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // 4. Coordinate Crosshairs at Major Grid Intersections
      const majorStep = gridSize * 4;
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.22)';
      ctx.lineWidth = 1;
      const crossSize = 5;

      for (let mx = majorStep; mx < w; mx += majorStep) {
        for (let my = majorStep; my < h; my += majorStep) {
          ctx.beginPath();
          ctx.moveTo(mx - crossSize, my);
          ctx.lineTo(mx + crossSize, my);
          ctx.moveTo(mx, my - crossSize);
          ctx.lineTo(mx, my + crossSize);
          ctx.stroke();
        }
      }

      // 5. Update and Draw Luminous Constellation Particles
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x = w;
        if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h;
        if (p.y > h) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = p.color + p.alpha + ')';
        ctx.shadowBlur = 12;
        ctx.shadowColor = '#818cf8';
        ctx.fill();
        ctx.shadowBlur = 0;

        // Neural links between nearby particles (purple-blue gradient)
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const dist = Math.hypot(dx, dy);
          if (dist < 140) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(129, 140, 248, ${0.18 * (1 - dist / 140)})`;
            ctx.lineWidth = 0.7;
            ctx.stroke();
          }
        }
      }

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animId);
    };
  }, []);

  return <canvas ref={canvasRef} className="cyber-bg-canvas" />;
}
