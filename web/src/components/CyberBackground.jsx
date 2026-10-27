import { useEffect, useRef } from 'react';

/**
 * CyberBackground — AURA V2 High-Contrast Aerospace Grid & Neural Particles.
 * Renders the iconic electric cyan grid with crosshair coordinate nodes,
 * luminous constellation vectors, and floating data particles directly
 * inspired by the AURA OS splash screen.
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
    const count = 50;
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.45,
        vy: (Math.random() - 0.5) * 0.45,
        radius: Math.random() * 1.8 + 0.8,
        color: Math.random() > 0.4 ? 'rgba(0, 240, 255, ' : 'rgba(0, 255, 157, ',
        alpha: Math.random() * 0.6 + 0.2,
      });
    }

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const w = canvas.width;
      const h = canvas.height;

      // 1. Deep Space Cyber Abyss Vignette
      const bgGrad = ctx.createRadialGradient(w / 2, h / 2, 50, w / 2, h / 2, Math.max(w, h));
      bgGrad.addColorStop(0, '#030a1c');
      bgGrad.addColorStop(0.6, '#020612');
      bgGrad.addColorStop(1, '#01030a');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);

      // 2. Crisp Aerospace Tactical Grid (Matches Splash Screen)
      const gridSize = 60;
      ctx.lineWidth = 0.8;

      for (let x = 0; x < w; x += gridSize) {
        ctx.strokeStyle = x % (gridSize * 4) === 0 ? 'rgba(0, 240, 255, 0.12)' : 'rgba(0, 240, 255, 0.04)';
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }

      for (let y = 0; y < h; y += gridSize) {
        ctx.strokeStyle = y % (gridSize * 4) === 0 ? 'rgba(0, 240, 255, 0.12)' : 'rgba(0, 240, 255, 0.04)';
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // 3. Coordinate Crosshairs at Major Grid Intersections
      const majorStep = gridSize * 4; // every 240px
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.28)';
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

      // 4. Update and Draw Luminous Constellation Particles
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
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#00f0ff';
        ctx.fill();
        ctx.shadowBlur = 0;

        // Neural links between nearby particles
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const dist = Math.hypot(dx, dy);
          if (dist < 130) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(0, 240, 255, ${0.22 * (1 - dist / 130)})`;
            ctx.lineWidth = 0.8;
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
