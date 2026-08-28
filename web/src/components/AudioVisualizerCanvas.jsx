import { useEffect, useRef } from 'react';

/**
 * AudioVisualizerCanvas — Dynamic multi-frequency audio equalizer waveform canvas
 * with pulsating neon ripples for real-time speech and voice assistant activity.
 */
export default function AudioVisualizerCanvas({ isActive = false }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let phase = 0;

    const bars = 18;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const barWidth = (w - (bars - 1) * 3) / bars;
      phase += isActive ? 0.08 : 0.02;

      for (let i = 0; i < bars; i++) {
        // Multi-harmonic sine frequency amplitude
        const multiplier = isActive ? 1.0 : 0.25;
        const s1 = Math.sin(phase + i * 0.45);
        const s2 = Math.cos(phase * 1.3 + i * 0.3);
        const amp = (Math.abs(s1 * 0.6 + s2 * 0.4) * (h * 0.85) + 3) * multiplier;

        const x = i * (barWidth + 3);
        const y = (h - amp) / 2;

        const grad = ctx.createLinearGradient(0, y, 0, y + amp);
        if (isActive) {
          grad.addColorStop(0, '#00f0ff');
          grad.addColorStop(0.5, '#00ff9d');
          grad.addColorStop(1, '#a855f7');
        } else {
          grad.addColorStop(0, 'rgba(0, 240, 255, 0.4)');
          grad.addColorStop(1, 'rgba(0, 240, 255, 0.1)');
        }

        ctx.fillStyle = grad;
        ctx.shadowColor = isActive ? '#00f0ff' : 'transparent';
        ctx.shadowBlur = isActive ? 10 : 0;

        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, Math.max(3, amp), [2]);
        ctx.fill();
      }

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [isActive]);

  return <canvas ref={canvasRef} width={130} height={26} className="voice-wave-canvas" />;
}
