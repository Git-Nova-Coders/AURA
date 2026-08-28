import { useEffect, useRef } from 'react';

/**
 * OscilloscopeCanvas — Real-time GPU compute & inference latency oscilloscope waveform.
 */
export default function OscilloscopeCanvas({ value = 12.0, color = '#00f0ff' }) {
  const canvasRef = useRef(null);
  const historyRef = useRef(new Array(40).fill(12.0));

  useEffect(() => {
    historyRef.current.push(value);
    if (historyRef.current.length > 40) {
      historyRef.current.shift();
    }
  }, [value]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // Grid background line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();

      const data = historyRef.current;
      const step = w / (data.length - 1);

      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        const val = data[i];
        // Normalized height between 0 and 40 ms
        const y = h - Math.min(h, Math.max(4, (val / 35.0) * h));
        const x = i * step;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Glow fill underneath
      ctx.lineTo(w, h);
      ctx.lineTo(0, h);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, 'rgba(0, 240, 255, 0.2)');
      grad.addColorStop(1, 'rgba(0, 240, 255, 0)');
      ctx.fillStyle = grad;
      ctx.fill();

      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [color]);

  return <canvas ref={canvasRef} width={80} height={24} className="oscilloscope-canvas" />;
}
