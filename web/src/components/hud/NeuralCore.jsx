import React from 'react';

/**
 * NeuralCore — Signature AURA V2 Circular Reactor Core.
 * Reusable animated visual identity displaying central cognitive reactor
 * surrounded by 5 active subsystem nodes:
 * VISION, SPATIAL, REASON, AUDIO, MEMORY.
 */
export default function NeuralCore({
  size = 120,
  activeSubsystems = {
    vision: true,
    spatial: true,
    reason: false,
    audio: false,
    memory: true,
  },
  statusText = 'AURA CORE',
  isThinking = false,
  className = '',
}) {
  const subsystems = [
    { id: 'vision', label: 'VISION', angle: -90, color: 'var(--aura-cyan)' },
    { id: 'spatial', label: 'SPATIAL', angle: -18, color: 'var(--aura-cyan)' },
    { id: 'reason', label: 'REASON', angle: 54, color: 'var(--aura-purple)' },
    { id: 'audio', label: 'AUDIO', angle: 126, color: 'var(--aura-emerald)' },
    { id: 'memory', label: 'MEMORY', angle: 198, color: 'var(--aura-amber)' },
  ];

  const radius = size * 0.42;
  const center = size / 2;

  return (
    <div
      className={`aura-neural-core-wrapper ${className} ${isThinking ? 'core-thinking' : ''}`}
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <filter id="coreGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* ── Outer Concentric Rotating Orbit Rings ── */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="rgba(0, 240, 255, 0.2)"
          strokeWidth="1"
          strokeDasharray="4 6"
          className="orbit-ring-outer"
        />
        <circle
          cx={center}
          cy={center}
          r={radius * 0.75}
          fill="none"
          stroke="rgba(0, 240, 255, 0.35)"
          strokeWidth="1.2"
          strokeDasharray="12 18"
          className="orbit-ring-inner"
        />

        {/* ── Subsystem Nodes & Linking Vector Rays ── */}
        {subsystems.map((sub) => {
          const rad = (sub.angle * Math.PI) / 180;
          const x = center + radius * Math.cos(rad);
          const y = center + radius * Math.sin(rad);
          const isActive = activeSubsystems[sub.id];

          return (
            <g key={sub.id} className={`subsystem-group ${isActive ? 'sub-active' : 'sub-idle'}`}>
              {/* Connecting Data Vector */}
              <line
                x1={center}
                y1={center}
                x2={x}
                y2={y}
                stroke={isActive ? sub.color : 'rgba(255, 255, 255, 0.1)'}
                strokeWidth={isActive ? '1.5' : '0.8'}
                strokeDasharray={isActive ? '3 3' : 'none'}
                className={isActive ? 'data-vector-pulse' : ''}
              />

              {/* Subsystem Node Circle */}
              <circle
                cx={x}
                cy={y}
                r={isActive ? 4 : 2.5}
                fill={isActive ? sub.color : 'rgba(255, 255, 255, 0.2)'}
                filter={isActive ? 'url(#coreGlow)' : 'none'}
              />
            </g>
          );
        })}

        {/* ── Central Reactor Core ── */}
        <circle
          cx={center}
          cy={center}
          r={size * 0.22}
          fill="rgba(4, 8, 16, 0.85)"
          stroke="var(--aura-cyan)"
          strokeWidth="1.5"
          filter="url(#coreGlow)"
        />
        <circle
          cx={center}
          cy={center}
          r={size * 0.12}
          fill="var(--aura-cyan)"
          className="central-reactor-pulse"
        />
      </svg>

      {/* Central Label Overlay */}
      <span className="neural-core-label">{statusText}</span>
    </div>
  );
}
