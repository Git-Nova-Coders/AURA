import React from 'react';

/**
 * HudGrid — Futuristic Perspective Wireframe Grid Overlay.
 * Rendered at 3–8% opacity to provide spatial depth over the camera
 * without competing with detections.
 */
export default function HudGrid({ opacity = 0.05 }) {
  return (
    <div className="aura-hud-grid-container" style={{ opacity }}>
      <svg
        className="aura-perspective-grid-svg"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 1000 600"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="gridGrad" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="var(--aura-cyan)" stopOpacity="0.8" />
            <stop offset="60%" stopColor="var(--aura-cyan)" stopOpacity="0.2" />
            <stop offset="100%" stopColor="var(--aura-cyan)" stopOpacity="0.0" />
          </linearGradient>
        </defs>

        {/* Perspective Vanishing Floor Grid */}
        <g stroke="url(#gridGrad)" strokeWidth="0.8" fill="none">
          {/* Horizontal Depth Rings / Latitude Lines */}
          <line x1="100" y1="580" x2="900" y2="580" />
          <line x1="200" y1="520" x2="800" y2="520" />
          <line x1="280" y1="470" x2="720" y2="470" />
          <line x1="340" y1="430" x2="660" y2="430" />
          <line x1="390" y1="400" x2="610" y2="400" />
          <line x1="430" y1="380" x2="570" y2="380" />

          {/* Perspective Ray Lines meeting toward vanishing point (500, 360) */}
          <line x1="100" y1="580" x2="430" y2="380" />
          <line x1="250" y1="580" x2="455" y2="380" />
          <line x1="400" y1="580" x2="480" y2="380" />
          <line x1="500" y1="580" x2="500" y2="380" />
          <line x1="600" y1="580" x2="520" y2="380" />
          <line x1="750" y1="580" x2="545" y2="380" />
          <line x1="900" y1="580" x2="570" y2="380" />
        </g>

        {/* Outer Perspective Alignment Brackets */}
        <g stroke="var(--aura-cyan)" strokeWidth="1" strokeOpacity="0.3" fill="none">
          <path d="M 50,50 L 80,50 M 50,50 L 50,80" />
          <path d="M 950,50 L 920,50 M 950,50 L 950,80" />
          <path d="M 50,550 L 80,550 M 50,550 L 50,520" />
          <path d="M 950,550 L 920,550 M 950,550 L 950,520" />
        </g>
      </svg>
    </div>
  );
}
