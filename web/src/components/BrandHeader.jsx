import { useState, useEffect } from 'react';

/**
 * AURA Brand Header — Glowing logo, system version, and connection status.
 */
export default function BrandHeader({ isConnected, telemetry }) {
  const [glowPhase, setGlowPhase] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => setGlowPhase((p) => (p + 1) % 360), 50);
    return () => clearInterval(interval);
  }, []);

  const glowColor = `hsl(${180 + Math.sin(glowPhase * 0.02) * 20}, 100%, 60%)`;

  return (
    <header className="brand-header glass-card" style={styles.header}>
      {/* Top gradient border */}
      <div style={styles.topBorder} />

      <div style={styles.content}>
        {/* Logo */}
        <div style={styles.logoGroup}>
          <div style={{ ...styles.logoIcon, textShadow: `0 0 20px ${glowColor}` }}>◉</div>
          <div>
            <h1 style={styles.title}>
              <span style={{ ...styles.titleGlow, color: glowColor }}>AURA</span>
            </h1>
            <p style={styles.subtitle}>Adaptive Understanding & Reasoning Architecture</p>
          </div>
        </div>

        {/* Status Badges */}
        <div style={styles.badges}>
          <span className="badge badge-cyan" style={styles.versionBadge}>v0.9.0</span>

          {telemetry && (
            <span className="badge" style={{
              ...styles.fpsBadge,
              background: telemetry.fps >= 25 ? 'var(--accent-emerald-dim)' :
                           telemetry.fps >= 15 ? 'rgba(255,176,32,0.15)' : 'rgba(255,64,96,0.15)',
              color: telemetry.fps >= 25 ? 'var(--accent-emerald)' :
                     telemetry.fps >= 15 ? 'var(--accent-amber)' : 'var(--accent-red)',
              borderColor: telemetry.fps >= 25 ? 'rgba(0,229,153,0.25)' :
                           telemetry.fps >= 15 ? 'rgba(255,176,32,0.25)' : 'rgba(255,64,96,0.25)',
            }}>
              {telemetry.fps.toFixed(1)} FPS
            </span>
          )}

          <div style={styles.connectionStatus}>
            <div style={{
              ...styles.statusDot,
              backgroundColor: isConnected ? '#00e599' : '#ff4060',
              boxShadow: isConnected ? '0 0 8px #00e599' : '0 0 8px #ff4060',
            }} />
            <span style={styles.statusText}>
              {isConnected ? 'Live' : 'Offline'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}

const styles = {
  header: {
    position: 'relative',
    overflow: 'hidden',
    padding: 0,
  },
  topBorder: {
    height: '2px',
    background: 'linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-emerald), var(--accent-violet), transparent)',
    opacity: 0.8,
  },
  content: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 20px',
  },
  logoGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '14px',
  },
  logoIcon: {
    fontSize: '2rem',
    color: 'var(--accent-cyan)',
    transition: 'text-shadow 0.3s ease',
  },
  title: {
    fontFamily: 'var(--font-display)',
    fontSize: '1.5rem',
    fontWeight: 800,
    letterSpacing: '0.15em',
    margin: 0,
    lineHeight: 1.1,
  },
  titleGlow: {
    transition: 'color 0.5s ease',
  },
  subtitle: {
    fontFamily: 'var(--font-body)',
    fontSize: '0.68rem',
    color: 'var(--text-muted)',
    letterSpacing: '0.06em',
    marginTop: '2px',
  },
  badges: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  versionBadge: {
    fontFamily: 'var(--font-mono)',
  },
  fpsBadge: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.7rem',
    padding: '2px 8px',
    borderRadius: 'var(--radius-full)',
    border: '1px solid',
  },
  connectionStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  statusDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    transition: 'all 0.3s ease',
  },
  statusText: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.7rem',
    color: 'var(--text-muted)',
  },
};
