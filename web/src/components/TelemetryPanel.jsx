/**
 * Telemetry Panel — Real-time performance gauges and system status indicators.
 */
export default function TelemetryPanel({ telemetry, onToggleSAHI, onToggleTracking }) {
  if (!telemetry) {
    return (
      <div className="glass-card" style={styles.container}>
        <div style={styles.inner}>
          <span style={styles.loadingText}>Connecting to AURA telemetry...</span>
          <div className="skeleton" style={{ width: '180px', height: '6px' }} />
        </div>
      </div>
    );
  }

  const fpsColor = telemetry.fps >= 25 ? 'var(--accent-emerald)' :
                   telemetry.fps >= 15 ? 'var(--accent-amber)' : 'var(--accent-red)';

  const latencyColor = telemetry.inference_latency_ms <= 50 ? 'var(--accent-emerald)' :
                       telemetry.inference_latency_ms <= 150 ? 'var(--accent-amber)' : 'var(--accent-red)';

  return (
    <div className="glass-card" style={styles.container}>
      <div style={styles.inner}>

        {/* FPS Gauge */}
        <div style={styles.metric}>
          <div style={styles.metricValue}>
            <svg width="38" height="38" viewBox="0 0 38 38">
              <circle cx="19" cy="19" r="16" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
              <circle
                cx="19" cy="19" r="16" fill="none"
                stroke={fpsColor}
                strokeWidth="3"
                strokeDasharray={`${Math.min(100, (telemetry.fps / 60) * 100)} 100`}
                strokeLinecap="round"
                transform="rotate(-90 19 19)"
                style={{ filter: `drop-shadow(0 0 4px ${fpsColor})`, transition: 'stroke-dasharray 0.5s ease' }}
              />
              <text x="19" y="22" textAnchor="middle" fill={fpsColor}
                    style={{ fontFamily: 'var(--font-mono)', fontSize: '9px', fontWeight: 600 }}>
                {telemetry.fps.toFixed(0)}
              </text>
            </svg>
          </div>
          <span style={styles.metricLabel}>FPS</span>
        </div>

        {/* Inference Latency */}
        <div style={styles.metric}>
          <span style={{ ...styles.metricNumber, color: latencyColor }}>
            {telemetry.inference_latency_ms.toFixed(0)}
          </span>
          <span style={styles.metricUnit}>ms</span>
          <span style={styles.metricLabel}>Latency</span>
        </div>

        {/* Detections */}
        <div style={styles.metric}>
          <span style={{ ...styles.metricNumber, color: 'var(--accent-cyan)' }}>
            {telemetry.detection_count}
          </span>
          <span style={styles.metricLabel}>Detections</span>
        </div>

        {/* Active Tracks */}
        <div style={styles.metric}>
          <span style={{ ...styles.metricNumber, color: 'var(--accent-emerald)' }}>
            {telemetry.active_tracks}
          </span>
          <span style={styles.metricLabel}>Tracks</span>
        </div>

        {/* OCR Texts */}
        <div style={styles.metric}>
          <span style={{ ...styles.metricNumber, color: 'var(--accent-violet)' }}>
            {telemetry.ocr_text_count}
          </span>
          <span style={styles.metricLabel}>OCR</span>
        </div>

        <div style={styles.divider} />

        {/* SAHI Toggle */}
        <button
          style={{
            ...styles.toggleBtn,
            ...(telemetry.sahi_enabled ? styles.toggleActive : {}),
          }}
          onClick={onToggleSAHI}
          title="Toggle SAHI Sliced Inference"
          id="toggle-sahi-btn"
        >
          <span style={styles.toggleDot(telemetry.sahi_enabled)} />
          SAHI
        </button>

        {/* Tracking Toggle */}
        <button
          style={{
            ...styles.toggleBtn,
            ...(telemetry.tracking_enabled ? styles.toggleActive : {}),
          }}
          onClick={onToggleTracking}
          title="Toggle Multi-Object Tracking"
          id="toggle-tracking-btn"
        >
          <span style={styles.toggleDot(telemetry.tracking_enabled)} />
          Tracking
        </button>

        {/* ANN Version */}
        <div style={styles.metric}>
          <span className="badge badge-emerald" style={{ fontSize: '0.6rem' }}>
            {telemetry.ann_version || 'ANN'}
          </span>
          <span style={styles.metricLabel}>Reliability</span>
        </div>

        {/* Frame Counter */}
        <div style={styles.metric}>
          <span style={{ ...styles.metricNumber, color: 'var(--text-muted)', fontSize: '0.75rem' }}>
            #{telemetry.frame_count}
          </span>
          <span style={styles.metricLabel}>Frame</span>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    padding: 0,
    overflow: 'hidden',
  },
  inner: {
    display: 'flex',
    alignItems: 'center',
    gap: '20px',
    padding: '10px 20px',
    overflowX: 'auto',
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  loadingText: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.75rem',
    color: 'var(--text-muted)',
  },
  metric: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '2px',
    minWidth: '48px',
  },
  metricValue: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  metricNumber: {
    fontFamily: 'var(--font-mono)',
    fontSize: '1.1rem',
    fontWeight: 600,
    lineHeight: 1,
  },
  metricUnit: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.6rem',
    color: 'var(--text-muted)',
    marginTop: '-2px',
  },
  metricLabel: {
    fontFamily: 'var(--font-display)',
    fontSize: '0.58rem',
    color: 'var(--text-muted)',
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
  },
  divider: {
    width: '1px',
    height: '32px',
    background: 'rgba(255,255,255,0.08)',
    flexShrink: 0,
  },
  toggleBtn: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.68rem',
    fontWeight: 500,
    padding: '5px 12px',
    borderRadius: 'var(--radius-full)',
    border: '1px solid rgba(255,255,255,0.1)',
    background: 'rgba(255,255,255,0.04)',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    transition: 'all var(--transition-fast)',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  toggleActive: {
    borderColor: 'rgba(0, 240, 255, 0.4)',
    background: 'rgba(0, 240, 255, 0.1)',
    color: 'var(--accent-cyan)',
    boxShadow: '0 0 12px rgba(0, 240, 255, 0.15)',
  },
  toggleDot: (active) => ({
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: active ? 'var(--accent-cyan)' : 'var(--text-muted)',
    boxShadow: active ? '0 0 6px var(--accent-cyan)' : 'none',
    transition: 'all var(--transition-fast)',
  }),
};
