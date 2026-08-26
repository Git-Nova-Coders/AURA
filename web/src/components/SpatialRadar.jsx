import { useMemo } from 'react';

/**
 * 3×3 Spatial Scene Radar — Top-down grid visualization of object positions.
 */

const GRID_LABELS = [
  ['top-left', 'top', 'top-right'],
  ['left', 'center', 'right'],
  ['bottom-left', 'bottom', 'bottom-right'],
];

const OBJECT_ICONS = {
  person: '👤',
  laptop: '💻',
  notebook: '📓',
  book: '📕',
  smartphone: '📱',
  phone: '📱',
  pen: '🖊️',
  pencil: '✏️',
  cup: '☕',
  bottle: '🍶',
  'water bottle': '🍶',
  backpack: '🎒',
  handbag: '👜',
  headphones: '🎧',
  glasses: '👓',
  keyboard: '⌨️',
  'computer mouse': '🖱️',
  chair: '🪑',
  desk: '🪑',
  'wrist watch': '⌚',
};

function getObjectIcon(className) {
  return OBJECT_ICONS[className?.toLowerCase()] || '📦';
}

export default function SpatialRadar({ scene }) {
  // Group entities by spatial region
  const entityMap = useMemo(() => {
    const map = {};
    GRID_LABELS.flat().forEach((label) => { map[label] = []; });

    if (scene?.entities) {
      scene.entities.forEach((entity) => {
        const region = entity.spatial_pos || 'center';
        if (map[region]) {
          map[region].push(entity);
        } else {
          map['center'].push(entity);
        }
      });
    }
    return map;
  }, [scene]);

  const totalEntities = scene?.entity_count || 0;

  return (
    <div className="glass-card" style={styles.container}>
      <div style={styles.titleRow}>
        <span className="panel-title"><span className="icon">🛰️</span> Spatial Radar</span>
        <span className="badge badge-emerald">{totalEntities} active</span>
      </div>

      <div style={styles.grid}>
        {GRID_LABELS.map((row, ri) =>
          row.map((cellLabel, ci) => {
            const entities = entityMap[cellLabel] || [];
            const hasItems = entities.length > 0;
            return (
              <div
                key={`${ri}-${ci}`}
                style={{
                  ...styles.cell,
                  ...(hasItems ? styles.cellActive : {}),
                  ...(cellLabel === 'center' ? styles.cellCenter : {}),
                }}
              >
                <span style={styles.cellLabel}>{cellLabel.replace('-', '\n')}</span>
                <div style={styles.entityDots}>
                  {entities.slice(0, 3).map((e, i) => (
                    <span
                      key={i}
                      style={styles.entityDot}
                      title={`${e.class_name} #${e.track_id ?? '?'}`}
                      className={hasItems ? 'animate-fade-in' : ''}
                    >
                      {getObjectIcon(e.class_name)}
                    </span>
                  ))}
                  {entities.length > 3 && (
                    <span style={styles.moreCount}>+{entities.length - 3}</span>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Spatial relations */}
      {scene?.relations?.length > 0 && (
        <div style={styles.relations}>
          {scene.relations.slice(0, 2).map((rel, i) => (
            <p key={i} style={styles.relationText}>{rel.sentence}</p>
          ))}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    padding: 0,
    overflow: 'hidden',
  },
  titleRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '0 8px 0 0',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gridTemplateRows: 'repeat(3, 1fr)',
    gap: '4px',
    padding: '0 var(--space-md) var(--space-sm)',
    aspectRatio: '1',
    maxHeight: '220px',
  },
  cell: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '4px',
    background: 'rgba(15, 19, 28, 0.5)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid rgba(255,255,255,0.04)',
    padding: '4px',
    transition: 'all var(--transition-base)',
    minHeight: '50px',
  },
  cellActive: {
    background: 'rgba(0, 240, 255, 0.06)',
    border: '1px solid rgba(0, 240, 255, 0.2)',
    boxShadow: '0 0 10px rgba(0, 240, 255, 0.08)',
  },
  cellCenter: {
    background: 'rgba(0, 229, 153, 0.04)',
    border: '1px solid rgba(0, 229, 153, 0.15)',
  },
  cellLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.55rem',
    color: 'var(--text-muted)',
    textAlign: 'center',
    opacity: 0.6,
    lineHeight: 1.2,
    whiteSpace: 'pre-line',
  },
  entityDots: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '2px',
    justifyContent: 'center',
  },
  entityDot: {
    fontSize: '0.9rem',
    cursor: 'default',
  },
  moreCount: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.6rem',
    color: 'var(--accent-cyan)',
  },
  relations: {
    padding: '4px var(--space-md) var(--space-sm)',
    borderTop: '1px solid rgba(255,255,255,0.04)',
  },
  relationText: {
    fontFamily: 'var(--font-body)',
    fontSize: '0.7rem',
    color: 'var(--text-muted)',
    marginTop: '2px',
  },
};
