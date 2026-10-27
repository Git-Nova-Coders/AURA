import { useMemo } from 'react';

/**
 * 3×3 Spatial Scene Radar — Holographic radar displaying real-time 3D object positions,
 * radar sweep animation, entity icons, and target lock-on highlights.
 */

const GRID_LABELS = [
  ['top-left', 'top', 'top-right'],
  ['left', 'center', 'right'],
  ['bottom-left', 'bottom', 'bottom-right'],
];

const OBJECT_ICONS = {
  person: '👤',
  face: '😀',
  hand: '✋',
  laptop: '💻',
  notebook: '📓',
  book: '📕',
  smartphone: '📱',
  phone: '📱',
  cup: '☕',
  bottle: '🍶',
  'water bottle': '🍶',
  backpack: '🎒',
  headphones: '🎧',
  glasses: '👓',
  keyboard: '⌨️',
  'computer mouse': '🖱️',
  chair: '🪑',
};

function getObjectIcon(className) {
  return OBJECT_ICONS[className?.toLowerCase()] || '📦';
}

export default function SpatialRadar({ scene, pointedTarget, onSelectEntity }) {
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
    <div className="spatial-radar-card glass-card">
      {/* ── Radar Title ── */}
      <div className="radar-title-row">
        <div className="radar-title-left">
          <span className="radar-icon animate-pulse">🛰️</span>
          <span className="panel-title">Spatial Radar</span>
        </div>
        <span className="badge badge-emerald">{totalEntities} Tracked</span>
      </div>

      {/* ── 3x3 Radar Grid ── */}
      <div className="radar-grid-wrapper">
        <div className="radar-sweep-beam" />
        <div className="radar-grid">
          {GRID_LABELS.map((row, ri) =>
            row.map((cellLabel, ci) => {
              const entities = entityMap[cellLabel] || [];
              const hasItems = entities.length > 0;
              const hasLockedTarget = entities.some(
                (e) => pointedTarget && e.class_name?.toLowerCase() === pointedTarget.toLowerCase()
              );

              return (
                <div
                  key={`${ri}-${ci}`}
                  className={`radar-cell ${hasItems ? 'radar-cell-active' : ''} ${
                    cellLabel === 'center' ? 'radar-cell-center' : ''
                  } ${hasLockedTarget ? 'radar-cell-locked' : ''}`}
                >
                  <span className="radar-cell-tag">{cellLabel.replace('-', ' ')}</span>
                  <div className="radar-blips">
                    {entities.slice(0, 3).map((e, idx) => (
                      <span
                        key={idx}
                        className={`radar-blip ${
                          pointedTarget && e.class_name?.toLowerCase() === pointedTarget.toLowerCase()
                            ? 'blip-locked'
                            : ''
                        }`}
                        title={`${e.class_name} #${e.track_id ?? '?'}`}
                        onClick={() => onSelectEntity && onSelectEntity(e)}
                      >
                        {getObjectIcon(e.class_name)}
                      </span>
                    ))}
                    {entities.length > 3 && (
                      <span className="radar-blip-more">+{entities.length - 3}</span>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
