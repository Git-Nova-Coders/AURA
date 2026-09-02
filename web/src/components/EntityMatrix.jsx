import React from 'react';
import TacticalRadarCanvas from './TacticalRadarCanvas';
import HudFrame from './hud/HudFrame';
import { soundFX } from '../utils/audioFx';

/**
 * EntityMatrix — AURA V2 Spatial Intelligence Hub.
 * Compact holographic spatial model, environmental radar, and
 * high-density entity track strips.
 */
export default function EntityMatrix({
  scene,
  pointedTarget,
  onSelectEntity,
  onHoverEntity,
  filterMode = 'ALL',
}) {
  const entities = scene?.entities || [];

  const handleEntityClick = (entity) => {
    soundFX.playLockOn();
    if (onSelectEntity) {
      onSelectEntity(entity);
    }
  };

  const getFilterBadge = () => {
    if (filterMode === 'OBJECTS_ONLY') return 'OBJECTS';
    if (filterMode === 'HUMANS_ONLY') return 'HUMANS';
    if (filterMode === 'OFF') return 'MUTED';
    return 'OMNI';
  };

  return (
    <div className="v2-spatial-matrix-container">
      <HudFrame
        sensorId="SPATIAL_RADAR_01"
        coords="FOV: 60° // RNG: 4.5m"
        status={`${entities.length} DETECTED`}
        className="v2-spatial-hud-frame"
      >
        {/* ── Section Header ── */}
        <div className="v2-spatial-header">
          <div className="v2-spatial-title-group">
            <span className="v2-spatial-icon">🛰️</span>
            <div>
              <h3 className="v2-spatial-title">SPATIAL MATRIX</h3>
              <span className="v2-spatial-sub">ENVIRONMENTAL MAP // {getFilterBadge()}</span>
            </div>
          </div>
        </div>

        {/* ── Miniature Spatial Model Radar ── */}
        <TacticalRadarCanvas
          scene={scene}
          pointedTarget={pointedTarget}
          onSelectEntity={handleEntityClick}
          onHoverEntity={onHoverEntity}
        />

        {/* ── Below Radar: Compact Entity Strips ── */}
        <div className="v2-entity-strips-section">
          <div className="v2-strips-header">
            <span className="v2-strips-title">ACTIVE TRACKS</span>
            <span className="v2-strips-count">{entities.length} IN VIEW</span>
          </div>

          <div className="v2-strips-list">
            {entities.length === 0 ? (
              <div className="v2-strips-empty">
                <span className="empty-ring">⬢</span>
                <p>NO DETECTIONS IN FIELD</p>
                <span className="empty-sub">Surveillance active</span>
              </div>
            ) : (
              entities.map((entity, idx) => {
                const isLocked = pointedTarget && entity.class_name?.toLowerCase() === pointedTarget.toLowerCase();
                const conf = Math.round((entity.confidence || 0.9) * 100);
                const trackId = entity.track_id != null ? String(entity.track_id).padStart(2, '0') : String(idx + 1).padStart(2, '0');
                const spatialLoc = entity.spatial_pos || 'CENTER';

                return (
                  <div
                    key={entity.id || idx}
                    className={`v2-track-strip ${isLocked ? 'v2-strip-locked' : ''}`}
                    onClick={() => handleEntityClick(entity)}
                    onMouseEnter={() => onHoverEntity && onHoverEntity(entity)}
                    onMouseLeave={() => onHoverEntity && onHoverEntity(null)}
                  >
                    <div className="v2-strip-indicator">
                      <span className={`strip-dot ${isLocked ? 'dot-amber' : 'dot-cyan'}`} />
                    </div>

                    <div className="v2-strip-info">
                      <div className="v2-strip-line-1">
                        <span className="strip-track-id">TRACK #{trackId}</span>
                        <span className="strip-name">{entity.class_name.toUpperCase()}</span>
                      </div>
                      <div className="v2-strip-line-2">
                        <span className="strip-loc">{spatialLoc}</span>
                        <span className="strip-conf">{conf}%</span>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </HudFrame>
    </div>
  );
}
