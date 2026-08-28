import React from 'react';
import TacticalRadarCanvas from './TacticalRadarCanvas';
import { soundFX } from '../utils/audioFx';

/**
 * EntityMatrix — Left Wing Tactical Intelligence Hub
 * featuring live Holographic Radar, spatial object listing, reliability bars, and quick lock actions.
 */
export default function EntityMatrix({ scene, pointedTarget, onSelectEntity }) {
  const entities = scene?.entities || [];

  const handleEntityClick = (entity) => {
    soundFX.playLockOn();
    if (onSelectEntity) {
      onSelectEntity(entity);
    }
  };

  const getEntityIcon = (name) => {
    const map = {
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
      backpack: '🎒',
      chair: '🪑',
    };
    return map[name?.toLowerCase()] || '📦';
  };

  return (
    <div className="entity-matrix-panel glass-panel">
      {/* ── Panel Header ── */}
      <div className="panel-hdr">
        <div className="panel-hdr-left">
          <span className="panel-icon">🛰️</span>
          <span className="panel-title">SPATIAL SCENE MATRIX</span>
        </div>
        <span className="badge badge-cyan">{entities.length} OBJECTS</span>
      </div>

      {/* ── Tactical Holographic Radar ── */}
      <TacticalRadarCanvas
        scene={scene}
        pointedTarget={pointedTarget}
        onSelectEntity={handleEntityClick}
      />

      {/* ── Real-Time Entity Stream List ── */}
      <div className="entity-stream-section">
        <div className="stream-header">
          <span className="stream-title">DETECTED ENTITY REGISTRY</span>
          <span className="stream-subtitle">TRACKED IN CAMERA VIEW</span>
        </div>

        <div className="entity-stream-list">
          {entities.length === 0 ? (
            <div className="stream-empty">
              <span className="empty-beacon">⬢</span>
              <p>NO ACTIVE OBJECTS DETECTED</p>
              <span className="empty-sub">Point camera at objects to detect</span>
            </div>
          ) : (
            entities.map((entity, idx) => {
              const isLocked = pointedTarget && entity.class_name?.toLowerCase() === pointedTarget.toLowerCase();
              const conf = Math.round((entity.confidence || 0.9) * 100);

              return (
                <div
                  key={idx}
                  className={`entity-item glass-card animate-fade-in ${isLocked ? 'entity-item-locked' : ''}`}
                  onClick={() => handleEntityClick(entity)}
                >
                  <div className="entity-item-top">
                    <div className="entity-item-name-group">
                      <span className="entity-icon">{getEntityIcon(entity.class_name)}</span>
                      <div>
                        <span className="entity-name">{entity.class_name.toUpperCase()}</span>
                        {entity.track_id != null && (
                          <span className="entity-track-id">#ID {entity.track_id}</span>
                        )}
                      </div>
                    </div>
                    <span className={`entity-conf ${conf >= 80 ? 'conf-high' : 'conf-mid'}`}>
                      {conf}%
                    </span>
                  </div>

                  {/* Confidence Progress Bar */}
                  <div className="entity-conf-bar-bg">
                    <div
                      className="entity-conf-bar-fill"
                      style={{ width: `${conf}%`, backgroundColor: isLocked ? '#00f0ff' : '#00ff9d' }}
                    />
                  </div>

                  {/* Spatial Tag & OCR Texts */}
                  <div className="entity-meta-row">
                    <span className="spatial-tag">LOC: {entity.spatial_pos || 'CENTER'}</span>
                    {entity.ocr_texts?.length > 0 && (
                      <span className="ocr-preview-tag">
                        TEXT: "{entity.ocr_texts[0]}"
                      </span>
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
