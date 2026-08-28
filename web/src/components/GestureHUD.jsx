import React from 'react';

/**
 * GestureHUD — Floating glassmorphism HUD badge directly on the Live Feed
 * showing the active recognized 3D hand gesture, tracking mode, and a button to view the guide.
 */
export default function GestureHUD({ telemetry, onOpenGuide }) {
  const activeGesture = telemetry?.active_gesture || 'none';
  const gestureMode = telemetry?.gesture_mode || 'ALL_OBJECTS';

  const gestureIcons = {
    open_palm: '🖐️',
    peace_sign: '✌️',
    pointing: '👉',
    pinch: '👌',
    thumbs_up: '👍',
    thumbs_down: '👎',
    fist: '✊',
    rock_on: '🤘',
    call_me: '🤙',
    none: '✋',
  };

  const gestureLabels = {
    open_palm: 'OPEN PALM (Clean View)',
    peace_sign: 'PEACE SIGN (Reset View)',
    pointing: 'POINTING (Target Raycast)',
    pinch: 'PINCH (Inspect Target)',
    thumbs_up: 'THUMBS UP (Snapshot Saved)',
    thumbs_down: 'THUMBS DOWN (Deselect)',
    fist: 'FIST (Freeze Overlay)',
    rock_on: 'ROCK ON (SAHI High-Res)',
    call_me: 'CALL ME (Voice Query)',
    none: 'AWAITING GESTURE',
  };

  const icon = gestureIcons[activeGesture] || '✋';
  const label = gestureLabels[activeGesture] || activeGesture.toUpperCase();
  const isRecognized = activeGesture !== 'none';

  return (
    <div className="gesture-hud-container">
      {/* ── Active 3D Gesture Pill ── */}
      <div className={`gesture-pill glass-card ${isRecognized ? 'gesture-pill-active' : ''}`}>
        <span className="gesture-icon-badge">{icon}</span>
        <div className="gesture-meta">
          <span className="gesture-title">{label}</span>
          <span className="gesture-mode-tag">MODE: {gestureMode}</span>
        </div>
      </div>

      {/* ── Quick Cheat Sheet Trigger ── */}
      <button
        className="gesture-guide-btn glass-card"
        onClick={onOpenGuide}
        title="View 21-Landmark Hand Gesture Guide"
        id="btn-gesture-guide"
      >
        <span className="guide-icon">📖</span>
        <span className="guide-label">Gesture Guide</span>
      </button>
    </div>
  );
}
