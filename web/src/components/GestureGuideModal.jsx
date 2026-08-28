import React from 'react';

/**
 * GestureGuideModal — Interactive cybernetic cheat-sheet modal
 * displaying all 9 3D hand gestures, kinematics rules, and real-world system actions.
 */
export default function GestureGuideModal({ isOpen, onClose, activeGesture }) {
  if (!isOpen) return null;

  const gestures = [
    {
      id: 'open_palm',
      icon: '🖐️',
      name: 'Open Palm',
      rule: 'All 5 fingers extended (θ > 140°)',
      action: 'Clean Camera View: Hides all bounding boxes for pure video.',
      badge: 'VIEW',
    },
    {
      id: 'peace_sign',
      icon: '✌️',
      name: 'Peace Sign ("V")',
      rule: 'Index + Middle extended, Ring + Pinky curled',
      action: 'Reset View: Restores all detections and tracking overlays.',
      badge: 'RESET',
    },
    {
      id: 'pointing',
      icon: '👉',
      name: 'Pointing Finger',
      rule: 'Index extended (θ > 140°), other fingers curled',
      action: 'Laser Lock-On: Projects raycast laser to lock target object with crosshair reticle.',
      badge: 'LOCK',
    },
    {
      id: 'pinch',
      icon: '👌',
      name: 'Pinch (Air Click)',
      rule: 'Thumb Tip + Index Tip Euclidean 3D distance < 0.055',
      action: 'Inspect Target: Triggers multimodal AI reasoning & voice explanation of targeted item.',
      badge: 'INSPECT',
    },
    {
      id: 'thumbs_up',
      icon: '👍',
      name: 'Thumbs Up',
      rule: 'Thumb pointing upward (+Y), 4 fingers curled',
      action: 'Save Snapshot: Captures high-res JPEG to captures/ directory.',
      badge: 'CAPTURE',
    },
    {
      id: 'thumbs_down',
      icon: '👎',
      name: 'Thumbs Down',
      rule: 'Thumb pointing downward (-Y), 4 fingers curled',
      action: 'Deselect Target: Clears current object lock and resets focus.',
      badge: 'CLEAR',
    },
    {
      id: 'fist',
      icon: '✊',
      name: 'Fist',
      rule: 'All 5 fingers curled into compact fist',
      action: 'Freeze Frame: Freezes/unfreezes current overlay and detections in place.',
      badge: 'FREEZE',
    },
    {
      id: 'rock_on',
      icon: '🤘',
      name: 'Rock On / Horns',
      rule: 'Index + Pinky extended, Middle + Ring curled',
      action: 'Toggle SAHI: Switches SAHI sliced high-resolution inference ON / OFF.',
      badge: 'SAHI',
    },
    {
      id: 'call_me',
      icon: '🤙',
      name: 'Call Me / Shaka',
      rule: 'Thumb + Pinky extended, Middle 3 fingers curled',
      action: 'Voice Assistant: Activates speech listening microphone for conversational queries.',
      badge: 'VOICE',
    },
  ];

  return (
    <div className="modal-backdrop animate-fade-in" onClick={onClose}>
      <div className="gesture-modal glass-card animate-slide-down" onClick={(e) => e.stopPropagation()}>
        {/* ── Modal Header ── */}
        <div className="gesture-modal-header">
          <div className="modal-title-wrap">
            <span className="modal-icon">🖐️</span>
            <div>
              <h2 className="modal-title">21-Landmark Gesture Control Engine</h2>
              <p className="modal-subtitle">Sub-millimeter 3D Kinematic Hand Tracking & Action System</p>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} id="btn-close-modal">✕</button>
        </div>

        {/* ── Gesture Grid ── */}
        <div className="gesture-grid">
          {gestures.map((g) => {
            const isCurrent = activeGesture === g.id;
            return (
              <div
                key={g.id}
                className={`gesture-card glass-card ${isCurrent ? 'gesture-card-current' : ''}`}
              >
                <div className="gesture-card-top">
                  <span className="gesture-card-icon">{g.icon}</span>
                  <span className="badge badge-cyan">{g.badge}</span>
                </div>
                <h3 className="gesture-card-name">{g.name}</h3>
                <p className="gesture-card-rule"><strong>Pose:</strong> {g.rule}</p>
                <p className="gesture-card-action"><strong>Action:</strong> {g.action}</p>
                {isCurrent && <div className="gesture-live-tag">● CURRENTLY ACTIVE</div>}
              </div>
            );
          })}
        </div>

        {/* ── Modal Footer ── */}
        <div className="gesture-modal-footer">
          <span className="modal-hint">💡 Perform gestures in front of the camera to execute real-time actions and trigger HUD notifications.</span>
          <button className="btn-primary" onClick={onClose}>Got It</button>
        </div>
      </div>
    </div>
  );
}
