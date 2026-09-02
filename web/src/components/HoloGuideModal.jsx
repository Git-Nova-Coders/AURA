import React from 'react';
import { soundFX } from '../utils/audioFx';

/**
 * HoloGuideModal — Interactive Holographic Gesture Command Manual
 * displaying all 9 3D hand gestures with live detection verification.
 */
export default function HoloGuideModal({
  isOpen,
  onClose,
  activeGesture,
  isGesturesArmed = false,
  onToggleGestures,
}) {
  if (!isOpen) return null;

  const gestures = [
    {
      id: 'open_palm',
      icon: '🖐️',
      name: 'OPEN PALM',
      rule: '5 fingers extended (θ > 140°)',
      action: 'Clean Camera View: Hides bounding boxes for pure video.',
      badge: 'VIEW',
    },
    {
      id: 'peace_sign',
      icon: '✌️',
      name: 'VICTORY SIGN ("V")',
      rule: 'Index + Middle extended (θ > 130°), others curled',
      action: 'Capture Snapshot: Saves high-res image to captures/ with shutter FX.',
      badge: 'CAPTURE',
    },
    {
      id: 'pointing',
      icon: '👉',
      name: 'POINTING RAY',
      rule: 'Index extended (θ > 130°), Middle/Ring/Pinky curled',
      action: 'Laser Lock-On: Projects raycast laser to lock target object with reticle.',
      badge: 'LOCK',
    },
    {
      id: 'pinch',
      icon: '👌',
      name: 'PINCH (AIR CLICK)',
      rule: 'Thumb Tip + Index Tip 3D dist < 0.055',
      action: 'Inspect Target: Triggers multimodal AI reasoning & voice explanation.',
      badge: 'INSPECT',
    },
    {
      id: 'thumbs_up',
      icon: '👍',
      name: 'THUMBS UP',
      rule: 'Thumb extended (+Y), 4 fingers curled',
      action: 'Restore All Boxes: Brings back all detections and resets target lock.',
      badge: 'RESET',
    },
    {
      id: 'thumbs_down',
      icon: '👎',
      name: 'THUMBS DOWN',
      rule: 'Thumb downward (-Y), 4 fingers curled',
      action: 'Deselect Target: Clears current object lock and resets focus.',
      badge: 'CLEAR',
    },
    {
      id: 'fist',
      icon: '✊',
      name: 'FIST',
      rule: 'All 5 fingers curled into compact fist',
      action: 'Freeze Frame: Freezes/unfreezes current overlay in place.',
      badge: 'FREEZE',
    },
    {
      id: 'rock_on',
      icon: '🤘',
      name: 'ROCK ON / HORNS',
      rule: 'Index + Pinky extended, Middle + Ring curled',
      action: 'Toggle SAHI: Switches SAHI sliced high-resolution inference ON / OFF.',
      badge: 'SAHI',
    },
    {
      id: 'call_me',
      icon: '🤙',
      name: 'CALL ME / SHAKA',
      rule: 'Thumb + Pinky extended, Middle 3 curled',
      action: 'Voice Assistant: Activates speech microphone listening.',
      badge: 'VOICE',
    },
  ];

  return (
    <div className="holo-modal-backdrop animate-fade-in" onClick={onClose}>
      <div className="holo-modal-dialog glass-panel animate-slide-down" onClick={(e) => e.stopPropagation()}>
        {/* ── Modal Header ── */}
        <div className="holo-modal-hdr">
          <div className="holo-modal-title-group">
            <span className="modal-reactor-icon">🖐️</span>
            <div>
              <h2 className="holo-modal-title">21-LANDMARK GESTURE COMMAND MATRIX</h2>
              <span className="holo-modal-subtitle">Sub-Millimeter 3D Kinematic Hand Tracking & Real-Time Action Engine</span>
            </div>
          </div>
          <button
            className="btn-modal-close"
            onClick={() => { soundFX.playToggle(false); onClose(); }}
          >
            ✕
          </button>
        </div>

        {/* ── Armed Status Alert Banner ── */}
        <div className={`holo-modal-status-banner ${isGesturesArmed ? 'banner-armed' : 'banner-standby'}`}>
          <div className="status-banner-left">
            <span className="status-banner-icon">{isGesturesArmed ? '✅' : '⚠️'}</span>
            <div>
              <span className="status-banner-title">
                {isGesturesArmed ? '3D GESTURES SYSTEM: ARMED & ACTIVE' : '3D GESTURES SYSTEM: STANDBY (OFF)'}
              </span>
              <p className="status-banner-desc">
                {isGesturesArmed
                  ? 'Kinematic hand tracking is live. Poses in front of the camera will execute commands.'
                  : 'Gestures are currently disabled. Click the ARM button below to activate 3D recognition.'}
              </p>
            </div>
          </div>
          <button
            className={`btn-toggle-arm ${isGesturesArmed ? 'btn-disarm' : 'btn-arm'}`}
            onClick={() => onToggleGestures && onToggleGestures()}
          >
            {isGesturesArmed ? 'DISARM GESTURES' : '⚡ ARM GESTURES'}
          </button>
        </div>

        {/* ── Gesture Grid ── */}
        <div className="holo-gesture-grid">
          {gestures.map((g) => {
            const isCurrent = activeGesture === g.id;
            return (
              <div
                key={g.id}
                className={`holo-gesture-card glass-card ${isCurrent ? 'gesture-active-verified' : ''}`}
              >
                <div className="gesture-card-hdr">
                  <span className="card-gesture-icon">{g.icon}</span>
                  <span className="badge badge-cyan">{g.badge}</span>
                </div>
                <h4 className="card-gesture-title">{g.name}</h4>
                <p className="card-gesture-rule"><strong>Pose:</strong> {g.rule}</p>
                <p className="card-gesture-action"><strong>Action:</strong> {g.action}</p>

                {isCurrent && (
                  <div className="card-live-indicator animate-pulse">
                    ● POSE VERIFIED & ACTIVE
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* ── Modal Footer ── */}
        <div className="holo-modal-ftr">
          <span className="modal-hint-text">
            💡 Perform gestures in front of the camera to execute real-time actions and trigger HUD notifications.
          </span>
          <button
            className="btn-holo-confirm"
            onClick={() => { soundFX.playToggle(true); onClose(); }}
          >
            CONFIRM & RETURN
          </button>
        </div>
      </div>
    </div>
  );
}
