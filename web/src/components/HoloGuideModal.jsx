import React from 'react';
import { soundFX } from '../utils/audioFx';

/**
 * HoloGuideModal — Ultra-Futuristic Cybernetic 3D Gesture Matrix Manual
 * High-contrast, vibrant, jaw-dropping aerospace HUD styling matching
 * the AURA OS splash screen with 100% legibility and glowing neon aesthetics.
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
      rule: 'All 5 fingers extended outward',
      action: 'Clean Camera View (Hides all bounding boxes)',
      badge: 'VIEW',
    },
    {
      id: 'peace_sign',
      icon: '✌️',
      name: 'VICTORY ("V")',
      rule: 'Index + Middle extended, others curled',
      action: 'High-Res Frame Snapshot to captures/ folder',
      badge: 'CAPTURE',
    },
    {
      id: 'pointing',
      icon: '👉',
      name: 'POINTING RAY',
      rule: 'Index finger extended, others curled',
      action: 'Projects 3D Raycast Laser to lock target',
      badge: 'LOCK',
    },
    {
      id: 'pinch',
      icon: '👌',
      name: 'PINCH (AIR CLICK)',
      rule: 'Thumb Tip + Index Tip pinched together',
      action: 'Inspect Target: Multimodal AI reasoning & dossier',
      badge: 'INSPECT',
    },
    {
      id: 'thumbs_up',
      icon: '👍',
      name: 'THUMBS UP',
      rule: 'Thumb pointing upward (+Y), fingers curled',
      action: 'Restore View: Unhides all boxes & resets lock',
      badge: 'RESET',
    },
    {
      id: 'thumbs_down',
      icon: '👎',
      name: 'THUMBS DOWN',
      rule: 'Thumb pointing downward (-Y), fingers curled',
      action: 'Deselect Target: Clears active focus & lock',
      badge: 'CLEAR',
    },
    {
      id: 'fist',
      icon: '✊',
      name: 'FIST',
      rule: 'All 5 fingers curled into a tight fist',
      action: 'Freeze Frame: Holds current perception overlay',
      badge: 'FREEZE',
    },
    {
      id: 'rock_on',
      icon: '🤘',
      name: 'ROCK ON / HORNS',
      rule: 'Index + Pinky extended, Middle + Ring curled',
      action: 'Toggle SAHI Matrix Sliced High-Res Inference',
      badge: 'SAHI',
    },
    {
      id: 'call_me',
      icon: '🤙',
      name: 'CALL ME / SHAKA',
      rule: 'Thumb + Pinky extended, Middle 3 curled',
      action: 'Activate Voice Assistant speech microphone',
      badge: 'VOICE',
    },
  ];

  return (
    <div className="v2-modal-backdrop animate-fade-in" onClick={onClose}>
      <div className="v2-modal-dialog animate-slide-down" onClick={(e) => e.stopPropagation()}>
        {/* ── 4 Signature Corner Reticles ── */}
        <span className="v2-modal-corner corner-tl" />
        <span className="v2-modal-corner corner-tr" />
        <span className="v2-modal-corner corner-bl" />
        <span className="v2-modal-corner corner-br" />

        {/* ── Top Laser Scanning Line ── */}
        <div className="v2-modal-laser-beam" />

        {/* ── Modal Header ── */}
        <div className="v2-modal-hdr">
          <div className="v2-modal-title-wrap">
            <div className="v2-modal-beacon animate-pulse">⬢</div>
            <div>
              <div className="v2-title-row">
                <h2 className="v2-modal-title">21-LANDMARK GESTURE COMMAND MATRIX</h2>
                <span className="v2-modal-tag">OPTICAL 3D SENSORS</span>
              </div>
              <span className="v2-modal-subtitle">
                Sub-Millimeter 3D Kinematic Hand Tracking & Real-Time Action Engine
              </span>
            </div>
          </div>
          <button
            className="v2-btn-modal-close"
            onClick={() => { soundFX.playToggle(false); onClose(); }}
            title="Close Gesture Matrix"
          >
            ✕
          </button>
        </div>

        {/* ── Armed Status Alert Banner ── */}
        <div className={`v2-modal-banner ${isGesturesArmed ? 'banner-armed' : 'banner-standby'}`}>
          <div className="v2-banner-left">
            <span className="v2-banner-icon">{isGesturesArmed ? '⚡' : '⚠️'}</span>
            <div>
              <span className="v2-banner-title">
                {isGesturesArmed ? '3D GESTURES: ARMED & SENSING' : '3D GESTURES: STANDBY (DISARMED)'}
              </span>
              <p className="v2-banner-desc">
                {isGesturesArmed
                  ? 'Kinematic hand tracking is ACTIVE. Poses detected in front of the lens will execute real-time commands.'
                  : 'Gestures are currently offline. Click the ARM button below to activate 3D recognition.'}
              </p>
            </div>
          </div>
          <button
            className={`v2-btn-arm-toggle ${isGesturesArmed ? 'btn-disarm' : 'btn-arm'}`}
            onClick={() => onToggleGestures && onToggleGestures()}
          >
            {isGesturesArmed ? 'DISARM GESTURES' : '⚡ ARM GESTURES'}
          </button>
        </div>

        {/* ── 9 High-Contrast Gesture Command Cards ── */}
        <div className="v2-gesture-cards-grid">
          {gestures.map((g) => {
            const isCurrent = activeGesture === g.id;
            return (
              <div
                key={g.id}
                className={`v2-gesture-card ${isCurrent ? 'card-active-verified' : ''}`}
              >
                {/* Corner reticles for each card */}
                <span className="card-bracket bracket-tl" />
                <span className="card-bracket bracket-tr" />
                <span className="card-bracket bracket-bl" />
                <span className="card-bracket bracket-br" />

                <div className="card-top-row">
                  <span className="card-icon">{g.icon}</span>
                  <span className="v2-card-badge">{g.badge}</span>
                </div>

                <h4 className="card-title">{g.name}</h4>

                <div className="card-info-box">
                  <div className="info-line">
                    <span className="lbl-pose">POSE:</span>
                    <span className="val-pose">{g.rule}</span>
                  </div>
                  <div className="info-line">
                    <span className="lbl-act">ACTION:</span>
                    <span className="val-act">{g.action}</span>
                  </div>
                </div>

                {isCurrent && (
                  <div className="card-live-pill animate-pulse">
                    ● POSE VERIFIED & EXECUTING
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* ── Modal Footer ── */}
        <div className="v2-modal-ftr">
          <span className="v2-ftr-hint">
            💡 Perform gestures in front of the camera to execute real-time actions and trigger HUD notifications.
          </span>
          <button
            className="v2-btn-confirm"
            onClick={() => { soundFX.playToggle(true); onClose(); }}
          >
            CONFIRM & RETURN TO COCKPIT ➔
          </button>
        </div>
      </div>
    </div>
  );
}
