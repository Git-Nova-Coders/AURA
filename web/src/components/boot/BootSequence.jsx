import React, { useState, useEffect } from 'react';
import NeuralCore from '../hud/NeuralCore';
import { soundFX } from '../../utils/audioFx';

/**
 * BootSequence — AURA V2 Cinematic Operating System Initialization.
 * 4-Phase aerospace startup experience:
 * Phase 01: VOID (Deep abyss with ignition particle)
 * Phase 02: CORE INITIALIZATION (A.U.R.A V2.0, sequential subsystem illumination)
 * Phase 03: SENSOR SYNCHRONIZATION (Data streams flow toward Central Core)
 * Phase 04: PERCEPTION ONLINE (100% sensor link established ➔ collapse into Cockpit)
 */
export default function BootSequence({ onComplete }) {
  const [phase, setPhase] = useState(1);
  const [subsystemsLoaded, setSubsystemsLoaded] = useState([]);
  const [progress, setProgress] = useState(0);

  const subsystems = [
    'MEMORY CORE',
    'VISION CORE',
    'TRACKING CORE',
    'SPATIAL ENGINE',
    'NEURAL CORE',
    'AUDIO CORE',
    'REASONING',
  ];

  useEffect(() => {
    soundFX.playBeep(440, 0.12, 'sine', 0.08);

    // Phase 1 -> Phase 2: Core Init (at 600ms)
    const t1 = setTimeout(() => {
      setPhase(2);
      soundFX.playBeep(660, 0.08, 'sine', 0.08);
      
      // Sequentially illuminate subsystems
      subsystems.forEach((sub, idx) => {
        setTimeout(() => {
          setSubsystemsLoaded((prev) => [...prev, sub]);
          soundFX.playBeep(800 + idx * 80, 0.04, 'triangle', 0.05);
        }, 150 * (idx + 1));
      });
    }, 700);

    // Phase 2 -> Phase 3: Sensor Synchronization (at 1900ms)
    const t2 = setTimeout(() => {
      setPhase(3);
      soundFX.playToggle(true);
    }, 1900);

    // Phase 3 -> Phase 4: Perception Online (at 2800ms)
    const t3 = setTimeout(() => {
      setPhase(4);
      soundFX.playVoiceChime();
      
      // Progress bar interpolation
      let p = 0;
      const interval = setInterval(() => {
        p += 10;
        setProgress(Math.min(100, p));
        if (p >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            if (onComplete) onComplete();
          }, 450);
        }
      }, 40);
    }, 2800);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, []);

  const handleSkip = () => {
    soundFX.playClick();
    if (onComplete) onComplete();
  };

  return (
    <div className={`v2-boot-overlay phase-${phase}`} onClick={handleSkip} title="Click anywhere to skip boot sequence">
      {/* ── Skip Button ── */}
      <button className="v2-btn-skip-boot" onClick={handleSkip}>
        ESC // SKIP BOOT ➔
      </button>

      {/* ── PHASE 01: VOID & IGNITION ── */}
      {phase === 1 && (
        <div className="boot-phase-void animate-fade-in">
          <div className="void-ignition-particle animate-pulse" />
          <span className="void-brand-title">AURA</span>
        </div>
      )}

      {/* ── PHASE 02: CORE INITIALIZATION ── */}
      {phase === 2 && (
        <div className="boot-phase-core animate-fade-in">
          <div className="boot-core-center">
            <NeuralCore size={140} isThinking={true} statusText="AURA 2.0" />
            <h1 className="boot-title">A.U.R.A V2.0</h1>
            <span className="boot-sub">UNIVERSAL COGNITIVE PERCEPTION SYSTEM</span>
          </div>

          <div className="boot-subsystems-list">
            {subsystems.map((sub, idx) => {
              const isLoaded = subsystemsLoaded.includes(sub);
              return (
                <div key={idx} className={`boot-sub-row ${isLoaded ? 'sub-online' : 'sub-pending'}`}>
                  <span className="sub-name">{sub}</span>
                  <span className="sub-dots">................</span>
                  <span className="sub-val">{isLoaded ? 'ONLINE' : 'BOOTING'}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── PHASE 03: SENSOR SYNCHRONIZATION ── */}
      {phase === 3 && (
        <div className="boot-phase-sync animate-fade-in">
          <NeuralCore size={160} isThinking={true} statusText="SYNC" />
          <h2 className="sync-title">SENSOR SYNCHRONIZATION</h2>
          
          <div className="sync-streams-matrix">
            <div className="stream-node">CAMERA ───┐</div>
            <div className="stream-node">GESTURE ───┼──→ AURA COGNITIVE CORE</div>
            <div className="stream-node">OCR ───────┤</div>
            <div className="stream-node">KNOWLEDGE ─┘</div>
          </div>
        </div>
      )}

      {/* ── PHASE 04: PERCEPTION ONLINE ── */}
      {phase === 4 && (
        <div className="boot-phase-online animate-fade-in">
          <div className="online-beacon animate-pulse">⬢</div>
          <h2 className="online-title">PERCEPTION MATRIX ACTIVE</h2>
          <div className="boot-progress-bar-container">
            <div className="boot-progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <span className="online-meta">{progress}% // ENVIRONMENTAL SENSOR LINK ESTABLISHED</span>
        </div>
      )}
    </div>
  );
}
