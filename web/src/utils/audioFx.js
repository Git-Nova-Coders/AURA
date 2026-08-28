/**
 * AURA Procedural Sci-Fi Audio Synthesizer
 * Uses the Web Audio API to generate cinematic tactical sound effects
 * (lock-on chimes, snapshot shutter clicks, mode toggles, voice hum) with zero external assets.
 */

class SoundFXEngine {
  constructor() {
    this.ctx = null;
    this.muted = false;
  }

  init() {
    if (!this.ctx && typeof window !== 'undefined') {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
  }

  playBeep(freq = 880, duration = 0.08, type = 'sine', gainVal = 0.05) {
    if (this.muted) return;
    try {
      this.init();
      if (!this.ctx) return;
      if (this.ctx.state === 'suspended') {
        this.ctx.resume();
      }

      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
      gain.gain.setValueAtTime(gainVal, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + duration);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch {
      // Ignore audio context errors
    }
  }

  // 🎯 Target Locked Sound (Double high-pitch ping)
  playLockOn() {
    this.playBeep(1200, 0.06, 'sine', 0.06);
    setTimeout(() => this.playBeep(1800, 0.09, 'sine', 0.07), 70);
  }

  // 📸 Camera Snapshot Shutter Sound
  playShutter() {
    if (this.muted) return;
    try {
      this.init();
      if (!this.ctx) return;
      const t = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(400, t);
      osc.frequency.exponentialRampToValueAtTime(80, t + 0.12);

      gain.gain.setValueAtTime(0.12, t);
      gain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start(t);
      osc.stop(t + 0.12);
    } catch {}
  }

  // 🤘 Toggle Switch Click Sound
  playToggle(state = true) {
    const freq = state ? 920 : 460;
    this.playBeep(freq, 0.06, 'triangle', 0.05);
  }

  // 🤙 Voice Assistant Trigger Sound (Rising melodic chime)
  playVoiceChime() {
    this.playBeep(523.25, 0.08, 'sine', 0.05); // C5
    setTimeout(() => this.playBeep(659.25, 0.08, 'sine', 0.05), 80); // E5
    setTimeout(() => this.playBeep(783.99, 0.12, 'sine', 0.06), 160); // G5
  }

  // 🖐️ Clean View / Reset Sound
  playSweep() {
    this.playBeep(600, 0.1, 'sine', 0.04);
  }

  toggleMute() {
    this.muted = !this.muted;
    return this.muted;
  }
}

export const soundFX = new SoundFXEngine();
