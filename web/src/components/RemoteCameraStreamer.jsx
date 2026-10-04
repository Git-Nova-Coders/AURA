import React, { useState, useEffect, useRef, useCallback } from 'react';

/**
 * RemoteCameraStreamer — Mobile sensor node page for AURA.
 * Opens on phone/tablet to stream camera feed to the laptop running AURA.
 * Sends frames via WebSocket (primary) or HTTP POST fallback.
 */
export default function RemoteCameraStreamer() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamIntervalRef = useRef(null);
  const isStreamingRef = useRef(false);

  const [isStreaming, setIsStreaming] = useState(false);
  const [facingMode, setFacingMode] = useState('environment');
  const [statusMessage, setStatusMessage] = useState('Initializing sensor...');
  const [fps, setFps] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [latency, setLatency] = useState(0);
  const [errorMsg, setErrorMsg] = useState(null);
  const [showHttpGuide, setShowHttpGuide] = useState(false);
  const [needsTap, setNeedsTap] = useState(false);

  // ── WebSocket connection ──────────────────────────────────────────────────
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    setStatusMessage('Connecting to AURA...');
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setStatusMessage('✅ Connected — transmitting frames');
      ws.send(JSON.stringify({ type: 'set_camera_source', source: 'remote' }));
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'telemetry') {
          setLatency(Math.round(msg.data?.inference_latency_ms || 0));
        }
      } catch { /* ignore */ }
    };

    ws.onerror = () => setStatusMessage('⚠ WS error — using HTTP fallback');
    ws.onclose = () => {
      if (isStreamingRef.current) {
        setStatusMessage('Reconnecting in 3s...');
        setTimeout(connectWebSocket, 3000);
      }
    };

    wsRef.current = ws;
  }, []);

  // ── Frame transmission loop ───────────────────────────────────────────────
  const startFrameTransmission = useCallback(() => {
    if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
    let sentFrames = 0;
    let lastTime = performance.now();
    let isUploadingHttp = false;

    streamIntervalRef.current = setInterval(() => {
      if (!videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      if (video.videoWidth === 0 || video.videoHeight === 0) return;

      const canvas = canvasRef.current;
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, 640, 480);
      const jpegData = canvas.toDataURL('image/jpeg', 0.65);

      const payload = {
        type: 'remote_frame',
        frame: jpegData,
        device_info: { userAgent: navigator.userAgent, facingMode, timestamp: Date.now() },
      };

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(payload));
        sentFrames++;
        setFrameCount((p) => p + 1);
      } else if (!isUploadingHttp) {
        isUploadingHttp = true;
        fetch('/api/remote/frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frame: jpegData, device_info: payload.device_info }),
        })
          .then(() => { sentFrames++; setFrameCount((p) => p + 1); })
          .catch(() => {})
          .finally(() => { isUploadingHttp = false; });
      }

      const now = performance.now();
      if (now - lastTime >= 1000) {
        setFps(Math.round((sentFrames * 1000) / (now - lastTime)));
        sentFrames = 0;
        lastTime = now;
      }
    }, 50); // ~20 FPS
  }, [facingMode]);

  // ── Start camera ──────────────────────────────────────────────────────────
  const startCamera = useCallback(async () => {
    setErrorMsg(null);
    setShowHttpGuide(false);
    setNeedsTap(false);
    setStatusMessage('Requesting camera access...');

    // Check API support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const isHttp = window.location.protocol === 'http:' && !window.location.hostname.includes('localhost');
      if (isHttp) {
        setShowHttpGuide(true);
        setStatusMessage('Camera blocked — see guide below');
      } else {
        setErrorMsg('Camera API not supported in this browser.');
        setStatusMessage('Camera unavailable');
      }
      return;
    }

    // Stop any existing stream
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
    }

    const constraints = {
      video: { facingMode: { ideal: facingMode }, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    };

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (firstErr) {
      // Fallback to any camera
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch (finalErr) {
        const isHttp = window.location.protocol === 'http:' && !window.location.hostname.includes('localhost');
        const blocked = finalErr.name === 'NotAllowedError' || finalErr.name === 'SecurityError' || finalErr.name === 'NotSupportedError';
        if (isHttp && blocked) {
          setShowHttpGuide(true);
          setStatusMessage('Camera blocked over HTTP — see guide below');
        } else {
          setErrorMsg(`Camera error: ${finalErr.message || finalErr.name}`);
          setStatusMessage('Permission denied');
        }
        return;
      }
    }

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.setAttribute('playsinline', 'true');
      videoRef.current.setAttribute('webkit-playsinline', 'true');
      videoRef.current.muted = true;
      try {
        await videoRef.current.play();
      } catch (playErr) {
        console.warn('Autoplay blocked — tap to start:', playErr);
        setNeedsTap(true);
        setStatusMessage('Tap the viewfinder to start');
        return;
      }
    }

    isStreamingRef.current = true;
    setIsStreaming(true);
    setStatusMessage('📡 Streaming to AURA laptop...');
    connectWebSocket();
    startFrameTransmission();
  }, [facingMode, connectWebSocket, startFrameTransmission]);

  // ── Stop camera ───────────────────────────────────────────────────────────
  const stopCamera = useCallback(() => {
    isStreamingRef.current = false;
    if (streamIntervalRef.current) { clearInterval(streamIntervalRef.current); streamIntervalRef.current = null; }
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    setIsStreaming(false);
    setStatusMessage('Streaming stopped');
    setFps(0);
  }, []);

  // ── Tap-to-play handler ───────────────────────────────────────────────────
  const handleViewfinderTap = useCallback(async () => {
    if (!needsTap || !videoRef.current) return;
    try {
      await videoRef.current.play();
      setNeedsTap(false);
      isStreamingRef.current = true;
      setIsStreaming(true);
      setStatusMessage('📡 Streaming to AURA laptop...');
      connectWebSocket();
      startFrameTransmission();
    } catch (e) {
      setErrorMsg(`Could not play video: ${e.message}`);
    }
  }, [needsTap, connectWebSocket, startFrameTransmission]);

  // ── Switch camera lens ────────────────────────────────────────────────────
  const toggleFacingMode = () => {
    setFacingMode((prev) => (prev === 'environment' ? 'user' : 'environment'));
  };

  // Re-start camera when facing mode changes (if streaming)
  useEffect(() => {
    if (isStreaming) startCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [facingMode]);

  // Auto-start on mount
  useEffect(() => {
    startCamera();
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="mobile-streamer-container">
      {/* ── Header ── */}
      <div className="mobile-streamer-header">
        <div className="header-brand">
          <span className="brand-dot animate-pulse"></span>
          <span className="brand-title">AURA SENSOR NODE</span>
        </div>
        <div className="header-status-badge">
          <span className={`status-led ${isStreaming ? 'led-emerald' : 'led-amber'}`}></span>
          <span>{isStreaming ? 'TRANSMITTING' : 'STANDBY'}</span>
        </div>
      </div>

      {/* ── Viewfinder ── */}
      <div className="mobile-viewfinder-wrapper" onClick={handleViewfinderTap} style={{ cursor: needsTap ? 'pointer' : 'default' }}>
        <video ref={videoRef} playsInline muted autoPlay className="mobile-video-preview" />
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {/* Tap overlay */}
        {needsTap && (
          <div className="viewfinder-tap-overlay">
            <div className="tap-circle">👆</div>
            <p>Tap to start camera</p>
          </div>
        )}

        {/* Corner overlays */}
        <div className="viewfinder-crosshair"></div>
        <div className="viewfinder-corner top-left"></div>
        <div className="viewfinder-corner top-right"></div>
        <div className="viewfinder-corner bottom-left"></div>
        <div className="viewfinder-corner bottom-right"></div>

        {/* Telemetry HUD */}
        <div className="mobile-telemetry-hud">
          <div className="telemetry-chip"><span className="chip-lbl">FPS</span><span className="chip-val">{fps}</span></div>
          <div className="telemetry-chip"><span className="chip-lbl">FRAMES</span><span className="chip-val">{frameCount}</span></div>
          <div className="telemetry-chip"><span className="chip-lbl">INFER</span><span className="chip-val">{latency}ms</span></div>
        </div>

        <div className="mobile-status-toast"><span>{statusMessage}</span></div>
      </div>

      {/* ── Error banner ── */}
      {errorMsg && (
        <div className="mobile-error-banner">⚠️ {errorMsg}</div>
      )}

      {/* ── HTTP Guide ── */}
      {showHttpGuide && (
        <div className="http-guide-panel">
          <div className="http-guide-title">📷 Camera blocked over HTTP</div>
          <p>Modern Chrome/Safari require <strong>HTTPS</strong> or a special flag to allow camera over plain <code>http://</code>.</p>
          <div className="http-guide-steps">
            <p><strong>Chrome on Android — quick fix:</strong></p>
            <ol>
              <li>Open a new tab and go to: <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code></li>
              <li>Paste <code>http://{window.location.host}</code> in the text box</li>
              <li>Set the dropdown to <strong>Enabled</strong></li>
              <li>Tap <strong>Relaunch</strong></li>
              <li>Return to this page and tap <strong>START STREAMING</strong></li>
            </ol>
            <p style={{ marginTop: '8px', color: '#aaa' }}>Alternatively, if AURA is configured with an SSL cert, open via <code>https://</code>.</p>
          </div>
          <button className="btn-stream-action btn-start" style={{ marginTop: '12px', width: '100%' }} onClick={startCamera}>
            🔄 Try Again
          </button>
        </div>
      )}

      {/* ── Controls ── */}
      {!showHttpGuide && (
        <div className="mobile-streamer-controls">
          {!isStreaming ? (
            <button className="btn-stream-action btn-start" onClick={startCamera}>
              🚀 START STREAMING
            </button>
          ) : (
            <button className="btn-stream-action btn-stop" onClick={stopCamera}>
              🛑 STOP STREAMING
            </button>
          )}
          <button
            className="btn-stream-action btn-flip"
            onClick={toggleFacingMode}
            title="Switch Front / Rear Lens"
          >
            🔄 {facingMode === 'environment' ? 'REAR LENS' : 'FRONT LENS'}
          </button>
        </div>
      )}

      {/* ── Footer ── */}
      <div className="mobile-streamer-footer">
        <p>AURA Adaptive Understanding &amp; Reasoning Architecture</p>
        <p className="subtext">Watch your laptop screen for real-time bounding boxes &amp; detections</p>
      </div>
    </div>
  );
}
