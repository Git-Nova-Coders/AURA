import React, { useState, useEffect, useRef } from 'react';

/**
 * RemoteCameraStreamer — Dedicated web page opened on mobile phones/tablets.
 * Accesses device camera via navigator.mediaDevices.getUserMedia, captures frames,
 * and streams them via WebSocket back to the AURA server on the laptop.
 */
export default function RemoteCameraStreamer() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamIntervalRef = useRef(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [facingMode, setFacingMode] = useState('environment'); // Default to rear camera
  const [statusMessage, setStatusMessage] = useState('Ready to connect');
  const [fps, setFps] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [latency, setLatency] = useState(0);
  const [hasCameraPermission, setHasCameraPermission] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  // Initialize WebSocket connection to AURA host
  const connectWebSocket = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    setStatusMessage('Connecting to AURA Dashboard...');
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setStatusMessage('Connected to AURA Neural Core');
      // Inform server of remote camera device
      ws.send(
        JSON.stringify({
          type: 'set_camera_source',
          source: 'remote',
        })
      );
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'telemetry') {
          if (msg.data?.fps) setLatency(Math.round(msg.data.inference_latency_ms || 0));
        }
      } catch {
        // Ignore non-json
      }
    };

    ws.onerror = () => {
      setStatusMessage('Connection error. Is AURA running on your laptop?');
    };

    ws.onclose = () => {
      setStatusMessage('Disconnected. Reconnecting in 3s...');
      setTimeout(() => {
        if (isStreaming) connectWebSocket();
      }, 3000);
    };

    wsRef.current = ws;
  };

  // Start mobile device camera stream
  const startCamera = async () => {
    setErrorMsg(null);
    try {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: facingMode,
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      setHasCameraPermission(true);
      setStatusMessage('Camera active. Starting transmission...');
      connectWebSocket();
      startFrameTransmission();
      setIsStreaming(true);
    } catch (err) {
      console.error('Camera access error:', err);
      setErrorMsg(`Camera error: ${err.message || 'Permission denied. Please allow camera access in browser.'}`);
      setStatusMessage('Camera access denied');
    }
  };

  // Stop camera and transmission
  const stopCamera = () => {
    if (streamIntervalRef.current) {
      clearInterval(streamIntervalRef.current);
      streamIntervalRef.current = null;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsStreaming(false);
    setStatusMessage('Streaming stopped');
  };

  // Frame capture loop ~20 FPS for optimal Wi-Fi balance
  const startFrameTransmission = () => {
    if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);

    let sentFrames = 0;
    let lastTime = performance.now();

    streamIntervalRef.current = setInterval(() => {
      if (!videoRef.current || !canvasRef.current || !wsRef.current) return;
      if (wsRef.current.readyState !== WebSocket.OPEN) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video.videoWidth === 0 || video.videoHeight === 0) return;

      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, 640, 480);

      // Low-compression JPEG string for fast transfer
      const jpegData = canvas.toDataURL('image/jpeg', 0.65);

      wsRef.current.send(
        JSON.stringify({
          type: 'remote_frame',
          frame: jpegData,
          device_info: {
            userAgent: navigator.userAgent,
            facingMode: facingMode,
            timestamp: Date.now(),
          },
        })
      );

      sentFrames++;
      setFrameCount((prev) => prev + 1);

      const now = performance.now();
      if (now - lastTime >= 1000) {
        setFps(Math.round((sentFrames * 1000) / (now - lastTime)));
        sentFrames = 0;
        lastTime = now;
      }
    }, 50); // 20 FPS
  };

  // Toggle Front / Rear camera
  const toggleFacingMode = () => {
    setFacingMode((prev) => (prev === 'environment' ? 'user' : 'environment'));
  };

  useEffect(() => {
    if (isStreaming) {
      startCamera();
    }
  }, [facingMode]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="mobile-streamer-container">
      {/* Top Header */}
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

      {/* Main Camera Viewfinder */}
      <div className="mobile-viewfinder-wrapper">
        <video
          ref={videoRef}
          playsInline
          muted
          autoPlay
          className="mobile-video-preview"
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {/* Viewfinder Target Overlays */}
        <div className="viewfinder-crosshair"></div>
        <div className="viewfinder-corner top-left"></div>
        <div className="viewfinder-corner top-right"></div>
        <div className="viewfinder-corner bottom-left"></div>
        <div className="viewfinder-corner bottom-right"></div>

        {/* Real-time Telemetry Overlay */}
        <div className="mobile-telemetry-hud">
          <div className="telemetry-chip">
            <span className="chip-lbl">FPS</span>
            <span className="chip-val">{fps}</span>
          </div>
          <div className="telemetry-chip">
            <span className="chip-lbl">FRAMES</span>
            <span className="chip-val">{frameCount}</span>
          </div>
          <div className="telemetry-chip">
            <span className="chip-lbl">INFER</span>
            <span className="chip-val">{latency}ms</span>
          </div>
        </div>

        {/* Status Toast */}
        <div className="mobile-status-toast">
          <span>{statusMessage}</span>
        </div>
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div className="mobile-error-banner">
          <span>⚠️ {errorMsg}</span>
        </div>
      )}

      {/* Controls Bar */}
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

      {/* Footer Info */}
      <div className="mobile-streamer-footer">
        <p>AURA Adaptive Understanding & Reasoning Architecture</p>
        <p className="subtext">Watch your laptop screen for real-time bounding boxes & detections</p>
      </div>
    </div>
  );
}
