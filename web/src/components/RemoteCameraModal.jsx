import React, { useState, useEffect } from 'react';
import { QRCodeSVG } from 'qrcode.react';
import { soundFX } from '../utils/audioFx';

/**
 * RemoteCameraModal — Connect any smartphone, tablet, or external device camera to AURA.
 * Displays QR code and direct URL to open `/remote-camera` in any mobile browser.
 */
export default function RemoteCameraModal({
  isOpen,
  onClose,
  telemetry,
  onSelectCameraSource,
}) {
  const [networkIps, setNetworkIps] = useState([]);
  const [selectedIp, setSelectedIp] = useState('');
  const [copied, setCopied] = useState(false);
  const [port, setPort] = useState(8420);

  // Fetch host IP addresses
  useEffect(() => {
    if (!isOpen) return;

    fetch('/api/network/ips')
      .then((res) => res.json())
      .then((data) => {
        if (data.ips && data.ips.length > 0) {
          setNetworkIps(data.ips);
          // Prefer non-localhost IP
          const nonLocal = data.ips.find((ip) => !ip.includes('localhost') && !ip.startsWith('127.'));
          setSelectedIp(nonLocal || data.ips[0]);
          if (data.port) setPort(data.port);
        } else {
          setNetworkIps([window.location.hostname]);
          setSelectedIp(window.location.hostname);
          setPort(window.location.port || 8420);
        }
      })
      .catch(() => {
        const host = window.location.hostname || 'localhost';
        setNetworkIps([host]);
        setSelectedIp(host);
        setPort(window.location.port || 8420);
      });
  }, [isOpen]);

  if (!isOpen) return null;

  const currentSource = telemetry?.camera_source || 'local';
  const isRemoteConnected = telemetry?.remote_device_connected ?? false;

  // Remote URL to open on the mobile phone
  const effectivePort = port ? `:${port}` : '';
  const protocol = window.location.protocol;
  const remoteUrl = `${protocol}//${selectedIp}${effectivePort}/remote-camera`;

  const handleCopy = () => {
    navigator.clipboard.writeText(remoteUrl);
    setCopied(true);
    soundFX.playClick();
    setTimeout(() => setCopied(false), 2200);
  };

  const handleSwitchSource = (source) => {
    soundFX.playToggle(true);
    if (onSelectCameraSource) {
      onSelectCameraSource(source);
    }
  };

  return (
    <div className="holo-modal-overlay animate-fade-in" onClick={onClose}>
      <div
        className="holo-modal-card remote-camera-modal glass-card"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-title-group">
            <span className="modal-icon">📱</span>
            <div>
              <h2 className="modal-title">REMOTE DEVICE CAMERA LINK</h2>
              <p className="modal-subtitle">
                CONNECT SMARTPHONE / TABLET AS REAL-TIME MACHINE VISION SENSOR
              </p>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} title="Close Modal">
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body remote-camera-body">
          {/* Top Camera Source Selector */}
          <div className="camera-source-selector">
            <span className="selector-label">ACTIVE SENSOR INPUT:</span>
            <div className="source-tabs">
              <button
                className={`source-tab ${currentSource === 'local' ? 'source-tab-active' : ''}`}
                onClick={() => handleSwitchSource('local')}
              >
                💻 LAPTOP WEBCAM
              </button>
              <button
                className={`source-tab ${currentSource === 'remote' ? 'source-tab-active-emerald' : ''}`}
                onClick={() => handleSwitchSource('remote')}
              >
                📱 REMOTE DEVICE {isRemoteConnected && <span className="tab-live-badge">LIVE</span>}
              </button>
              <button
                className={`source-tab ${currentSource === 'synthetic' ? 'source-tab-active' : ''}`}
                onClick={() => handleSwitchSource('synthetic')}
              >
                ⚡ SYNTHETIC SIMULATOR
              </button>
            </div>
          </div>

          {/* Connection Status Banner */}
          <div className={`remote-connection-banner ${isRemoteConnected ? 'banner-connected' : 'banner-waiting'}`}>
            <span className={`pulse-beacon ${isRemoteConnected ? 'beacon-emerald' : 'beacon-amber'}`} />
            <div className="banner-text">
              <span className="banner-status-title">
                {isRemoteConnected ? 'REMOTE SENSOR LINK ESTABLISHED' : 'WAITING FOR REMOTE DEVICE STREAM...'}
              </span>
              <span className="banner-status-desc">
                {isRemoteConnected
                  ? 'Frames streaming to AURA pipeline. Object detection, pose estimation, and tracking active on laptop screen.'
                  : 'Scan the QR code below or open the URL on your mobile browser (same Wi-Fi network).'}
              </span>
            </div>
          </div>

          {/* QR Code and Instructions Grid */}
          <div className="remote-grid-layout">
            {/* QR Code Column */}
            <div className="remote-qr-box">
              <div className="qr-frame">
                <div className="qr-white-card">
                  <QRCodeSVG
                    value={remoteUrl}
                    size={200}
                    bgColor="#ffffff"
                    fgColor="#000000"
                    level="M"
                    includeMargin={true}
                  />
                </div>
                <span className="qr-scan-guide">⚡ SCAN WITH PHONE CAMERA</span>
              </div>
            </div>

            {/* Steps & Direct URL Column */}
            <div className="remote-details-col">
              <div className="instruction-step">
                <span className="step-num">1</span>
                <div className="step-content">
                  <strong>Ensure Same Wi-Fi Network</strong>
                  <p>Connect your phone to the same Wi-Fi or local hotspot as your laptop.</p>
                </div>
              </div>

              <div className="instruction-step">
                <span className="step-num">2</span>
                <div className="step-content">
                  <strong>Scan QR Code or Open URL</strong>
                  <p>Open this URL on your mobile phone browser (Chrome, Safari, Firefox):</p>
                  <div className="url-copy-bar">
                    <input
                      type="text"
                      readOnly
                      value={remoteUrl}
                      className="url-display-input"
                    />
                    <button className="btn-copy" onClick={handleCopy}>
                      {copied ? '✓ COPIED' : '📋 COPY'}
                    </button>
                  </div>
                </div>
              </div>

              <div className="instruction-step">
                <span className="step-num">3</span>
                <div className="step-content">
                  <strong>Allow Camera Access & Stream</strong>
                  <p>Tap "START STREAMING". Your laptop screen will instantly display and detect objects from your phone's lens!</p>
                </div>
              </div>

              {/* IP selector if multiple network interfaces exist */}
              {networkIps.length > 1 && (
                <div className="ip-selector-group">
                  <label htmlFor="ip-select">Host IP Address:</label>
                  <select
                    id="ip-select"
                    value={selectedIp}
                    onChange={(e) => setSelectedIp(e.target.value)}
                    className="cyber-select"
                  >
                    {networkIps.map((ip) => (
                      <option key={ip} value={ip}>
                        {ip}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="modal-footer">
          <button className="v2-btn-manual" onClick={onClose}>
            DISMISS
          </button>
        </div>
      </div>
    </div>
  );
}
