import { useState, useCallback, useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import CyberBackground from './components/CyberBackground';
import HoloHeader from './components/HoloHeader';
import TacticalViewport from './components/TacticalViewport';
import EntityMatrix from './components/EntityMatrix';
import NeuralTerminal from './components/NeuralTerminal';
import TacticalDeck from './components/TacticalDeck';
import HoloGuideModal from './components/HoloGuideModal';
import BootSequence from './components/boot/BootSequence';
import './App.css';

/**
 * AURA V2 — Master Futuristic Command Center & Multimodal AI Cockpit
 * JARVIS + Iron Man HUD + Aerospace Mission Operations Center.
 * The Camera Viewport is the hero heart (75–85% presence).
 */
export default function App() {
  const ws = useWebSocket();
  const [isBooting, setIsBooting] = useState(true);
  const [isGuideOpen, setIsGuideOpen] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [hoveredEntity, setHoveredEntity] = useState(null);
  const [inspectHistory, setInspectHistory] = useState([]);

  // Spatial Tracker Active State (Left Panel collapses when tracker is OFF)
  const isTrackingActive = ws.telemetry?.tracking_enabled ?? true;

  const [isPanelDismissed, setIsPanelDismissed] = useState(false);

  // Handle entity selection via click or pinch gesture
  const handleSelectEntity = useCallback((entity) => {
    setIsPanelDismissed(false);
    setSelectedEntity(entity);
    ws.inspectEntity(entity.class_name);
  }, [ws]);

  // When inspectResponse arrives from pinch gesture or click, record to inspectHistory
  useEffect(() => {
    if (ws.inspectResponse) {
      setIsPanelDismissed(false);
      const targetName = ws.inspectResponse.target || selectedEntity?.class_name || 'OBJECT';
      const explanation = ws.inspectResponse.response?.response_text || 'Tactical analysis completed.';
      const confidence = ws.inspectResponse.response?.confidence || selectedEntity?.confidence || 0.95;
      const spatial_pos = ws.inspectResponse.response?.spatial_pos || selectedEntity?.spatial_pos || 'CENTER';
      const ocr_texts = ws.inspectResponse.response?.ocr_texts || selectedEntity?.ocr_texts || [];
      const bbox = selectedEntity?.bbox || null;

      const newEntry = {
        id: Date.now() + Math.random(),
        target: targetName,
        class_name: targetName,
        explanation,
        confidence,
        spatial_pos,
        ocr_texts,
        bbox,
        timestamp: Date.now(),
      };

      setInspectHistory((prev) => {
        // Prevent immediate duplicates within 2 seconds
        if (prev.length > 0 && prev[0].target === targetName && Date.now() - prev[0].timestamp < 2000) {
          return [newEntry, ...prev.slice(1)];
        }
        return [newEntry, ...prev];
      });
    }
  }, [ws.inspectResponse]);

  // Deselect all inspected objects and dismiss intelligence layer
  const handleDeselectAll = useCallback(() => {
    setIsPanelDismissed(true);
    setSelectedEntity(null);
    setInspectHistory([]);
    if (ws.clearInspect) ws.clearInspect();
  }, [ws]);

  // Remove a single dossier card from history
  const handleRemoveInspectItem = useCallback((index) => {
    setInspectHistory((prev) => {
      const updated = [...prev];
      updated.splice(index, 1);
      if (updated.length === 0) {
        setSelectedEntity(null);
        setIsPanelDismissed(true);
        if (ws.clearInspect) ws.clearInspect();
      }
      return updated;
    });
  }, [ws]);

  // Automatically hide panel on Thumbs Down gesture
  useEffect(() => {
    if (ws.telemetry?.active_gesture === 'thumbs_down') {
      handleDeselectAll();
    }
  }, [ws.telemetry?.active_gesture, handleDeselectAll]);

  // Right Panel is visible ONLY when actively inspecting at least one entity and not dismissed
  const isInspecting = !isPanelDismissed && inspectHistory.length > 0;

  return (
    <>
      {/* ── 0. Cinematic AURA OS Boot Sequence (Initial Load or Diagnostic Trigger) ── */}
      {isBooting && <BootSequence onComplete={() => setIsBooting(false)} />}

      {/* ── Layer 5: Cyber Particle Background Canvas ── */}
      <CyberBackground />

      {/* ── Master HUD Grid with Hero Viewport & Projected Layers ── */}
      <div
        className={`aura-cockpit-grid ${!isTrackingActive ? 'left-collapsed' : ''} ${
          !isInspecting ? 'right-collapsed' : ''
        }`}
      >
        {/* ── Layer 4: Ultra-Thin Continuous Aerospace HUD Bar ── */}
        <HoloHeader
          isConnected={ws.isConnected}
          telemetry={ws.telemetry}
          onOpenGuide={() => setIsGuideOpen(true)}
          onTriggerBoot={() => setIsBooting(true)}
        />

        {/* ── Layer 4: Left Wing: Spatial Scene Matrix (Auto-Collapses when Tracker is OFF) ── */}
        <div className="left-panel-wrapper">
          <EntityMatrix
            scene={ws.scene}
            pointedTarget={ws.telemetry?.pointed_target}
            onSelectEntity={handleSelectEntity}
            onHoverEntity={setHoveredEntity}
            filterMode={ws.telemetry?.target_filter_mode}
          />
        </div>

        {/* ── Layer 1-3: Center Cockpit: HERO CAMERA VIEWPORT (75–85% Visual Presence) ── */}
        <div className="center-viewport-wrapper">
          <TacticalViewport
            frame={ws.lastFrame}
            scene={ws.scene}
            telemetry={ws.telemetry}
            activeToast={ws.activeToast}
            filterMode={ws.telemetry?.target_filter_mode}
            onObjectClick={handleSelectEntity}
            onOpenGuide={() => setIsGuideOpen(true)}
            onToggleGestures={ws.toggleGestures}
            onSetTargetFilter={ws.setTargetFilter}
          />
        </div>

        {/* ── Layer 2: Right Wing: Neural Intelligence Terminal (Appears On-Demand) ── */}
        <div className="right-panel-wrapper">
          {isInspecting && (
            <NeuralTerminal
              onSendChat={ws.sendChat}
              chatResponse={ws.chatResponse}
              ragResponse={ws.ragResponse}
              onSearchRAG={ws.searchRAG}
              selectedEntity={selectedEntity}
              inspectResponse={ws.inspectResponse}
              inspectHistory={inspectHistory}
              onDeselectAll={handleDeselectAll}
              onRemoveInspectItem={handleRemoveInspectItem}
              isVoiceListening={ws.telemetry?.voice_listening}
            />
          )}
        </div>

        {/* ── Layer 4: Floating Holographic Aerospace Command Dock ── */}
        <TacticalDeck
          telemetry={ws.telemetry}
          filterMode={ws.telemetry?.target_filter_mode}
          onToggleSAHI={ws.toggleSAHI}
          onToggleTracking={ws.toggleTracking}
          onToggleOCR={ws.toggleOCR}
          onToggleVoice={ws.toggleVoice}
          onToggleGestures={ws.toggleGestures}
          onCycleTargetFilter={ws.cycleTargetFilter}
          onSetTargetFilter={ws.setTargetFilter}
        />

        {/* ── 3D Gesture Manual Modal ── */}
        <HoloGuideModal
          isOpen={isGuideOpen}
          onClose={() => setIsGuideOpen(false)}
          activeGesture={ws.telemetry?.active_gesture}
          isGesturesArmed={ws.telemetry?.gestures_enabled}
          onToggleGestures={ws.toggleGestures}
        />
      </div>
    </>
  );
}
