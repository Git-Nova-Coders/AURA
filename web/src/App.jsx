import { useState, useCallback, useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import CyberBackground from './components/CyberBackground';
import HoloHeader from './components/HoloHeader';
import TacticalViewport from './components/TacticalViewport';
import EntityMatrix from './components/EntityMatrix';
import NeuralTerminal from './components/NeuralTerminal';
import TacticalDeck from './components/TacticalDeck';
import HoloGuideModal from './components/HoloGuideModal';
import './App.css';

/**
 * AURA Master Cybernetic Command Center & 3D Gesture Matrix
 * AAA Sci-Fi Aerospace HUD Layout with Dynamic Auto-Adapting Fluid Panels
 */
export default function App() {
  const ws = useWebSocket();
  const [isGuideOpen, setIsGuideOpen] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [inspectHistory, setInspectHistory] = useState([]);

  // Spatial Tracker Active State (Left Panel collapses when tracker is OFF)
  const isTrackingActive = ws.telemetry?.tracking_enabled ?? true;

  // Handle entity selection via click or pinch gesture
  const handleSelectEntity = useCallback((entity) => {
    setSelectedEntity(entity);
    ws.inspectEntity(entity.class_name);
  }, [ws]);

  // When inspectResponse arrives from pinch gesture or click, record to inspectHistory
  useEffect(() => {
    if (ws.inspectResponse) {
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
        // Prevent immediate duplicates of the same target within 2 seconds
        if (prev.length > 0 && prev[0].target === targetName && Date.now() - prev[0].timestamp < 2000) {
          return [newEntry, ...prev.slice(1)];
        }
        return [newEntry, ...prev];
      });
    }
  }, [ws.inspectResponse, selectedEntity]);

  // Deselect all inspected objects and close the right panel
  const handleDeselectAll = useCallback(() => {
    setSelectedEntity(null);
    setInspectHistory([]);
  }, []);

  // Remove a single dossier card from history
  const handleRemoveInspectItem = useCallback((index) => {
    setInspectHistory((prev) => {
      const updated = [...prev];
      updated.splice(index, 1);
      if (updated.length === 0) {
        setSelectedEntity(null);
      }
      return updated;
    });
  }, []);

  // Right Panel is visible ONLY when actively inspecting at least one entity
  const isInspecting = inspectHistory.length > 0 || selectedEntity !== null;

  return (
    <>
      {/* ── 0. Cyber Particle Background Canvas ── */}
      <CyberBackground />

      {/* ── Master HUD Grid with Dynamic Fluid Classes ── */}
      <div
        className={`aura-command-center ${!isTrackingActive ? 'left-collapsed' : ''} ${
          !isInspecting ? 'right-collapsed' : ''
        }`}
      >
        {/* ── 1. Top Aerospace Telemetry Header ── */}
        <HoloHeader
          isConnected={ws.isConnected}
          telemetry={ws.telemetry}
          onOpenGuide={() => setIsGuideOpen(true)}
        />

        {/* ── 2. Left Wing: Spatial Scene Matrix & Radar (Auto-Collapses when Tracker is OFF) ── */}
        <div className="left-panel-wrapper">
          <EntityMatrix
            scene={ws.scene}
            pointedTarget={ws.telemetry?.pointed_target}
            onSelectEntity={handleSelectEntity}
            filterMode={ws.telemetry?.target_filter_mode}
            onSetTargetFilter={ws.setTargetFilter}
          />
        </div>

        {/* ── 3. Center HUD: Tactical Primary Sensor Feed (Dynamically Expands) ── */}
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

        {/* ── 4. Right Wing: Neural Inspection Terminal (Appears ONLY when Inspecting Objects) ── */}
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

        {/* ── 5. Bottom Deck: Auto-Popping Holographic Command Dock ── */}
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

        {/* ── 6. Holographic Gesture Command Manual Modal ── */}
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
