import { useState, useCallback } from 'react';
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
 * AAA Sci-Fi Aerospace HUD Layout
 */
export default function App() {
  const ws = useWebSocket();
  const [isGuideOpen, setIsGuideOpen] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState(null);

  // Handle entity selection via click or pinch gesture
  const handleSelectEntity = useCallback((entity) => {
    setSelectedEntity(entity);
    ws.inspectEntity(entity.class_name);
  }, [ws]);

  return (
    <>
      {/* ── 0. Cyber Particle Background Canvas ── */}
      <CyberBackground />

      {/* ── Master HUD Grid ── */}
      <div className="aura-command-center">
        {/* ── 1. Top Aerospace Telemetry Header ── */}
        <HoloHeader
          isConnected={ws.isConnected}
          telemetry={ws.telemetry}
          onOpenGuide={() => setIsGuideOpen(true)}
        />

        {/* ── 2. Left Wing: Spatial Scene Matrix & Radar ── */}
        <EntityMatrix
          scene={ws.scene}
          pointedTarget={ws.telemetry?.pointed_target}
          onSelectEntity={handleSelectEntity}
          filterMode={ws.telemetry?.target_filter_mode}
          onSetTargetFilter={ws.setTargetFilter}
        />

        {/* ── 3. Center HUD: Tactical Primary Sensor Feed ── */}
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

        {/* ── 4. Right Wing: Neural Intelligence & Multimodal Terminal ── */}
        <NeuralTerminal
          onSendChat={ws.sendChat}
          chatResponse={ws.chatResponse}
          ragResponse={ws.ragResponse}
          onSearchRAG={ws.searchRAG}
          selectedEntity={selectedEntity}
          inspectResponse={ws.inspectResponse}
          isVoiceListening={ws.telemetry?.voice_listening}
        />

        {/* ── 5. Bottom Deck: Tactical Control Matrix & LED Switches ── */}
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
