import { useState, useCallback } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import BrandHeader from './components/BrandHeader';
import LiveFeed from './components/LiveFeed';
import SpatialRadar from './components/SpatialRadar';
import ChatHub from './components/ChatHub';
import MemoryTimeline from './components/MemoryTimeline';
import RAGDrawer from './components/RAGDrawer';
import TelemetryPanel from './components/TelemetryPanel';
import GestureGuideModal from './components/GestureGuideModal';
import ObjectInspector from './components/ObjectInspector';
import './App.css';

/**
 * AURA Dashboard — Main Application Layout
 * 3-column cybernetic layout:
 * - Left: Spatial Radar + Object Inspector / Memory Timeline
 * - Center: Cybernetic Live Feed + Holographic Scanlines + Real-time Gesture HUD
 * - Right: Multimodal Chat Hub + RAG Manual Search
 * - Bottom: Interactive Control Deck (SAHI, Tracking, OCR, Voice Toggles)
 */
export default function App() {
  const ws = useWebSocket();
  const [rightTab, setRightTab] = useState('chat'); // 'chat' | 'rag' | 'inspector'
  const [isGuideOpen, setIsGuideOpen] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState(null);

  // Handle object selection via click or pinch gesture
  const handleObjectClick = useCallback((entity) => {
    setSelectedEntity(entity);
    ws.inspectEntity(entity.class_name);
    setRightTab('inspector');
  }, [ws]);

  const handleSendInspectionQuery = useCallback((query) => {
    ws.sendChat(query);
    setRightTab('chat');
  }, [ws]);

  return (
    <div className="dashboard">
      {/* ── Brand & Status Header ── */}
      <div className="dashboard-header">
        <BrandHeader
          isConnected={ws.isConnected}
          telemetry={ws.telemetry}
          onOpenGuide={() => setIsGuideOpen(true)}
        />
      </div>

      {/* ── Left Column: Spatial Radar + Memory Timeline ── */}
      <div className="dashboard-left">
        <SpatialRadar
          scene={ws.scene}
          pointedTarget={ws.telemetry?.pointed_target}
          onSelectEntity={handleObjectClick}
        />
        <MemoryTimeline memoryResponse={ws.memoryResponse} />
      </div>

      {/* ── Center: Cybernetic Live Video Perception Feed ── */}
      <div className="dashboard-center">
        <LiveFeed
          frame={ws.lastFrame}
          scene={ws.scene}
          telemetry={ws.telemetry}
          activeToast={ws.activeToast}
          onObjectClick={handleObjectClick}
          onOpenGuide={() => setIsGuideOpen(true)}
        />
      </div>

      {/* ── Right Column: Chat + Object Inspector + RAG (Tabbed) ── */}
      <div className="dashboard-right">
        {/* Tab Switcher */}
        <div className="tab-bar glass-card">
          <button
            className={`tab-btn ${rightTab === 'chat' ? 'tab-active' : ''}`}
            onClick={() => setRightTab('chat')}
            id="tab-chat"
          >
            💬 Chat AI
          </button>
          <button
            className={`tab-btn ${rightTab === 'inspector' ? 'tab-active' : ''}`}
            onClick={() => setRightTab('inspector')}
            id="tab-inspector"
          >
            👌 Inspector
          </button>
          <button
            className={`tab-btn ${rightTab === 'rag' ? 'tab-active' : ''}`}
            onClick={() => setRightTab('rag')}
            id="tab-rag"
          >
            📚 RAG Manuals
          </button>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {rightTab === 'chat' && (
            <ChatHub
              onSendChat={ws.sendChat}
              chatResponse={ws.chatResponse}
              isVoiceListening={ws.telemetry?.voice_listening}
            />
          )}

          {rightTab === 'inspector' && (
            <ObjectInspector
              targetEntity={selectedEntity}
              inspectResponse={ws.inspectResponse}
              onSendQuery={handleSendInspectionQuery}
              onClose={() => setRightTab('chat')}
            />
          )}

          {rightTab === 'rag' && (
            <RAGDrawer
              onSearchRAG={ws.searchRAG}
              ragResponse={ws.ragResponse}
            />
          )}
        </div>
      </div>

      {/* ── Footer: Interactive Control Deck ── */}
      <div className="dashboard-footer">
        <TelemetryPanel
          telemetry={ws.telemetry}
          onToggleSAHI={ws.toggleSAHI}
          onToggleTracking={ws.toggleTracking}
          onToggleOCR={ws.toggleOCR}
          onToggleVoice={ws.toggleVoice}
        />
      </div>

      {/* ── 3D Hand Gesture Interactive Guide Modal ── */}
      <GestureGuideModal
        isOpen={isGuideOpen}
        onClose={() => setIsGuideOpen(false)}
        activeGesture={ws.telemetry?.active_gesture}
      />
    </div>
  );
}
