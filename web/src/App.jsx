import { useState, useCallback } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import BrandHeader from './components/BrandHeader';
import LiveFeed from './components/LiveFeed';
import SpatialRadar from './components/SpatialRadar';
import ChatHub from './components/ChatHub';
import MemoryTimeline from './components/MemoryTimeline';
import RAGDrawer from './components/RAGDrawer';
import TelemetryPanel from './components/TelemetryPanel';
import './App.css';

/**
 * AURA Dashboard — Main Application Layout
 * 3-column grid: Left (Radar + Memory) | Center (Feed) | Right (Chat + RAG)
 * Bottom: Telemetry bar
 */
export default function App() {
  const ws = useWebSocket();
  const [rightTab, setRightTab] = useState('chat'); // 'chat' | 'rag'

  // Handle object click on Live Feed → trigger knowledge lookup via chat
  const handleObjectClick = useCallback((entity) => {
    const query = `Tell me about the ${entity.class_name}`;
    ws.sendChat(query);
    setRightTab('chat');
  }, [ws]);

  return (
    <div className="dashboard">
      {/* ── Header ── */}
      <div className="dashboard-header">
        <BrandHeader isConnected={ws.isConnected} telemetry={ws.telemetry} />
      </div>

      {/* ── Left Column: Spatial Radar + Memory Timeline ── */}
      <div className="dashboard-left">
        <SpatialRadar scene={ws.scene} />
        <MemoryTimeline memoryResponse={ws.memoryResponse} />
      </div>

      {/* ── Center: Live Video Feed ── */}
      <div className="dashboard-center">
        <LiveFeed
          frame={ws.lastFrame}
          scene={ws.scene}
          onObjectClick={handleObjectClick}
        />
      </div>

      {/* ── Right Column: Chat + RAG (Tabbed) ── */}
      <div className="dashboard-right">
        {/* Tab Switcher */}
        <div className="tab-bar glass-card">
          <button
            className={`tab-btn ${rightTab === 'chat' ? 'tab-active' : ''}`}
            onClick={() => setRightTab('chat')}
            id="tab-chat"
          >
            💬 Chat Hub
          </button>
          <button
            className={`tab-btn ${rightTab === 'rag' ? 'tab-active' : ''}`}
            onClick={() => setRightTab('rag')}
            id="tab-rag"
          >
            📚 RAG Search
          </button>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {rightTab === 'chat' ? (
            <ChatHub
              onSendChat={ws.sendChat}
              chatResponse={ws.chatResponse}
            />
          ) : (
            <RAGDrawer
              onSearchRAG={ws.searchRAG}
              ragResponse={ws.ragResponse}
            />
          )}
        </div>
      </div>

      {/* ── Footer: Telemetry Bar ── */}
      <div className="dashboard-footer">
        <TelemetryPanel
          telemetry={ws.telemetry}
          onToggleSAHI={ws.toggleSAHI}
          onToggleTracking={ws.toggleTracking}
        />
      </div>
    </div>
  );
}
