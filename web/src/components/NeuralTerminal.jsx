import { useState, useEffect, useRef } from 'react';
import AudioVisualizerCanvas from './AudioVisualizerCanvas';
import { soundFX } from '../utils/audioFx';

/**
 * NeuralTerminal — Right Wing Multimodal AI Intelligence Terminal
 * Displays active object inspection dossiers (scrollable timeline),
 * AI Conversational Reasoning, and RAG Technical Manuals.
 */
export default function NeuralTerminal({
  onSendChat,
  chatResponse,
  ragResponse,
  onSearchRAG,
  selectedEntity,
  inspectResponse,
  inspectHistory = [],
  onDeselectAll,
  onRemoveInspectItem,
  isVoiceListening,
}) {
  const [activeTab, setActiveTab] = useState('dossier'); // 'dossier' | 'chat' | 'rag'
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [ragQuery, setRagQuery] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-switch to dossier and add to chat stream when inspect response arrives
  useEffect(() => {
    if (inspectResponse) {
      setActiveTab('dossier');
      soundFX.playBeep(880, 0.08, 'triangle', 0.08);
      const targetName = (inspectResponse.target || 'PINCHED_OBJECT').toUpperCase();
      const explanation = inspectResponse.response?.response_text || 'Tactical analysis completed.';
      
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: `👌 [TACTICAL DOSSIER: ${targetName}]\n${explanation}`,
          intent: 'INSPECTION',
          sources: inspectResponse.response?.sources || [],
          timestamp: Date.now(),
        },
      ]);
    }
  }, [inspectResponse]);

  // Handle incoming chat response
  useEffect(() => {
    if (chatResponse) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          text: chatResponse.response_text,
          intent: chatResponse.intent,
          sources: chatResponse.sources || [],
          timestamp: Date.now(),
        },
      ]);
      setIsThinking(false);
    }
  }, [chatResponse]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isThinking]);

  const handleSend = (textToSend = null) => {
    const query = (textToSend || inputText).trim();
    if (!query) return;

    soundFX.playBeep(1100, 0.05, 'sine', 0.05);
    setMessages((prev) => [
      ...prev,
      { role: 'user', text: query, timestamp: Date.now() },
    ]);
    if (!textToSend) setInputText('');
    setIsThinking(true);

    if (onSendChat) onSendChat(query);
  };

  const handleRAGSearch = () => {
    if (!ragQuery.trim()) return;
    soundFX.playBeep(950, 0.05, 'sine', 0.05);
    if (onSearchRAG) onSearchRAG(ragQuery.trim());
  };

  const quickPrompts = [
    'What objects do you see?',
    'Inspect the pointed target in detail',
    'Read all text detected by OCR',
    'Explain the spatial layout',
  ];

  return (
    <div className="neural-terminal-panel glass-panel">
      {/* ── Terminal Tab Switcher & Close Header ── */}
      <div className="terminal-tab-bar">
        <button
          className={`terminal-tab ${activeTab === 'dossier' ? 'tab-active' : ''}`}
          onClick={() => { soundFX.playToggle(true); setActiveTab('dossier'); }}
        >
          👌 DOSSIERS ({inspectHistory.length || (selectedEntity || inspectResponse ? 1 : 0)})
        </button>
        <button
          className={`terminal-tab ${activeTab === 'chat' ? 'tab-active' : ''}`}
          onClick={() => { soundFX.playToggle(true); setActiveTab('chat'); }}
        >
          💬 AI TERMINAL
        </button>
        <button
          className={`terminal-tab ${activeTab === 'rag' ? 'tab-active' : ''}`}
          onClick={() => { soundFX.playToggle(true); setActiveTab('rag'); }}
        >
          📚 RAG
        </button>

        {/* 1-Click Deselect All & Close Button */}
        <button
          className="btn-deselect-all"
          onClick={() => {
            soundFX.playClick();
            if (onDeselectAll) onDeselectAll();
          }}
          title="Deselect all inspected objects and close panel"
        >
          ✕ DESELECT ALL
        </button>
      </div>

      {/* ── Tab Content ── */}
      <div className="terminal-body">
        {/* 1. SCROLLABLE OBJECT DOSSIER TAB */}
        {activeTab === 'dossier' && (
          <div className="tab-pane-dossier animate-fade-in">
            {/* Scrollable list of all inspected objects */}
            <div className="dossier-scroll-list">
              {inspectHistory.length > 0 ? (
                inspectHistory.map((item, idx) => (
                  <div key={item.id || idx} className="dossier-timeline-card glass-card animate-slide-in">
                    <div className="dossier-card-hdr">
                      <div className="dossier-card-title-group">
                        <span className="dossier-item-icon">👌</span>
                        <div>
                          <h4 className="dossier-item-title">{(item.target || item.class_name || 'OBJECT').toUpperCase()}</h4>
                          <span className="dossier-item-meta">
                            {new Date(item.timestamp || Date.now()).toLocaleTimeString()} // {item.spatial_pos || 'ACTIVE FOCUS'}
                          </span>
                        </div>
                      </div>
                      <div className="dossier-card-actions">
                        <span className="badge badge-cyan">{Math.round((item.confidence || 0.95) * 100)}% Match</span>
                        {onRemoveInspectItem && (
                          <button
                            className="btn-dismiss-item"
                            onClick={(e) => {
                              e.stopPropagation();
                              onRemoveInspectItem(idx);
                            }}
                            title="Remove from history"
                          >
                            ✕
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Stats Grid */}
                    <div className="dossier-mini-grid">
                      <div className="dossier-mini-stat">
                        <span className="mini-lbl">STATUS</span>
                        <span className="mini-val val-emerald">INSPECTED</span>
                      </div>
                      <div className="dossier-mini-stat">
                        <span className="mini-lbl">BOUNDS</span>
                        <span className="mini-val">
                          {item.bbox ? `[${Math.round(item.bbox[0])}, ${Math.round(item.bbox[1])}]` : 'TRACKED'}
                        </span>
                      </div>
                    </div>

                    {/* Extracted OCR text if present */}
                    {item.ocr_texts?.length > 0 && (
                      <div className="dossier-block-compact">
                        <span className="block-title">EXTRACTED TEXT</span>
                        <div className="dossier-tags">
                          {item.ocr_texts.map((t, ti) => (
                            <span key={ti} className="badge badge-cyan">"{t}"</span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Intelligence explanation */}
                    <div className="dossier-explanation-box">
                      <p>{item.explanation || item.response_text || 'Tactical telemetry and spatial orientation recorded.'}</p>
                    </div>

                    <button
                      className="btn-dossier-ask-compact"
                      onClick={() => {
                        handleSend(`Explain what this ${(item.target || item.class_name || 'object')} is and its context in the scene.`);
                        setActiveTab('chat');
                      }}
                    >
                      💬 Inquire AI About This
                    </button>
                  </div>
                ))
              ) : (
                /* Fallback single inspected entity view */
                <div className="dossier-timeline-card glass-card animate-slide-in">
                  <div className="dossier-header">
                    <span className="dossier-icon">👌</span>
                    <div>
                      <h3 className="dossier-title">
                        {(inspectResponse?.target || selectedEntity?.class_name || 'PINCHED_OBJECT').toUpperCase()}
                      </h3>
                      <span className="dossier-sub">TACTICAL INTELLIGENCE PROFILE</span>
                    </div>
                  </div>

                  <div className="dossier-stats-grid">
                    <div className="dossier-card">
                      <span className="card-lbl">CONFIDENCE</span>
                      <span className="card-val val-cyan">
                        {inspectResponse?.response?.confidence
                          ? Math.round(inspectResponse.response.confidence * 100)
                          : selectedEntity?.confidence
                          ? Math.round(selectedEntity.confidence * 100)
                          : 98}%
                      </span>
                    </div>
                    <div className="dossier-card">
                      <span className="card-lbl">VERIFICATION</span>
                      <span className="card-val val-emerald">VERIFIED</span>
                    </div>
                    <div className="dossier-card">
                      <span className="card-lbl">SPATIAL LOC</span>
                      <span className="card-val">
                        {inspectResponse?.response?.spatial_pos || selectedEntity?.spatial_pos || 'CENTER'}
                      </span>
                    </div>
                  </div>

                  {(inspectResponse?.response?.ocr_texts?.length > 0 || selectedEntity?.ocr_texts?.length > 0) && (
                    <div className="dossier-block">
                      <span className="block-title">EXTRACTED OCR TEXT</span>
                      <div className="dossier-tags">
                        {(inspectResponse?.response?.ocr_texts || selectedEntity?.ocr_texts || []).map((txt, idx) => (
                          <span key={idx} className="badge badge-cyan">"{txt}"</span>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="dossier-block">
                    <span className="block-title">AI ANALYSIS & REASONING</span>
                    <div className="dossier-explanation">
                      <p>
                        {inspectResponse?.response?.response_text ||
                          'Pinch or click any entity in the scene to inspect and view real-time intelligence.'}
                      </p>
                    </div>
                  </div>

                  <button
                    className="btn-dossier-ask"
                    onClick={() => {
                      const target = selectedEntity?.class_name || inspectResponse?.target || 'object';
                      handleSend(`Explain what this ${target} is and its context in the scene.`);
                      setActiveTab('chat');
                    }}
                  >
                    💬 Deep Conversational Inquiry
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 2. CHAT TAB */}
        {activeTab === 'chat' && (
          <div className="tab-pane-chat animate-fade-in">
            {/* Audio Wave Header */}
            <div className="voice-status-header">
              <span className="voice-label">
                {isVoiceListening ? '🎙️ VOICE LISTENING...' : 'NEURAL CONVERSATION STREAM'}
              </span>
              <AudioVisualizerCanvas isActive={isVoiceListening || isThinking} />
            </div>

            {/* Chat Feed */}
            <div className="terminal-chat-feed">
              {messages.length === 0 && (
                <div className="terminal-empty-state">
                  <span className="terminal-beacon animate-pulse">⬢</span>
                  <p className="terminal-empty-title">AURA MULTIMODAL REASONER</p>
                  <p className="terminal-empty-sub">Ask about scene entities, memory events, or RAG manuals.</p>
                  <div className="quick-chip-list">
                    {quickPrompts.map((q, idx) => (
                      <button
                        key={idx}
                        className="quick-chip glass-card"
                        onClick={() => handleSend(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`terminal-msg animate-fade-in ${
                    msg.role === 'user' ? 'msg-user' : 'msg-aura'
                  }`}
                >
                  <div className="msg-header">
                    <span className="msg-author">{msg.role === 'user' ? '🧑 OPERATOR' : '⬢ AURA_AI'}</span>
                    {msg.intent && <span className="badge badge-cyan">{msg.intent}</span>}
                  </div>
                  <p className="msg-text">{msg.text}</p>
                  {msg.sources?.length > 0 && (
                    <div className="msg-citations">
                      {msg.sources.map((s, si) => (
                        <span key={si} className="citation-badge">
                          📄 {s}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}

              {isThinking && (
                <div className="terminal-msg msg-aura animate-pulse">
                  <div className="msg-header">
                    <span className="msg-author">⬢ AURA_AI</span>
                    <span className="badge badge-cyan">DECODING</span>
                  </div>
                  <div className="matrix-thinking-dots">
                    <span>●</span><span>●</span><span>●</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Line */}
            <div className="terminal-input-row">
              <input
                type="text"
                className="terminal-input"
                placeholder="Enter query or use Call Me 🤙 gesture..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleSend(); }}
              />
              <button
                className="btn-terminal-send"
                onClick={() => handleSend()}
                disabled={!inputText.trim()}
              >
                TRANSMIT ➔
              </button>
            </div>
          </div>
        )}

        {/* 3. RAG MANUALS TAB */}
        {activeTab === 'rag' && (
          <div className="tab-pane-rag animate-fade-in">
            <div className="rag-input-box">
              <input
                type="text"
                className="terminal-input"
                placeholder="Search indexed technical manuals..."
                value={ragQuery}
                onChange={(e) => setRagQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleRAGSearch(); }}
              />
              <button className="btn-terminal-send" onClick={handleRAGSearch}>
                SEARCH
              </button>
            </div>

            <div className="rag-results-list">
              {ragResponse?.results?.length > 0 ? (
                ragResponse.results.map((res, idx) => (
                  <div key={idx} className="rag-result-card glass-card">
                    <div className="rag-res-hdr">
                      <span className="res-doc-id">📄 {res.source || `DOC_${idx + 1}`}</span>
                      <span className="badge badge-cyan">{Math.round(res.score * 100)}% Match</span>
                    </div>
                    <p className="res-snippet">{res.text || res.content}</p>
                  </div>
                ))
              ) : (
                <div className="rag-empty">
                  <span className="empty-beacon">📚</span>
                  <p>VECTOR STORE KNOWLEDGE RETRIEVER</p>
                  <span className="empty-sub">Query indexed equipment manuals and knowledge documents</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
