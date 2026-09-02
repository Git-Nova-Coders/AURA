import { useState, useEffect, useRef } from 'react';
import AudioVisualizerCanvas from './AudioVisualizerCanvas';
import NeuralCore from './hud/NeuralCore';
import HudFrame from './hud/HudFrame';
import { soundFX } from '../utils/audioFx';

/**
 * NeuralTerminal — AURA V2 Neural Intelligence & Reasoning Terminal.
 * Slides/fades out of the environment when an object is inspected.
 * Features visible neural reasoning pipeline:
 * VISION ... ✓ | TRACK ... ✓ | OCR ... ✓ | KNOWLEDGE ... ✓ | GROUNDING ...
 * Target dossier cards, AI conversation stream, and vector manual RAG.
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
  const [reasoningStage, setReasoningStage] = useState(0); // 0 to 5
  const messagesEndRef = useRef(null);

  // Animate neural reasoning stages when a new inspection arrives
  useEffect(() => {
    if (inspectResponse) {
      setActiveTab('dossier');
      soundFX.playBeep(880, 0.08, 'triangle', 0.08);

      // Step-by-step reasoning illumination sequence
      setReasoningStage(1); // Vision & Track
      const t1 = setTimeout(() => setReasoningStage(2), 200); // OCR
      const t2 = setTimeout(() => setReasoningStage(3), 400); // Knowledge
      const t3 = setTimeout(() => setReasoningStage(4), 650); // Correlating & Grounding
      const t4 = setTimeout(() => setReasoningStage(5), 900); // Complete

      const targetName = (inspectResponse.target || 'TARGET').toUpperCase();
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

      return () => {
        clearTimeout(t1);
        clearTimeout(t2);
        clearTimeout(t3);
        clearTimeout(t4);
      };
    }
  }, [inspectResponse]);

  // Handle incoming conversational response
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
    <div className="v2-neural-terminal-container">
      <HudFrame
        sensorId="NEURAL_REASONER"
        coords="STATE: ACTIVE"
        status="COGNITION"
        className="v2-neural-hud-frame"
      >
        {/* ── Top Terminal Bar with Tabs & Deselect All ── */}
        <div className="v2-terminal-tab-bar">
          <button
            className={`v2-tab ${activeTab === 'dossier' ? 'v2-tab-active' : ''}`}
            onClick={() => { soundFX.playToggle(true); setActiveTab('dossier'); }}
            title="Target Intelligence Dossier"
          >
            👌 DOSSIER {inspectHistory.length > 0 ? `(${inspectHistory.length})` : ''}
          </button>
          <button
            className={`v2-tab ${activeTab === 'chat' ? 'v2-tab-active' : ''}`}
            onClick={() => { soundFX.playToggle(true); setActiveTab('chat'); }}
            title="Multimodal Cognitive Reasoning Stream"
          >
            💬 REASONING
          </button>
          <button
            className={`v2-tab ${activeTab === 'rag' ? 'v2-tab-active' : ''}`}
            onClick={() => { soundFX.playToggle(true); setActiveTab('rag'); }}
            title="Knowledge Base / RAG Documents"
          >
            📚 RAG
          </button>

          {/* 1-Click Deselect All Button */}
          <button
            className="v2-btn-close-inspect"
            onClick={() => {
              soundFX.playClick();
              if (onDeselectAll) onDeselectAll();
            }}
            title="Deselect all inspected objects and close intelligence layer"
          >
            ✕
          </button>
        </div>

        {/* ── Body Content ── */}
        <div className="v2-terminal-body">
          {/* 1. DOSSIER TAB */}
          {activeTab === 'dossier' && (
            <div className="v2-pane-dossier animate-fade-in">
              {/* Visible Neural Reasoning Status Animation */}
              <div className="v2-reasoning-banner">
                <div className="v2-reasoning-core-box">
                  <NeuralCore size={42} isThinking={isThinking || reasoningStage < 5} statusText="AURA" />
                </div>
                <div className="v2-reasoning-flow">
                  <span className="flow-title">NEURAL REASONING PIPELINE</span>
                  <div className="flow-steps">
                    <span className={`flow-step ${reasoningStage >= 1 ? 'step-done' : ''}`}>VISION ✓</span>
                    <span className="flow-arrow">→</span>
                    <span className={`flow-step ${reasoningStage >= 2 ? 'step-done' : ''}`}>TRACK ✓</span>
                    <span className="flow-arrow">→</span>
                    <span className={`flow-step ${reasoningStage >= 3 ? 'step-done' : ''}`}>RAG ✓</span>
                    <span className="flow-arrow">→</span>
                    <span className={`flow-step ${reasoningStage >= 4 ? 'step-done' : ''}`}>GROUND ✓</span>
                  </div>
                </div>
              </div>

              {/* Scrollable Dossier Cards */}
              <div className="v2-dossier-scroll-list">
                {inspectHistory.length > 0 ? (
                  inspectHistory.map((item, idx) => (
                    <div key={item.id || idx} className="v2-dossier-card animate-slide-in">
                      <div className="dossier-card-top">
                        <div className="dossier-title-area">
                          <span className="dossier-icon">👌</span>
                          <div>
                            <h4 className="dossier-item-name">{(item.target || item.class_name || 'OBJECT').toUpperCase()}</h4>
                            <span className="dossier-time-stamp">
                              {new Date(item.timestamp || Date.now()).toLocaleTimeString()} // {item.spatial_pos || 'CENTER'}
                            </span>
                          </div>
                        </div>

                        <div className="dossier-badge-area">
                          <span className="v2-badge-emerald">{Math.round((item.confidence || 0.95) * 100)}% CONF</span>
                          {onRemoveInspectItem && (
                            <button
                              className="v2-btn-dismiss"
                              onClick={(e) => {
                                e.stopPropagation();
                                onRemoveInspectItem(idx);
                              }}
                              title="Dismiss from history"
                            >
                              ✕
                            </button>
                          )}
                        </div>
                      </div>

                      {/* State Pills */}
                      <div className="dossier-check-pills">
                        <span className="chk-pill">◉ IDENTIFIED</span>
                        <span className="chk-pill">◉ TRACKED</span>
                        <span className="chk-pill">◉ GROUNDED</span>
                      </div>

                      {/* Extracted OCR Text Tags */}
                      {item.ocr_texts?.length > 0 && (
                        <div className="dossier-ocr-block">
                          <span className="block-label">EXTRACTED TEXT</span>
                          <div className="dossier-text-tags">
                            {item.ocr_texts.map((txt, ti) => (
                              <span key={ti} className="v2-badge-cyan">"{txt}"</span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Explanation */}
                      <div className="dossier-explanation-paragraph">
                        <p>{item.explanation || item.response_text || 'Tactical telemetry and spatial orientation recorded.'}</p>
                      </div>

                      <button
                        className="v2-btn-inquire"
                        onClick={() => {
                          handleSend(`Explain what this ${(item.target || item.class_name || 'object')} is and its context.`);
                          setActiveTab('chat');
                        }}
                      >
                        💬 Inquire AI About This
                      </button>
                    </div>
                  ))
                ) : (
                  <div className="v2-dossier-card animate-slide-in">
                    <div className="dossier-card-top">
                      <div className="dossier-title-area">
                        <span className="dossier-icon">👌</span>
                        <div>
                          <h4 className="dossier-item-name">
                            {(inspectResponse?.target || selectedEntity?.class_name || 'OBJECT').toUpperCase()}
                          </h4>
                          <span className="dossier-time-stamp">REAL-TIME TELEMETRY</span>
                        </div>
                      </div>
                      <span className="v2-badge-emerald">
                        {Math.round((inspectResponse?.response?.confidence || selectedEntity?.confidence || 0.95) * 100)}% CONF
                      </span>
                    </div>

                    <div className="dossier-explanation-paragraph">
                      <p>
                        {inspectResponse?.response?.response_text ||
                          'Point your index finger or perform a Pinch gesture at any object to inspect real-time intelligence.'}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 2. CHAT TAB */}
          {activeTab === 'chat' && (
            <div className="v2-pane-chat animate-fade-in">
              <div className="v2-voice-wave-header">
                <span className="v2-voice-lbl">
                  {isVoiceListening ? '🎙️ VOICE LISTENING...' : 'NEURAL CONVERSATION STREAM'}
                </span>
                <AudioVisualizerCanvas isActive={isVoiceListening || isThinking} />
              </div>

              <div className="v2-chat-feed">
                {messages.length === 0 && (
                  <div className="v2-chat-empty">
                    <span className="v2-chat-beacon animate-pulse">⬢</span>
                    <p className="v2-empty-title">AURA MULTIMODAL REASONER</p>
                    <p className="v2-empty-sub">Ask about scene entities, memory events, or RAG manuals.</p>
                    <div className="v2-quick-chips">
                      {quickPrompts.map((q, idx) => (
                        <button key={idx} className="v2-quick-chip" onClick={() => handleSend(q)}>
                          {q}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`v2-chat-msg animate-fade-in ${msg.role === 'user' ? 'msg-user' : 'msg-aura'}`}
                  >
                    <div className="v2-msg-header">
                      <span className="v2-msg-author">{msg.role === 'user' ? '🧑 OPERATOR' : '⬢ AURA_AI'}</span>
                      {msg.intent && <span className="v2-badge-cyan">{msg.intent}</span>}
                    </div>
                    <p className="v2-msg-text">{msg.text}</p>
                    {msg.sources?.length > 0 && (
                      <div className="v2-msg-sources">
                        {msg.sources.map((s, si) => (
                          <span key={si} className="v2-badge-citation">📄 {s}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}

                {isThinking && (
                  <div className="v2-chat-msg msg-aura animate-pulse">
                    <div className="v2-msg-header">
                      <span className="v2-msg-author">⬢ AURA_AI</span>
                      <span className="v2-badge-cyan">REASONING</span>
                    </div>
                    <div className="v2-thinking-dots">
                      <span>●</span><span>●</span><span>●</span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="v2-chat-input-row">
                <input
                  type="text"
                  className="v2-input-text"
                  placeholder="Inquire AI or use Call Me 🤙 gesture..."
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleSend(); }}
                />
                <button
                  className="v2-btn-send"
                  onClick={() => handleSend()}
                  disabled={!inputText.trim()}
                >
                  TRANSMIT ➔
                </button>
              </div>
            </div>
          )}

          {/* 3. RAG TAB */}
          {activeTab === 'rag' && (
            <div className="v2-pane-rag animate-fade-in">
              <div className="v2-rag-search-row">
                <input
                  type="text"
                  className="v2-input-text"
                  placeholder="Search technical manuals & vector docs..."
                  value={ragQuery}
                  onChange={(e) => setRagQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleRAGSearch(); }}
                />
                <button className="v2-btn-send" onClick={handleRAGSearch}>
                  SEARCH
                </button>
              </div>

              <div className="v2-rag-results-list">
                {ragResponse?.results?.length > 0 ? (
                  ragResponse.results.map((res, idx) => (
                    <div key={idx} className="v2-rag-card">
                      <div className="rag-card-hdr">
                        <span className="rag-doc-name">📄 {res.source || `DOC_${idx + 1}`}</span>
                        <span className="v2-badge-cyan">{Math.round(res.score * 100)}% Match</span>
                      </div>
                      <p className="rag-doc-text">{res.text || res.content}</p>
                    </div>
                  ))
                ) : (
                  <div className="v2-rag-empty">
                    <span className="empty-book">📚</span>
                    <p>VECTOR STORE KNOWLEDGE RETRIEVER</p>
                    <span className="empty-sub">Query indexed equipment manuals and technical guides</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </HudFrame>
    </div>
  );
}
