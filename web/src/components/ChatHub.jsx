import { useState, useEffect, useRef } from 'react';

/**
 * Chat Hub — Conversational Multimodal AI interface with speech visualizer,
 * prompt suggestions, and vector RAG citations.
 */
export default function ChatHub({ onSendChat, chatResponse, isVoiceListening }) {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const messagesEndRef = useRef(null);

  // Handle incoming chat responses
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

  // Auto-scroll on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isThinking]);

  const handleSend = (textToSend = null) => {
    const query = (textToSend || inputText).trim();
    if (!query) return;

    setMessages((prev) => [
      ...prev,
      { role: 'user', text: query, timestamp: Date.now() },
    ]);
    if (!textToSend) setInputText('');
    setIsThinking(true);

    if (onSendChat) onSendChat(query);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const intentBadgeColor = (intent) => {
    const map = {
      scene_summary: 'badge-cyan',
      object_info: 'badge-emerald',
      object_location: 'badge-violet',
      memory_spatial: 'badge-amber',
      memory_temporal: 'badge-amber',
      document_rag: 'badge-cyan',
      ocr_read: 'badge-emerald',
      reliability_check: 'badge-violet',
    };
    return map[intent] || 'badge-cyan';
  };

  const promptSuggestions = [
    'What do you see in the scene?',
    'Describe the locked target object',
    'Read all text detected by OCR',
    'Where is the laptop located?',
  ];

  return (
    <div className="chat-hub-card glass-card">
      <div className="chat-header">
        <div className="chat-title-wrap">
          <span className="chat-icon">💬</span>
          <span className="panel-title">Multimodal Conversational AI</span>
        </div>
        {isVoiceListening && (
          <div className="audio-visualizer">
            <div className="audio-bar"></div>
            <div className="audio-bar"></div>
            <div className="audio-bar"></div>
            <div className="audio-bar"></div>
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty-state">
            <span className="empty-icon animate-pulse">🤖</span>
            <p className="empty-title">AURA Multimodal Intelligence</p>
            <p className="empty-desc">Ask questions about scene objects, spatial layout, or user gestures.</p>
            <div className="quick-prompts">
              {promptSuggestions.map((prompt, idx) => (
                <button
                  key={idx}
                  className="prompt-chip glass-card"
                  onClick={() => handleSend(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`chat-bubble animate-fade-in ${
              msg.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-assistant'
            }`}
          >
            <div className="bubble-header">
              <span className="bubble-role">
                {msg.role === 'user' ? '🧑 You' : '⬢ AURA AI'}
              </span>
              {msg.intent && (
                <span className={`badge ${intentBadgeColor(msg.intent)}`}>
                  {msg.intent.replace(/_/g, ' ')}
                </span>
              )}
            </div>
            <p className="bubble-text">{msg.text}</p>
            {msg.sources?.length > 0 && (
              <div className="bubble-sources">
                <span className="sources-label">Sources:</span>
                {msg.sources.map((s, si) => (
                  <span key={si} className="badge badge-cyan">
                    📄 {s}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}

        {isThinking && (
          <div className="chat-bubble chat-bubble-assistant animate-pulse">
            <div className="bubble-header">
              <span className="bubble-role">⬢ AURA AI</span>
              <span className="badge badge-cyan">Reasoning</span>
            </div>
            <div className="thinking-dots">
              <span>●</span>
              <span>●</span>
              <span>●</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <div className="chat-input-bar">
        <input
          type="text"
          className="input-field"
          placeholder="Ask AURA anything about what it sees..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          id="chat-input-field"
        />
        <button
          className="btn-send"
          onClick={() => handleSend()}
          disabled={!inputText.trim()}
          id="chat-send-btn"
        >
          <span>Send</span>
          <span className="send-arrow">➔</span>
        </button>
      </div>
    </div>
  );
}
