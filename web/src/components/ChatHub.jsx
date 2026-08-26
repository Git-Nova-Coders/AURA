import { useState, useEffect, useRef } from 'react';

/**
 * Chat Hub — Conversational interface with animated mic button and message history.
 */
export default function ChatHub({ onSendChat, chatResponse }) {
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
  }, [messages]);

  const handleSend = () => {
    const query = inputText.trim();
    if (!query) return;

    setMessages((prev) => [
      ...prev,
      { role: 'user', text: query, timestamp: Date.now() },
    ]);
    setInputText('');
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

  return (
    <div className="glass-card" style={styles.container}>
      <span className="panel-title"><span className="icon">💬</span> Voice & Chat Hub</span>

      {/* Messages */}
      <div style={styles.messagesContainer}>
        {messages.length === 0 && (
          <div style={styles.emptyState}>
            <span style={styles.emptyIcon}>🤖</span>
            <p style={styles.emptyText}>Ask AURA about what it sees</p>
            <p style={styles.emptyHint}>"What do you see?" · "Where is the laptop?" · "How to use the notebook?"</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              ...styles.messageBubble,
              ...(msg.role === 'user' ? styles.userBubble : styles.assistantBubble),
            }}
            className="animate-fade-in"
          >
            <div style={styles.messageHeader}>
              <span style={styles.messageRole}>
                {msg.role === 'user' ? '🧑 You' : '◉ AURA'}
              </span>
              {msg.intent && (
                <span className={`badge ${intentBadgeColor(msg.intent)}`}>
                  {msg.intent.replace(/_/g, ' ')}
                </span>
              )}
            </div>
            <p style={styles.messageText}>{msg.text}</p>
            {msg.sources?.length > 0 && (
              <div style={styles.sources}>
                {msg.sources.map((s, j) => (
                  <span key={j} style={styles.sourceChip}>{s}</span>
                ))}
              </div>
            )}
          </div>
        ))}

        {/* Thinking indicator */}
        {isThinking && (
          <div style={{ ...styles.messageBubble, ...styles.assistantBubble }} className="animate-fade-in">
            <div style={styles.thinking}>
              <span style={styles.thinkingDot} className="animate-pulse">●</span>
              <span style={{ ...styles.thinkingDot, animationDelay: '0.2s' }} className="animate-pulse">●</span>
              <span style={{ ...styles.thinkingDot, animationDelay: '0.4s' }} className="animate-pulse">●</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <div style={styles.inputBar}>
        <input
          type="text"
          className="input-field"
          placeholder="Ask AURA anything..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          style={styles.inputField}
          id="chat-input"
        />
        <button
          className="btn btn-primary"
          onClick={handleSend}
          disabled={!inputText.trim() || isThinking}
          style={styles.sendBtn}
          id="chat-send-btn"
        >
          ➤
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    flex: 1,
    minHeight: '300px',
    overflow: 'hidden',
    padding: 0,
  },
  messagesContainer: {
    flex: 1,
    overflowY: 'auto',
    padding: 'var(--space-sm) var(--space-md)',
    display: 'flex',
    flexDirection: 'column',
    gap: 'var(--space-sm)',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    textAlign: 'center',
    opacity: 0.5,
    padding: 'var(--space-xl)',
  },
  emptyIcon: {
    fontSize: '2.5rem',
    marginBottom: '8px',
  },
  emptyText: {
    fontFamily: 'var(--font-display)',
    fontSize: '0.9rem',
    color: 'var(--text-secondary)',
    fontWeight: 500,
  },
  emptyHint: {
    fontSize: '0.7rem',
    color: 'var(--text-muted)',
    marginTop: '8px',
    lineHeight: 1.5,
  },
  messageBubble: {
    padding: '10px 14px',
    borderRadius: 'var(--radius-md)',
    maxWidth: '90%',
  },
  userBubble: {
    alignSelf: 'flex-end',
    background: 'linear-gradient(135deg, rgba(138,43,226,0.2), rgba(138,43,226,0.1))',
    border: '1px solid rgba(138,43,226,0.25)',
  },
  assistantBubble: {
    alignSelf: 'flex-start',
    background: 'linear-gradient(135deg, rgba(0,229,153,0.1), rgba(0,240,255,0.05))',
    border: '1px solid rgba(0,229,153,0.2)',
  },
  messageHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '4px',
  },
  messageRole: {
    fontFamily: 'var(--font-display)',
    fontSize: '0.72rem',
    fontWeight: 600,
    color: 'var(--text-muted)',
  },
  messageText: {
    fontSize: '0.82rem',
    lineHeight: 1.55,
    color: 'var(--text-primary)',
  },
  sources: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '4px',
    marginTop: '6px',
  },
  sourceChip: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.6rem',
    padding: '1px 6px',
    borderRadius: 'var(--radius-full)',
    background: 'rgba(0,240,255,0.08)',
    color: 'var(--text-muted)',
    border: '1px solid rgba(0,240,255,0.1)',
  },
  thinking: {
    display: 'flex',
    gap: '6px',
    padding: '4px 0',
  },
  thinkingDot: {
    color: 'var(--accent-emerald)',
    fontSize: '0.8rem',
  },
  inputBar: {
    display: 'flex',
    gap: 'var(--space-sm)',
    padding: 'var(--space-sm) var(--space-md) var(--space-md)',
    borderTop: '1px solid rgba(255,255,255,0.04)',
  },
  inputField: {
    flex: 1,
    fontSize: '0.82rem',
  },
  sendBtn: {
    padding: '10px 16px',
    fontSize: '1rem',
    minWidth: '44px',
  },
};
