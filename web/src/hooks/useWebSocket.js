import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Custom React hook for WebSocket connection to AURA backend.
 * Handles auto-reconnect, message parsing, and bidirectional communication.
 */
export function useWebSocket(url = null) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastFrame, setLastFrame] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [scene, setScene] = useState(null);
  const [chatResponse, setChatResponse] = useState(null);
  const [ragResponse, setRagResponse] = useState(null);
  const [memoryResponse, setMemoryResponse] = useState(null);
  const [configUpdate, setConfigUpdate] = useState(null);

  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const reconnectDelay = useRef(1000);

  const wsUrl = url || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setIsConnected(true);
        reconnectDelay.current = 1000;
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          switch (msg.type) {
            case 'frame':
              setLastFrame(msg.data);
              break;
            case 'telemetry':
              setTelemetry(msg.data);
              break;
            case 'scene':
              setScene(msg.data);
              break;
            case 'chat_response':
              setChatResponse(msg.data);
              break;
            case 'rag_response':
              setRagResponse(msg.data);
              break;
            case 'memory_response':
              setMemoryResponse(msg.data);
              break;
            case 'config_update':
              setConfigUpdate(msg.data);
              break;
            case 'pong':
              break;
            default:
              break;
          }
        } catch {
          // Ignore non-JSON messages
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;
        // Auto-reconnect with exponential backoff
        reconnectTimer.current = setTimeout(() => {
          reconnectDelay.current = Math.min(reconnectDelay.current * 1.5, 10000);
          connect();
        }, reconnectDelay.current);
      };

      ws.onerror = () => {
        ws.close();
      };

      wsRef.current = ws;
    } catch {
      // Connection failed, will retry
    }
  }, [wsUrl]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const sendMessage = useCallback((msg) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  const sendChat = useCallback((query) => {
    sendMessage({ type: 'chat', query });
  }, [sendMessage]);

  const searchRAG = useCallback((query) => {
    sendMessage({ type: 'rag_search', query });
  }, [sendMessage]);

  const searchMemory = useCallback((query) => {
    sendMessage({ type: 'memory_search', query });
  }, [sendMessage]);

  const toggleSAHI = useCallback(() => {
    sendMessage({ type: 'toggle_sahi' });
  }, [sendMessage]);

  const toggleTracking = useCallback(() => {
    sendMessage({ type: 'toggle_tracking' });
  }, [sendMessage]);

  return {
    isConnected,
    lastFrame,
    telemetry,
    scene,
    chatResponse,
    ragResponse,
    memoryResponse,
    configUpdate,
    sendChat,
    searchRAG,
    searchMemory,
    toggleSAHI,
    toggleTracking,
    sendMessage,
  };
}
