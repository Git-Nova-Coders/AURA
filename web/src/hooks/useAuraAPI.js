import { useCallback } from 'react';

const API_BASE = '/api';

/**
 * Custom React hook for REST API calls to the AURA backend.
 */
export function useAuraAPI() {
  const request = useCallback(async (path, options = {}) => {
    const url = `${API_BASE}${path}`;
    const config = {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    };
    try {
      const res = await fetch(url, config);
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }
      return await res.json();
    } catch (err) {
      console.error(`API error [${path}]:`, err);
      throw err;
    }
  }, []);

  const getStatus = useCallback(() => request('/status'), [request]);
  const getScene = useCallback(() => request('/scene'), [request]);
  const getTelemetry = useCallback(() => request('/telemetry'), [request]);
  const getConfig = useCallback(() => request('/config'), [request]);

  const sendChat = useCallback(
    (query) => request('/chat', { method: 'POST', body: JSON.stringify({ query }) }),
    [request],
  );

  const searchRAG = useCallback(
    (query, topK = 3) =>
      request('/rag/search', { method: 'POST', body: JSON.stringify({ query, top_k: topK }) }),
    [request],
  );

  const getRAGDocuments = useCallback(() => request('/rag/documents'), [request]);

  const searchMemory = useCallback(
    (query) => request('/memory/search', { method: 'POST', body: JSON.stringify({ query }) }),
    [request],
  );

  const getMemoryHistory = useCallback(
    (limit = 20) => request(`/memory/history?limit=${limit}`),
    [request],
  );

  const updateConfig = useCallback(
    (config) => request('/config', { method: 'PUT', body: JSON.stringify(config) }),
    [request],
  );

  return {
    getStatus,
    getScene,
    getTelemetry,
    getConfig,
    sendChat,
    searchRAG,
    getRAGDocuments,
    searchMemory,
    getMemoryHistory,
    updateConfig,
  };
}
