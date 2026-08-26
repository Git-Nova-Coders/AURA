import { useState, useEffect } from 'react';

/**
 * RAG Drawer — Document search interface with similarity-scored results.
 */
export default function RAGDrawer({ onSearchRAG, ragResponse }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  useEffect(() => {
    if (ragResponse) {
      setResults(ragResponse);
      setIsSearching(false);
    }
  }, [ragResponse]);

  const handleSearch = () => {
    const q = query.trim();
    if (!q) return;
    setIsSearching(true);
    if (onSearchRAG) onSearchRAG(q);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSearch();
  };

  const getScoreColor = (score) => {
    if (score >= 0.5) return { bg: 'var(--accent-emerald-dim)', color: 'var(--accent-emerald)', border: 'rgba(0,229,153,0.3)' };
    if (score >= 0.3) return { bg: 'rgba(255,176,32,0.15)', color: 'var(--accent-amber)', border: 'rgba(255,176,32,0.3)' };
    return { bg: 'rgba(255,64,96,0.1)', color: 'var(--accent-red)', border: 'rgba(255,64,96,0.2)' };
  };

  return (
    <div className="glass-card" style={styles.container}>
      <span className="panel-title"><span className="icon">📚</span> Document RAG & Manuals</span>

      {/* Search bar */}
      <div style={styles.searchBar}>
        <input
          type="text"
          className="input-field"
          placeholder="Search manuals, safety guidelines..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          style={styles.searchInput}
          id="rag-search-input"
        />
        <button
          className="btn btn-primary"
          onClick={handleSearch}
          disabled={isSearching || !query.trim()}
          style={styles.searchBtn}
          id="rag-search-btn"
        >
          {isSearching ? '⏳' : '🔍'}
        </button>
      </div>

      {/* Results */}
      <div style={styles.results}>
        {!results && (
          <div style={styles.emptyState}>
            <span style={styles.emptyIcon}>📖</span>
            <p style={styles.emptyText}>Search equipment manuals & safety procedures</p>
            <p style={styles.emptyHint}>e.g. "How to use the laptop" · "Safety guidelines"</p>
          </div>
        )}

        {results && results.count === 0 && (
          <div style={styles.noResults} className="animate-fade-in">
            <p style={styles.noResultsText}>No matching documents found for "{results.query}"</p>
          </div>
        )}

        {results && results.documents?.map((doc, i) => {
          const score = results.scores?.[i] || 0;
          const scoreStyle = getScoreColor(score);
          return (
            <div key={doc.doc_id || i} style={styles.resultCard} className="animate-fade-in">
              <div style={styles.resultHeader}>
                <span style={styles.resultTitle}>{doc.title}</span>
                <div style={{
                  ...styles.scoreBadge,
                  background: scoreStyle.bg,
                  color: scoreStyle.color,
                  borderColor: scoreStyle.border,
                }}>
                  {(score * 100).toFixed(0)}%
                </div>
              </div>

              {/* Similarity bar */}
              <div style={styles.scoreBarTrack}>
                <div style={{
                  ...styles.scoreBarFill,
                  width: `${Math.min(100, score * 100)}%`,
                  background: `linear-gradient(90deg, ${scoreStyle.color}, transparent)`,
                }} />
              </div>

              <p style={styles.resultContent}>
                {doc.content?.slice(0, 200)}{doc.content?.length > 200 ? '...' : ''}
              </p>

              <div style={styles.resultMeta}>
                <span className="badge badge-cyan" style={{ fontSize: '0.58rem' }}>
                  {doc.category}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
    padding: 0,
    minHeight: '200px',
  },
  searchBar: {
    display: 'flex',
    gap: '6px',
    padding: '0 var(--space-md) var(--space-sm)',
  },
  searchInput: {
    flex: 1,
    fontSize: '0.78rem',
    padding: '8px 12px',
  },
  searchBtn: {
    padding: '8px 12px',
    fontSize: '0.85rem',
    minWidth: '38px',
  },
  results: {
    flex: 1,
    overflowY: 'auto',
    padding: '0 var(--space-md) var(--space-md)',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    textAlign: 'center',
    padding: 'var(--space-lg)',
    opacity: 0.5,
  },
  emptyIcon: {
    fontSize: '2rem',
    marginBottom: '8px',
  },
  emptyText: {
    fontFamily: 'var(--font-display)',
    fontSize: '0.82rem',
    color: 'var(--text-secondary)',
    fontWeight: 500,
  },
  emptyHint: {
    fontSize: '0.68rem',
    color: 'var(--text-muted)',
    marginTop: '6px',
  },
  noResults: {
    textAlign: 'center',
    padding: 'var(--space-lg)',
  },
  noResultsText: {
    fontSize: '0.8rem',
    color: 'var(--text-muted)',
  },
  resultCard: {
    padding: '12px',
    background: 'rgba(0,240,255,0.03)',
    border: '1px solid rgba(0,240,255,0.1)',
    borderRadius: 'var(--radius-md)',
    marginBottom: 'var(--space-sm)',
    transition: 'border-color var(--transition-fast)',
  },
  resultHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '8px',
    marginBottom: '6px',
  },
  resultTitle: {
    fontFamily: 'var(--font-display)',
    fontSize: '0.8rem',
    fontWeight: 600,
    color: 'var(--accent-cyan)',
    flex: 1,
  },
  scoreBadge: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.68rem',
    fontWeight: 600,
    padding: '2px 8px',
    borderRadius: 'var(--radius-full)',
    border: '1px solid',
    flexShrink: 0,
  },
  scoreBarTrack: {
    height: '3px',
    background: 'rgba(255,255,255,0.05)',
    borderRadius: '2px',
    marginBottom: '8px',
    overflow: 'hidden',
  },
  scoreBarFill: {
    height: '100%',
    borderRadius: '2px',
    transition: 'width 0.5s ease',
  },
  resultContent: {
    fontSize: '0.72rem',
    color: 'var(--text-secondary)',
    lineHeight: 1.55,
  },
  resultMeta: {
    display: 'flex',
    gap: '6px',
    marginTop: '8px',
  },
};
