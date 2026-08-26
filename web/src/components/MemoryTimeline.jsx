import { useState, useEffect } from 'react';
import { useAuraAPI } from '../hooks/useAuraAPI';

/**
 * Episodic Memory Timeline — Visual timeline of recently observed objects.
 */

const EVENT_ICONS = {
  person: '👤',
  laptop: '💻',
  notebook: '📓',
  book: '📕',
  smartphone: '📱',
  phone: '📱',
  cup: '☕',
  bottle: '🍶',
  pen: '🖊️',
  backpack: '🎒',
  headphones: '🎧',
  chair: '🪑',
  keyboard: '⌨️',
};

function formatTimeAgo(timestamp) {
  const now = Date.now() / 1000;
  const diff = Math.max(0, Math.floor(now - timestamp));
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

export default function MemoryTimeline({ memoryResponse }) {
  const [events, setEvents] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const api = useAuraAPI();

  // Load initial history
  useEffect(() => {
    api.getMemoryHistory(15)
      .then((data) => setEvents(data.events || []))
      .catch(() => {});
  }, []);

  // Refresh periodically
  useEffect(() => {
    const interval = setInterval(() => {
      api.getMemoryHistory(15)
        .then((data) => setEvents(data.events || []))
        .catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Handle incoming memory search responses via WS
  useEffect(() => {
    if (memoryResponse) {
      setSearchResult(memoryResponse);
      setIsSearching(false);
    }
  }, [memoryResponse]);

  const handleSearch = async () => {
    const q = searchQuery.trim();
    if (!q) return;
    setIsSearching(true);
    try {
      const result = await api.searchMemory(q);
      setSearchResult(result);
    } catch {
      setSearchResult({ found: false, description: 'Search failed.' });
    }
    setIsSearching(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSearch();
  };

  return (
    <div className="glass-card" style={styles.container}>
      <span className="panel-title"><span className="icon">🧠</span> Episodic Memory</span>

      {/* Memory Search */}
      <div style={styles.searchBar}>
        <input
          type="text"
          className="input-field"
          placeholder='Where did I leave my keys?'
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          style={styles.searchInput}
          id="memory-search-input"
        />
        <button
          className="btn btn-primary"
          onClick={handleSearch}
          disabled={isSearching}
          style={styles.searchBtn}
          id="memory-search-btn"
        >
          🔍
        </button>
      </div>

      {/* Search result */}
      {searchResult && (
        <div style={{
          ...styles.searchResult,
          borderColor: searchResult.found ? 'rgba(0,229,153,0.3)' : 'rgba(255,176,32,0.3)',
        }} className="animate-fade-in">
          <span style={styles.resultIcon}>{searchResult.found ? '📍' : '❓'}</span>
          <p style={styles.resultText}>{searchResult.description}</p>
        </div>
      )}

      {/* Timeline Events */}
      <div style={styles.timeline}>
        {events.length === 0 ? (
          <p style={styles.emptyText}>No memory events recorded yet...</p>
        ) : (
          events.map((ev, i) => (
            <div key={ev.id || i} style={styles.timelineItem} className="animate-fade-in">
              <div style={styles.timeDot}>
                <div style={styles.dot} />
                {i < events.length - 1 && <div style={styles.connector} />}
              </div>
              <div style={styles.eventCard}>
                <div style={styles.eventHeader}>
                  <span style={styles.eventIcon}>
                    {EVENT_ICONS[ev.class_name] || '📦'}
                  </span>
                  <span style={styles.eventName}>{ev.class_name}</span>
                  <span className="badge badge-cyan" style={{ fontSize: '0.6rem' }}>
                    {ev.spatial_region}
                  </span>
                </div>
                <div style={styles.eventMeta}>
                  <span style={styles.eventTime}>{formatTimeAgo(ev.timestamp)}</span>
                  {ev.associated_text && (
                    <span style={styles.eventText}>📝 "{ev.associated_text}"</span>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    flex: 1,
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
  searchResult: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '8px',
    margin: '0 var(--space-md) var(--space-sm)',
    padding: '10px 12px',
    background: 'rgba(0, 229, 153, 0.06)',
    borderRadius: 'var(--radius-md)',
    border: '1px solid',
  },
  resultIcon: {
    fontSize: '1.1rem',
    flexShrink: 0,
  },
  resultText: {
    fontSize: '0.78rem',
    color: 'var(--text-secondary)',
    lineHeight: 1.5,
  },
  timeline: {
    flex: 1,
    overflowY: 'auto',
    padding: '0 var(--space-md) var(--space-md)',
  },
  emptyText: {
    fontSize: '0.78rem',
    color: 'var(--text-muted)',
    textAlign: 'center',
    padding: 'var(--space-lg)',
    opacity: 0.5,
  },
  timelineItem: {
    display: 'flex',
    gap: '10px',
  },
  timeDot: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    paddingTop: '6px',
  },
  dot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    background: 'var(--accent-cyan)',
    boxShadow: '0 0 6px rgba(0,240,255,0.3)',
    flexShrink: 0,
  },
  connector: {
    width: '1px',
    flex: 1,
    background: 'rgba(0,240,255,0.15)',
    margin: '4px 0',
  },
  eventCard: {
    flex: 1,
    padding: '6px 0 12px',
  },
  eventHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
  },
  eventIcon: {
    fontSize: '0.9rem',
  },
  eventName: {
    fontFamily: 'var(--font-display)',
    fontSize: '0.78rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
    textTransform: 'capitalize',
  },
  eventMeta: {
    display: 'flex',
    gap: '8px',
    marginTop: '3px',
    flexWrap: 'wrap',
  },
  eventTime: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.65rem',
    color: 'var(--text-muted)',
  },
  eventText: {
    fontSize: '0.65rem',
    color: 'var(--text-muted)',
    fontStyle: 'italic',
  },
};
