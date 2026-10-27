import React from 'react';

/**
 * ObjectInspector — Slide-out detailed inspection panel
 * displaying deep AI analysis, OCR text, RAG knowledge citations, and spatial metrics for an object.
 */
export default function ObjectInspector({ targetEntity, inspectResponse, onSendQuery, onClose }) {
  if (!targetEntity && !inspectResponse) return null;

  const entityName = targetEntity?.class_name || inspectResponse?.target || 'Object';
  const confidence = targetEntity?.confidence ? Math.round(targetEntity.confidence * 100) : 95;
  const isReliable = targetEntity?.is_reliable ?? true;
  const ocrTexts = targetEntity?.ocr_texts || [];
  const bbox = targetEntity?.bbox || [0, 0, 0, 0];
  const responseText = inspectResponse?.response?.response_text || 'Point and Pinch 👌 at an object or click below to inspect with Multimodal AI.';
  const citations = inspectResponse?.response?.rag_citations || [];

  return (
    <div className="inspector-panel glass-card animate-slide-in">
      {/* ── Inspector Header ── */}
      <div className="inspector-header">
        <div className="inspector-title-wrap">
          <span className="inspector-icon">👌</span>
          <div>
            <h3 className="inspector-title">{entityName.toUpperCase()}</h3>
            <span className="inspector-subtitle">Target Entity Telemetry & AI Inspection</span>
          </div>
        </div>
        {onClose && (
          <button className="inspector-close-btn" onClick={onClose} title="Close Inspector">✕</button>
        )}
      </div>

      {/* ── Metric Grid ── */}
      <div className="inspector-metrics">
        <div className="metric-chip">
          <span className="chip-label">CONFIDENCE</span>
          <span className="chip-val chip-val-cyan">{confidence}%</span>
        </div>
        <div className="metric-chip">
          <span className="chip-label">STATUS</span>
          <span className={`chip-val ${isReliable ? 'chip-val-emerald' : 'chip-val-amber'}`}>
            {isReliable ? 'VERIFIED' : 'CALIBRATING'}
          </span>
        </div>
        <div className="metric-chip">
          <span className="chip-label">DIMENSIONS</span>
          <span className="chip-val">{Math.round(bbox[2] - bbox[0])} × {Math.round(bbox[3] - bbox[1])} px</span>
        </div>
      </div>

      {/* ── OCR Texts Detected in BBox ── */}
      {ocrTexts.length > 0 && (
        <div className="inspector-section">
          <span className="section-label">DETECTED TEXT (OCR)</span>
          <div className="ocr-tags">
            {ocrTexts.map((txt, idx) => (
              <span key={idx} className="ocr-tag badge badge-cyan">
                "{txt}"
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ── Multimodal AI Inspection Reasoning ── */}
      <div className="inspector-section">
        <span className="section-label">AI REASONING & CONTEXT</span>
        <div className="ai-explanation-box">
          <p className="ai-text">{responseText}</p>
        </div>
      </div>

      {/* ── RAG Citations ── */}
      {citations.length > 0 && (
        <div className="inspector-section">
          <span className="section-label">RAG MANUAL CITATIONS</span>
          <div className="citations-list">
            {citations.map((cite, idx) => (
              <div key={idx} className="citation-pill">
                📄 {cite.source || cite.doc_id || 'Manual Reference'} ({Math.round((cite.score || 0.8) * 100)}% match)
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Quick Action Buttons ── */}
      <div className="inspector-actions">
        <button
          className="btn-action btn-action-primary"
          onClick={() => onSendQuery(`What is the purpose of this ${entityName} and what should I know about it?`)}
        >
          💬 Ask AURA About {entityName}
        </button>
      </div>
    </div>
  );
}
