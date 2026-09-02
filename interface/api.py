"""
AURA REST API Routes (Milestone 9)
FastAPI router providing JSON endpoints for scene queries, chat, RAG search,
episodic memory, telemetry, and runtime configuration.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("AURA.API")

router = APIRouter(prefix="/api", tags=["AURA API"])

# Global reference to AuraBridge — injected at startup by server.py
_bridge = None


def set_bridge(bridge) -> None:
    """Sets the global bridge reference for API handlers."""
    global _bridge
    _bridge = bridge


def _get_bridge():
    if _bridge is None:
        raise HTTPException(status_code=503, detail="AURA pipeline not initialized.")
    return _bridge


# ── Request / Response Models ──

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User chat query")


class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="RAG search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Max documents to return")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Object name to search memory for")


class ConfigUpdateRequest(BaseModel):
    sahi_enabled: Optional[bool] = None
    tracking_enabled: Optional[bool] = None
    ocr_enabled: Optional[bool] = None
    gestures_enabled: Optional[bool] = None
    target_filter_mode: Optional[str] = None


# ── Endpoints ──

@router.get("/status")
async def get_status():
    """Returns overall system health and pipeline state."""
    bridge = _get_bridge()
    return bridge.get_status()


@router.get("/scene")
async def get_scene():
    """Returns the current SceneContext with entities, texts, and spatial relations."""
    bridge = _get_bridge()
    scene = bridge.get_scene()
    if not scene:
        return {"timestamp": 0, "frame_index": 0, "entity_count": 0,
                "entities": [], "texts": [], "relations": [],
                "summary": "No scene data available yet.", "frame_shape": [480, 640]}
    return scene


@router.get("/detections")
async def get_detections():
    """Returns latest raw detections from current frame."""
    bridge = _get_bridge()
    return {"detections": bridge.get_detections()}


@router.post("/chat")
async def post_chat(request: ChatRequest):
    """Processes user queries against the multimodal system."""
    bridge = _get_bridge()
    return bridge.send_chat(request.query)


@router.post("/rag/search")
async def search_rag(request: RAGSearchRequest):
    """Performs semantic vector search over manual/spec documents."""
    bridge = _get_bridge()
    return bridge.search_rag(request.query, top_k=request.top_k)


@router.get("/rag/documents")
async def get_rag_documents():
    """Lists all indexed RAG documents with chunk counts."""
    bridge = _get_bridge()
    if not bridge.rag_engine:
        return {"documents": [], "total_chunks": 0}
    return {
        "documents": [
            {"doc_id": doc_id, "total_chunks": len(chunks)}
            for doc_id, chunks in bridge.rag_engine.documents.items()
        ],
        "total_chunks": len(bridge.rag_engine.chunks),
    }


@router.post("/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Queries episodic memory for object interaction history."""
    bridge = _get_bridge()
    return bridge.search_memory(request.query)


@router.get("/memory/history")
async def get_memory_history(limit: int = 20):
    """Retrieves recent scene events from episodic memory."""
    bridge = _get_bridge()
    if not bridge.episodic_memory:
        return {"events": [], "count": 0}
    events = bridge.episodic_memory.get_recent_events(limit=limit)
    return {"events": events, "count": len(events)}


@router.get("/telemetry")
async def get_telemetry():
    """Returns real-time telemetry snapshot (FPS, latency, tracks, features)."""
    bridge = _get_bridge()
    return bridge.get_telemetry()


@router.get("/config")
async def get_config():
    """Returns current runtime configuration toggles."""
    bridge = _get_bridge()
    sahi_active = bool(bridge.detector.sahi_config and bridge.detector.sahi_config.enabled)
    return {
        "sahi_enabled": sahi_active,
        "tracking_enabled": bridge._tracking_enabled,
        "ocr_enabled": bridge._ocr_enabled,
        "gestures_enabled": bridge._gestures_enabled,
        "target_filter_mode": bridge._target_filter_mode.value,
        "memory_enabled": bridge._enable_memory,
        "rag_enabled": bridge._enable_rag,
    }


@router.put("/config")
async def update_config(request: ConfigUpdateRequest):
    """Updates runtime configuration toggles."""
    bridge = _get_bridge()
    result = {}

    if request.sahi_enabled is not None:
        current = bool(bridge.detector.sahi_config and bridge.detector.sahi_config.enabled)
        if request.sahi_enabled != current:
            new_state = bridge.toggle_sahi()
            result["sahi_enabled"] = new_state
        else:
            result["sahi_enabled"] = current

    if request.tracking_enabled is not None:
        if request.tracking_enabled != bridge._tracking_enabled:
            new_state = bridge.toggle_tracking()
            result["tracking_enabled"] = new_state
        else:
            result["tracking_enabled"] = bridge._tracking_enabled

    if request.ocr_enabled is not None:
        if request.ocr_enabled != bridge._ocr_enabled:
            new_state = bridge.set_ocr(request.ocr_enabled)
            result["ocr_enabled"] = new_state
        else:
            result["ocr_enabled"] = bridge._ocr_enabled

    if request.gestures_enabled is not None:
        if request.gestures_enabled != bridge._gestures_enabled:
            new_state = bridge.set_gestures(request.gestures_enabled)
            result["gestures_enabled"] = new_state
        else:
            result["gestures_enabled"] = bridge._gestures_enabled

    if request.target_filter_mode is not None:
        new_mode = bridge.set_target_filter_mode(request.target_filter_mode)
        result["target_filter_mode"] = new_mode

    return {"updated": True, "config": result}
