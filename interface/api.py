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
    """Returns the latest detection list."""
    bridge = _get_bridge()
    return {"detections": bridge.get_detections()}


@router.post("/chat")
async def post_chat(request: ChatRequest):
    """Sends a user query to the conversation engine and returns a grounded response."""
    bridge = _get_bridge()
    try:
        result = bridge.send_chat(request.query)
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@router.post("/rag/search")
async def search_rag(request: RAGSearchRequest):
    """Searches indexed RAG documents with cosine similarity ranking."""
    bridge = _get_bridge()
    try:
        result = bridge.search_rag(request.query, top_k=request.top_k)
        return result
    except Exception as e:
        logger.error(f"RAG search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG search error: {str(e)}")


@router.get("/rag/documents")
async def get_rag_documents():
    """Lists all indexed RAG documents."""
    bridge = _get_bridge()
    return {"documents": bridge.get_rag_documents()}


@router.post("/memory/search")
async def search_memory(request: MemorySearchRequest):
    """Searches episodic memory for when/where an object was last seen."""
    bridge = _get_bridge()
    try:
        result = bridge.search_memory(request.query)
        return result
    except Exception as e:
        logger.error(f"Memory search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Memory search error: {str(e)}")


@router.get("/memory/history")
async def get_memory_history(limit: int = 20):
    """Returns the most recent episodic memory events."""
    bridge = _get_bridge()
    return {"events": bridge.get_memory_history(limit=min(limit, 50))}


@router.get("/telemetry")
async def get_telemetry():
    """Returns current performance telemetry (FPS, latency, tracks, etc.)."""
    bridge = _get_bridge()
    return bridge.get_telemetry()


@router.get("/config")
async def get_config():
    """Returns the current runtime configuration state."""
    bridge = _get_bridge()
    status = bridge.get_status()
    return {
        "sahi_enabled": status["sahi_enabled"],
        "tracking_enabled": status["tracking_enabled"],
        "ocr_enabled": status["ocr_enabled"],
        "rag_enabled": status["rag_enabled"],
        "memory_enabled": status["memory_enabled"],
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

    return {"updated": True, "config": result}
