"""
AURA WebSocket Streaming Hub (Milestone 9)
Real-time bidirectional WebSocket endpoint streaming annotated video frames,
telemetry metrics, scene updates, and accepting chat queries from web clients.
"""

import json
import time
import asyncio
import logging
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("AURA.WS")

# Global bridge reference — injected at startup
_bridge = None

# Connected clients
_clients: Set[WebSocket] = set()


def set_bridge(bridge) -> None:
    """Sets the global bridge reference for WebSocket handlers."""
    global _bridge
    _bridge = bridge


async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Main WebSocket handler for real-time streaming.
    Sends: video frames (base64 JPEG), telemetry, scene updates.
    Receives: chat queries, config toggles.
    """
    await websocket.accept()
    _clients.add(websocket)
    client_id = id(websocket)
    logger.info(f"WebSocket client {client_id} connected. Total: {len(_clients)}")

    # Start background streaming tasks
    frame_task = asyncio.create_task(_stream_frames(websocket))
    telemetry_task = asyncio.create_task(_stream_telemetry(websocket))
    scene_task = asyncio.create_task(_stream_scene(websocket))

    try:
        # Listen for incoming messages from the client
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "chat":
                    query = msg.get("query", "").strip()
                    if query and _bridge:
                        result = _bridge.send_chat(query)
                        await websocket.send_text(json.dumps({
                            "type": "chat_response",
                            "data": result,
                        }))

                elif msg_type == "rag_search":
                    query = msg.get("query", "").strip()
                    if query and _bridge:
                        result = _bridge.search_rag(query)
                        await websocket.send_text(json.dumps({
                            "type": "rag_response",
                            "data": result,
                        }))

                elif msg_type == "memory_search":
                    query = msg.get("query", "").strip()
                    if query and _bridge:
                        result = _bridge.search_memory(query)
                        await websocket.send_text(json.dumps({
                            "type": "memory_response",
                            "data": result,
                        }))

                elif msg_type == "toggle_sahi":
                    if _bridge:
                        new_state = _bridge.toggle_sahi()
                        await websocket.send_text(json.dumps({
                            "type": "config_update",
                            "data": {"sahi_enabled": new_state},
                        }))

                elif msg_type == "toggle_tracking":
                    if _bridge:
                        new_state = _bridge.toggle_tracking()
                        await websocket.send_text(json.dumps({
                            "type": "config_update",
                            "data": {"tracking_enabled": new_state},
                        }))

                elif msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client {client_id}")
            except Exception as e:
                logger.error(f"Error handling WS message: {e}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        _clients.discard(websocket)
        frame_task.cancel()
        telemetry_task.cancel()
        scene_task.cancel()
        logger.info(f"WebSocket client {client_id} cleaned up. Remaining: {len(_clients)}")


async def _stream_frames(websocket: WebSocket) -> None:
    """Streams annotated video frames as base64 JPEG at ~15 FPS."""
    try:
        while True:
            if _bridge:
                frame_b64 = _bridge.get_frame_base64()
                if frame_b64:
                    await websocket.send_text(json.dumps({
                        "type": "frame",
                        "data": frame_b64,
                    }))
            await asyncio.sleep(1.0 / 15.0)  # ~15 FPS streaming to browser
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _stream_telemetry(websocket: WebSocket) -> None:
    """Streams telemetry snapshots every 500ms."""
    try:
        while True:
            if _bridge:
                telemetry = _bridge.get_telemetry()
                await websocket.send_text(json.dumps({
                    "type": "telemetry",
                    "data": telemetry,
                }))
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _stream_scene(websocket: WebSocket) -> None:
    """Streams scene context updates every 1 second."""
    last_frame_index = -1
    try:
        while True:
            if _bridge:
                scene = _bridge.get_scene()
                if scene and scene.get("frame_index", -1) != last_frame_index:
                    last_frame_index = scene.get("frame_index", -1)
                    await websocket.send_text(json.dumps({
                        "type": "scene",
                        "data": scene,
                    }))
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass
