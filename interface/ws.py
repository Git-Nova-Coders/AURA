"""
AURA WebSocket Streaming Hub (Milestone 9)
Real-time bidirectional WebSocket endpoint streaming annotated video frames,
telemetry metrics, scene updates, and accepting chat queries from web clients.
"""

import json
import time
import asyncio
import logging
from typing import Set, Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("AURA.WS")

# Global bridge reference — injected at startup
_bridge = None

# Connected clients & event loop
_clients: Set[WebSocket] = set()
_loop: Optional[asyncio.AbstractEventLoop] = None


def broadcast_sync(msg_dict: dict) -> None:
    """Thread-safe event broadcast to all connected web clients."""
    if not _clients:
        return
    msg_json = json.dumps(msg_dict)
    for ws in list(_clients):
        try:
            if _loop and _loop.is_running():
                asyncio.run_coroutine_threadsafe(ws.send_text(msg_json), _loop)
        except Exception:
            pass


def set_bridge(bridge) -> None:
    """Sets the global bridge reference for WebSocket handlers."""
    global _bridge
    _bridge = bridge
    if bridge and hasattr(bridge, "set_event_broadcaster"):
        bridge.set_event_broadcaster(broadcast_sync)


async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Main WebSocket handler for real-time streaming.
    Sends: video frames (base64 JPEG), telemetry, scene updates.
    Receives: chat queries, config toggles.
    """
    global _loop
    _loop = asyncio.get_running_loop()

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

                elif msg_type == "toggle_ocr":
                    if _bridge:
                        new_state = _bridge.toggle_ocr()
                        await websocket.send_text(json.dumps({
                            "type": "config_update",
                            "data": {"ocr_enabled": new_state},
                        }))

                elif msg_type == "toggle_voice":
                    if _bridge:
                        new_state = _bridge.toggle_voice()
                        await websocket.send_text(json.dumps({
                            "type": "config_update",
                            "data": {"voice_listening": new_state},
                        }))

                elif msg_type == "inspect_entity":
                    query = msg.get("query", "")
                    if _bridge:
                        result = _bridge.inspect_target(query)
                        await websocket.send_text(json.dumps({
                            "type": "inspect_response",
                            "data": result,
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
    """Streams annotated video frames as base64 JPEG at ~25 FPS."""
    try:
        while True:
            if _bridge:
                frame_b64 = _bridge.get_frame_base64()
                if frame_b64:
                    await websocket.send_text(json.dumps({
                        "type": "frame",
                        "data": frame_b64,
                    }))
            await asyncio.sleep(1.0 / 25.0)  # ~25 FPS streaming to browser
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def _stream_telemetry(websocket: WebSocket) -> None:
    """Streams telemetry snapshots every 100ms (10 Hz) for real-time responsiveness."""
    try:
        while True:
            if _bridge:
                telemetry = _bridge.get_telemetry()
                await websocket.send_text(json.dumps({
                    "type": "telemetry",
                    "data": telemetry,
                }))
            await asyncio.sleep(0.1)  # 100ms update rate
    except asyncio.CancelledError:
        pass
    except Exception:
        pass
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
