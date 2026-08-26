"""
AURA FastAPI Web Server (Milestone 9)
Serves the React dashboard, REST API, and WebSocket endpoints.
Mounts the built frontend from web/dist/ as static files.
"""

import os
import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .api import router as api_router, set_bridge as api_set_bridge
from .ws import websocket_endpoint, set_bridge as ws_set_bridge
from .bridge import AuraBridge

logger = logging.getLogger("AURA.Server")


def create_app(bridge: Optional[AuraBridge] = None) -> FastAPI:
    """
    Creates and configures the FastAPI application.

    Args:
        bridge: Optional AuraBridge instance. If None, endpoints
                return 503 until a bridge is injected.
    """
    app = FastAPI(
        title="AURA Dashboard",
        description="Adaptive Understanding & Reasoning Architecture — Interactive Web Dashboard",
        version="0.9.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS for development (Vite dev server on port 5173)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Inject bridge into API and WS modules
    if bridge:
        api_set_bridge(bridge)
        ws_set_bridge(bridge)

    # Mount REST API routes
    app.include_router(api_router)

    # Mount WebSocket endpoint
    app.add_api_websocket_route("/ws", websocket_endpoint)

    # Serve built React frontend from web/dist/
    web_dist_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "web", "dist",
    )

    if os.path.isdir(web_dist_path):
        # Serve static assets (JS, CSS, images)
        assets_path = os.path.join(web_dist_path, "assets")
        if os.path.isdir(assets_path):
            app.mount(
                "/assets",
                StaticFiles(directory=assets_path),
                name="assets",
            )

        # Serve index.html for SPA routing (catch-all)
        index_path = os.path.join(web_dist_path, "index.html")

        @app.get("/")
        async def serve_root():
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "AURA Dashboard API is running. Build the frontend with: cd web && npm run build"}

        @app.get("/{path:path}")
        async def serve_spa(path: str):
            # Check if file exists in dist directory
            file_path = os.path.join(web_dist_path, path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            # Otherwise, serve index.html for SPA client-side routing
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "AURA Dashboard"}

        logger.info(f"Serving frontend from: {web_dist_path}")
    else:
        @app.get("/")
        async def serve_root():
            return {
                "message": "AURA Dashboard API is running.",
                "note": "Frontend not built. Run: cd web && npm install && npm run build",
                "api_docs": "/docs",
            }
        logger.info("Frontend build not found. API-only mode.")

    return app


def run_server(
    bridge: AuraBridge,
    host: str = "0.0.0.0",
    port: int = 8420,
    log_level: str = "info",
) -> None:
    """
    Starts the AURA web server with uvicorn.

    Args:
        bridge: AuraBridge instance connecting to the vision pipeline.
        host: Bind address. Default: 0.0.0.0.
        port: Server port. Default: 8420.
        log_level: Logging level for uvicorn.
    """
    import uvicorn

    app = create_app(bridge)
    logger.info(f"Starting AURA Dashboard at http://{host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
    )
