"""FastAPI application for LiveContext server."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from livecontext.server.db import Database, get_db
from livecontext.server.models import (
    ContextSnapshot,
    ProxyRequest,
    SessionInfo,
    StreamingEvent,
)
from livecontext.server.websocket import get_websocket_manager
from livecontext.proxies.openai_proxy import OpenAIProxyHandler, AzureOpenAIProxyHandler
from livecontext.proxies.anthropic_proxy import AnthropicProxyHandler
from livecontext.proxies.ollama_proxy import OllamaProxyHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
_db: Optional[Database] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _db
    
    # Startup
    logger.info("Starting LiveContext server...")
    _db = get_db()
    
    yield
    
    # Shutdown
    logger.info("Shutting down LiveContext server...")


# Create FastAPI app
app = FastAPI(
    title="LiveContext",
    description="Real-time streaming context window monitor for LLM agents",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "service": "livecontext"
    }


# Sessions API
@app.get("/api/sessions", response_model=List[SessionInfo])
async def list_sessions(active_only: bool = Query(False)):
    """List all sessions."""
    db = get_db()
    return db.list_sessions(active_only=active_only)


@app.post("/api/sessions", response_model=SessionInfo)
async def create_session(session: SessionInfo):
    """Create a new session."""
    db = get_db()
    db.create_session(session)
    return session


@app.get("/api/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session by ID."""
    db = get_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    db = get_db()
    # Soft delete - mark as inactive
    success = db.update_session(session_id, is_active=False)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# Snapshots API
@app.get("/api/sessions/{session_id}/snapshots", response_model=List[ContextSnapshot])
async def get_snapshots(session_id: str, limit: int = Query(100, ge=1, le=1000)):
    """Get snapshots for a session."""
    db = get_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return db.get_snapshots(session_id, limit=limit)


@app.get("/api/snapshots/{snapshot_id}", response_model=ContextSnapshot)
async def get_snapshot(snapshot_id: str):
    """Get snapshot by ID."""
    db = get_db()
    snapshot = db.get_snapshot(snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return snapshot


# Proxy endpoints
@app.post("/api/proxy/openai")
async def proxy_openai(request: ProxyRequest):
    """Proxy request to OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    handler = OpenAIProxyHandler(api_key=api_key)
    
    async def stream_response() -> AsyncGenerator[str, None]:
        try:
            async for chunk in handler.handle_request(request):
                yield chunk
        finally:
            await handler.close()
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )


@app.post("/api/proxy/anthropic")
async def proxy_anthropic(request: ProxyRequest):
    """Proxy request to Anthropic."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    handler = AnthropicProxyHandler(api_key=api_key)
    
    async def stream_response() -> AsyncGenerator[str, None]:
        try:
            async for chunk in handler.handle_request(request):
                yield chunk
        finally:
            await handler.close()
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )


@app.post("/api/proxy/ollama")
async def proxy_ollama(request: ProxyRequest):
    """Proxy request to Ollama."""
    base_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    
    handler = OllamaProxyHandler(base_url=base_url)
    
    async def stream_response() -> AsyncGenerator[str, None]:
        try:
            async for chunk in handler.handle_request(request):
                yield chunk
        finally:
            await handler.close()
    
    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )


# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates."""
    ws_manager = get_websocket_manager()
    
    await websocket.accept()
    await ws_manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive and handle client messages
            data = await websocket.receive_text()
            try:
                import json
                message = json.loads(data)
                
                # Handle ping
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON: {data[:100]}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)


# Stats endpoint
@app.get("/api/stats")
async def get_stats():
    """Get database statistics."""
    db = get_db()
    return db.get_stats()


# Serve static files (frontend)
frontend_path = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")


@app.get("/")
async def root():
    """Root endpoint - serve frontend if available."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "LiveContext API",
        "version": "0.1.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "livecontext.server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
