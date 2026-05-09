"""WebSocket manager for real-time streaming."""

import asyncio
import json
import logging
from typing import Dict, List, Set

from livecontext.server.models import ContextSnapshot, Eviction, StreamingEvent

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        # Session ID -> Set of connections
        self.connections: Dict[str, Set] = {}
        # Connection -> Session ID
        self.connection_sessions: Dict = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket, session_id: str) -> None:
        """Register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            session_id: Session ID to subscribe to
        """
        async with self._lock:
            if session_id not in self.connections:
                self.connections[session_id] = set()
            
            self.connections[session_id].add(websocket)
            self.connection_sessions[websocket] = session_id
        
        logger.info(f"WebSocket connected: {session_id}")
    
    async def disconnect(self, websocket) -> None:
        """Unregister a WebSocket connection.
        
        Args:
            websocket: WebSocket connection to remove
        """
        async with self._lock:
            session_id = self.connection_sessions.pop(websocket, None)
            
            if session_id and session_id in self.connections:
                self.connections[session_id].discard(websocket)
                
                if not self.connections[session_id]:
                    del self.connections[session_id]
        
        logger.info(f"WebSocket disconnected: {session_id}")
    
    async def broadcast_to_session(
        self,
        session_id: str,
        event: StreamingEvent
    ) -> int:
        """Broadcast event to all connections for a session.
        
        Args:
            session_id: Session ID
            event: Event to broadcast
            
        Returns:
            Number of connections notified
        """
        message = json.dumps(event.model_dump(mode="json"))
        
        async with self._lock:
            connections = self.connections.get(session_id, set()).copy()
        
        disconnected = []
        sent_count = 0
        
        for conn in connections:
            try:
                await conn.send_text(message)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(conn)
        
        # Clean up disconnected
        for conn in disconnected:
            await self.disconnect(conn)
        
        return sent_count
    
    async def broadcast_snapshot(
        self,
        session_id: str,
        snapshot: ContextSnapshot
    ) -> int:
        """Broadcast snapshot update.
        
        Args:
            session_id: Session ID
            snapshot: Snapshot to broadcast
            
        Returns:
            Number of connections notified
        """
        event = StreamingEvent(
            event_type="snapshot",
            payload=snapshot,
            session_id=session_id
        )
        return await self.broadcast_to_session(session_id, event)
    
    async def broadcast_eviction(
        self,
        session_id: str,
        eviction: Eviction
    ) -> int:
        """Broadcast eviction event.
        
        Args:
            session_id: Session ID
            eviction: Eviction to broadcast
            
        Returns:
            Number of connections notified
        """
        event = StreamingEvent(
            event_type="eviction",
            payload=eviction,
            session_id=session_id
        )
        return await self.broadcast_to_session(session_id, event)
    
    async def broadcast_error(
        self,
        session_id: str,
        error_message: str
    ) -> int:
        """Broadcast error event.
        
        Args:
            session_id: Session ID
            error_message: Error message
            
        Returns:
            Number of connections notified
        """
        event = StreamingEvent(
            event_type="error",
            payload={"error": error_message},
            session_id=session_id
        )
        return await self.broadcast_to_session(session_id, event)
    
    def get_session_count(self, session_id: str) -> int:
        """Get number of connections for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Number of connections
        """
        return len(self.connections.get(session_id, set()))
    
    def get_total_connections(self) -> int:
        """Get total number of active connections.
        
        Returns:
            Total connections
        """
        return len(self.connection_sessions)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of sessions with active connections.
        
        Returns:
            List of session IDs
        """
        return list(self.connections.keys())


# Global WebSocket manager instance
_ws_manager: WebSocketManager = None


def get_websocket_manager() -> WebSocketManager:
    """Get or create global WebSocket manager.
    
    Returns:
        WebSocketManager instance
    """
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager


def reset_websocket_manager() -> None:
    """Reset global WebSocket manager (for testing)."""
    global _ws_manager
    _ws_manager = None
