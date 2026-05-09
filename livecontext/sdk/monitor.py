"""LiveContext SDK for programmatic monitoring."""

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

import httpx
import websockets

from livecontext.core.tokenizer import Tokenizer, get_tokenizer
from livecontext.server.models import (
    ContextMessage,
    ContextSnapshot,
    Eviction,
    EvictionStrategy,
    Role,
    SessionInfo,
)

logger = logging.getLogger(__name__)


class LiveContextMonitor:
    """SDK client for LiveContext monitoring."""
    
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        ws_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize LiveContext monitor.
        
        Args:
            server_url: LiveContext server URL
            ws_url: WebSocket URL (defaults to ws://server_url)
            api_key: Optional API key
        """
        self.server_url = server_url.rstrip("/")
        self.ws_url = ws_url or server_url.replace("http://", "ws://").replace("https://", "wss://")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=30.0)
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._session_id: Optional[str] = None
        self._message_handlers: List[Callable] = []
        self._running = False
    
    async def create_session(
        self,
        model_name: str = "unknown",
        provider: str = "unknown",
        max_tokens: int = 4096
    ) -> SessionInfo:
        """Create a new monitoring session.
        
        Args:
            model_name: Model name
            provider: Provider name
            max_tokens: Maximum tokens
            
        Returns:
            SessionInfo
        """
        session = SessionInfo(
            id=str(uuid.uuid4()),
            model_name=model_name,
            provider=provider,
            max_tokens=max_tokens
        )
        
        response = await self._client.post(
            f"{self.server_url}/api/sessions",
            json=session.model_dump(mode="json")
        )
        response.raise_for_status()
        
        self._session_id = session.id
        logger.info(f"Created session: {session.id}")
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            SessionInfo or None
        """
        try:
            response = await self._client.get(
                f"{self.server_url}/api/sessions/{session_id}"
            )
            response.raise_for_status()
            return SessionInfo(**response.json())
        except httpx.HTTPStatusError:
            return None
    
    async def list_sessions(self, active_only: bool = False) -> List[SessionInfo]:
        """List all sessions.
        
        Args:
            active_only: Only return active sessions
            
        Returns:
            List of SessionInfo
        """
        response = await self._client.get(
            f"{self.server_url}/api/sessions",
            params={"active_only": active_only}
        )
        response.raise_for_status()
        return [SessionInfo(**s) for s in response.json()]
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if deleted
        """
        try:
            response = await self._client.delete(
                f"{self.server_url}/api/sessions/{session_id}"
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError:
            return False
    
    async def get_snapshots(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[ContextSnapshot]:
        """Get snapshots for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of snapshots
            
        Returns:
            List of ContextSnapshot
        """
        response = await self._client.get(
            f"{self.server_url}/api/sessions/{session_id}/snapshots",
            params={"limit": limit}
        )
        response.raise_for_status()
        return [ContextSnapshot(**s) for s in response.json()]
    
    async def get_snapshot(self, snapshot_id: str) -> Optional[ContextSnapshot]:
        """Get snapshot by ID.
        
        Args:
            snapshot_id: Snapshot ID
            
        Returns:
            ContextSnapshot or None
        """
        try:
            response = await self._client.get(
                f"{self.server_url}/api/snapshots/{snapshot_id}"
            )
            response.raise_for_status()
            return ContextSnapshot(**response.json())
        except httpx.HTTPStatusError:
            return None
    
    async def proxy_openai(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4",
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Proxy request to OpenAI through LiveContext.
        
        Args:
            messages: Messages to send
            model: Model name
            stream: Whether to stream
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        from livecontext.server.models import ProxyRequest
        
        request = ProxyRequest(
            provider="openai",
            model=model,
            messages=messages,
            stream=stream,
            **kwargs
        )
        
        async with self._client.stream(
            "POST",
            f"{self.server_url}/api/proxy/openai",
            json=request.model_dump(mode="json")
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line
    
    async def proxy_anthropic(
        self,
        messages: List[Dict[str, Any]],
        model: str = "claude-3-opus-20240229",
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Proxy request to Anthropic through LiveContext.
        
        Args:
            messages: Messages to send
            model: Model name
            stream: Whether to stream
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        from livecontext.server.models import ProxyRequest
        
        request = ProxyRequest(
            provider="anthropic",
            model=model,
            messages=messages,
            stream=stream,
            **kwargs
        )
        
        async with self._client.stream(
            "POST",
            f"{self.server_url}/api/proxy/anthropic",
            json=request.model_dump(mode="json")
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line
    
    async def proxy_ollama(
        self,
        messages: List[Dict[str, Any]],
        model: str = "llama2",
        stream: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Proxy request to Ollama through LiveContext.
        
        Args:
            messages: Messages to send
            model: Model name
            stream: Whether to stream
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        from livecontext.server.models import ProxyRequest
        
        request = ProxyRequest(
            provider="ollama",
            model=model,
            messages=messages,
            stream=stream,
            **kwargs
        )
        
        async with self._client.stream(
            "POST",
            f"{self.server_url}/api/proxy/ollama",
            json=request.model_dump(mode="json")
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                yield line
    
    async def connect_websocket(
        self,
        session_id: Optional[str] = None
    ) -> None:
        """Connect to WebSocket for real-time updates.
        
        Args:
            session_id: Session ID to subscribe to
        """
        sid = session_id or self._session_id
        if not sid:
            raise ValueError("No session ID provided")
        
        self._ws = await websockets.connect(
            f"{self.ws_url}/ws/{sid}"
        )
        self._running = True
        
        # Start listening
        asyncio.create_task(self._listen_websocket())
        
        logger.info(f"Connected to WebSocket: {sid}")
    
    async def _listen_websocket(self) -> None:
        """Listen for WebSocket messages."""
        if not self._ws:
            return
        
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    # Notify handlers
                    for handler in self._message_handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(data)
                            else:
                                handler(data)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message[:100]}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket closed")
        finally:
            self._running = False
    
    def on_message(self, handler: Callable) -> Callable:
        """Register a message handler.
        
        Args:
            handler: Callback function
            
        Returns:
            The handler (for decorator use)
        """
        self._message_handlers.append(handler)
        return handler
    
    async def disconnect_websocket(self) -> None:
        """Disconnect WebSocket."""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self._running = False
    
    async def close(self) -> None:
        """Close all connections."""
        await self.disconnect_websocket()
        await self._client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class ContextTracker:
    """High-level context tracking utility."""
    
    def __init__(
        self,
        monitor: LiveContextMonitor,
        session_id: str,
        model_name: str,
        provider: str,
        max_tokens: int = 4096
    ):
        """Initialize context tracker.
        
        Args:
            monitor: LiveContextMonitor instance
            session_id: Session ID
            model_name: Model name
            provider: Provider name
            max_tokens: Maximum tokens
        """
        self.monitor = monitor
        self.session_id = session_id
        self.model_name = model_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.tokenizer = get_tokenizer(model_name, provider)
        self._messages: List[ContextMessage] = []
    
    def add_message(
        self,
        role: Role,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextMessage:
        """Add a message to context.
        
        Args:
            role: Message role
            content: Message content
            metadata: Optional metadata
            
        Returns:
            Created message
        """
        message = ContextMessage(
            role=role,
            content=content,
            token_count=self.tokenizer.count_tokens(content),
            metadata=metadata or {}
        )
        
        self._messages.append(message)
        return message
    
    def get_messages(self) -> List[ContextMessage]:
        """Get all messages."""
        return self._messages.copy()
    
    def get_token_count(self) -> int:
        """Get total token count."""
        return self.tokenizer.count_messages_tokens(self._messages)
    
    def get_utilization(self) -> float:
        """Get context utilization percentage."""
        if self.max_tokens == 0:
            return 0.0
        return (self.get_token_count() / self.max_tokens) * 100
    
    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
    
    def to_openai_format(self) -> List[Dict[str, str]]:
        """Convert messages to OpenAI format."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self._messages
        ]
    
    def to_anthropic_format(self) -> List[Dict[str, str]]:
        """Convert messages to Anthropic format."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self._messages
        ]


# Convenience functions
async def create_monitor(
    server_url: str = "http://localhost:8000",
    **kwargs
) -> LiveContextMonitor:
    """Create and return a monitor instance.
    
    Args:
        server_url: Server URL
        **kwargs: Additional arguments
        
    Returns:
        LiveContextMonitor
    """
    return LiveContextMonitor(server_url, **kwargs)


async def quick_session(
    model_name: str = "gpt-4",
    provider: str = "openai",
    server_url: str = "http://localhost:8000"
) -> str:
    """Quickly create a session and return ID.
    
    Args:
        model_name: Model name
        provider: Provider name
        server_url: Server URL
        
    Returns:
        Session ID
    """
    async with LiveContextMonitor(server_url) as monitor:
        session = await monitor.create_session(model_name, provider)
        return session.id


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/daksh/7May/projects/livecontext")
    
    print("Testing LiveContext SDK")
    print("=" * 50)
    
    async def test_sdk():
        async with LiveContextMonitor() as monitor:
            # Test health check
            try:
                response = await monitor._client.get("http://localhost:8000/health")
                if response.status_code == 200:
                    print("✅ Server is healthy")
                else:
                    print("❌ Server health check failed")
                    return
            except Exception as e:
                print(f"❌ Server not running: {e}")
                return
            
            # Test session creation
            session = await monitor.create_session("gpt-4", "openai", 4096)
            print(f"✅ Created session: {session.id[:8]}...")
            
            # Test listing sessions
            sessions = await monitor.list_sessions()
            print(f"✅ Listed {len(sessions)} sessions")
            
            # Test getting session
            fetched = await monitor.get_session(session.id)
            if fetched:
                print(f"✅ Fetched session: {fetched.model_name}")
            
            # Test context tracker
            tracker = ContextTracker(
                monitor, session.id, "gpt-4", "openai", 4096
            )
            
            tracker.add_message(Role.SYSTEM, "You are a helpful assistant.")
            tracker.add_message(Role.USER, "Hello!")
            
            print(f"✅ Tracker has {len(tracker.get_messages())} messages")
            print(f"✅ Token count: {tracker.get_token_count()}")
            print(f"✅ Utilization: {tracker.get_utilization():.2f}%")
            
            # Cleanup
            await monitor.delete_session(session.id)
            print("✅ Deleted test session")
    
    asyncio.run(test_sdk())
    print("\n✅ SDK tests complete!")
