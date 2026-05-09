"""Base proxy handler for LLM providers."""

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from livecontext.core.tokenizer import Tokenizer, get_tokenizer
from livecontext.server.db import Database, get_db
from livecontext.server.models import (
    ContextMessage,
    ContextSnapshot,
    Eviction,
    EvictionStrategy,
    ProxyRequest,
    ProxyResponse,
    Role,
    SessionInfo,
)

logger = logging.getLogger(__name__)


class ProxyHandler(ABC):
    """Base class for LLM proxy handlers."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        db: Optional[Database] = None
    ):
        """Initialize proxy handler.
        
        Args:
            api_key: API key for the provider
            base_url: Base URL for API (for custom endpoints)
            db: Database instance for persistence
        """
        self.api_key = api_key
        self.base_url = base_url
        self.db = db or get_db()
        self.client = httpx.AsyncClient(timeout=300.0)
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name."""
        pass
    
    @property
    @abstractmethod
    def default_base_url(self) -> str:
        """Return default base URL."""
        pass
    
    @abstractmethod
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[ContextMessage]:
        """Convert provider messages to ContextMessage."""
        pass
    
    @abstractmethod
    def _build_request_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Build request payload for provider API."""
        pass
    
    @abstractmethod
    def _parse_stream_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """Parse a streaming chunk from provider."""
        pass
    
    @abstractmethod
    def _extract_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from response."""
        pass
    
    async def handle_request(
        self,
        request: ProxyRequest,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Handle proxy request with streaming support.
        
        Args:
            request: Proxy request
            session_id: Optional session ID
            
        Yields:
            Streaming chunks
        """
        start_time = time.time()
        
        # Create or get session
        if not session_id:
            session_id = str(uuid.uuid4())
            session = SessionInfo(
                id=session_id,
                model_name=request.model,
                provider=self.provider_name,
                max_tokens=request.max_tokens or 4096
            )
            self.db.create_session(session)
        else:
            session = self.db.get_session(session_id)
            if not session:
                session = SessionInfo(
                    id=session_id,
                    model_name=request.model,
                    provider=self.provider_name,
                    max_tokens=request.max_tokens or 4096
                )
                self.db.create_session(session)
        
        # Convert messages
        messages = self._convert_messages(request.messages)
        
        # Save messages to database
        for msg in messages:
            self.db.save_message(session_id, msg)
        
        # Update session message count
        self.db.update_session(
            session_id,
            message_count=len(messages),
            updated_at=time.time()
        )
        
        # Build request payload
        payload = self._build_request_payload(
            request.messages,
            request.model,
            request.stream,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Make request
        url = f"{self.base_url or self.default_base_url}"
        headers = self._get_headers()
        
        logger.info(f"Proxying request to {self.provider_name}: {request.model}")
        
        tokens_used = 0
        tokens_remaining = session.max_tokens
        
        try:
            if request.stream:
                async for chunk in self._stream_request(url, headers, payload, session_id, messages):
                    yield chunk
            else:
                async for chunk in self._non_stream_request(url, headers, payload, session_id, messages):
                    yield chunk
            
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            error_response = {
                "error": str(e),
                "provider": self.provider_name,
                "session_id": session_id
            }
            yield f"data: {json.dumps(error_response)}\n\n"
        
        finally:
            # Create final snapshot
            processing_time = (time.time() - start_time) * 1000
            
            # Get current messages
            current_messages = self.db.get_messages(session_id)
            
            # Calculate tokens
            tokenizer = get_tokenizer(request.model, self.provider_name)
            total_tokens = tokenizer.count_messages_tokens(current_messages)
            
            snapshot = ContextSnapshot(
                session_id=session_id,
                messages=current_messages,
                total_tokens=total_tokens,
                max_tokens=session.max_tokens,
                utilization_percent=(total_tokens / session.max_tokens * 100) if session.max_tokens > 0 else 0,
                model_name=request.model,
                provider=self.provider_name
            )
            
            self.db.save_snapshot(snapshot)
            
            # Update session
            self.db.update_session(
                session_id,
                message_count=len(current_messages),
                updated_at=time.time()
            )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def _stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        session_id: str,
        messages: List[ContextMessage]
    ) -> AsyncGenerator[str, None]:
        """Make streaming request."""
        full_response = []
        
        async with self.client.stream(
            "POST",
            url,
            headers=headers,
            json=payload
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue
                
                # Parse chunk
                parsed = self._parse_stream_chunk(line)
                if parsed:
                    # Add metadata
                    parsed["_livecontext"] = {
                        "session_id": session_id,
                        "provider": self.provider_name,
                        "timestamp": time.time()
                    }
                    
                    yield f"data: {json.dumps(parsed)}\n\n"
                    
                    # Track content for snapshot
                    content = self._extract_content_from_chunk(parsed)
                    if content:
                        full_response.append(content)
        
        # Save assistant message
        if full_response:
            assistant_content = "".join(full_response)
            if assistant_content.strip():
                assistant_msg = ContextMessage(
                    role=Role.ASSISTANT,
                    content=assistant_content
                )
                self.db.save_message(session_id, assistant_msg)
    
    async def _non_stream_request(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        session_id: str,
        messages: List[ContextMessage]
    ) -> AsyncGenerator[str, None]:
        """Make non-streaming request."""
        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Add metadata
        data["_livecontext"] = {
            "session_id": session_id,
            "provider": self.provider_name,
            "timestamp": time.time()
        }
        
        yield f"data: {json.dumps(data)}\n\n"
        
        # Save assistant message
        content = self._extract_content_from_response(data)
        if content:
            assistant_msg = ContextMessage(
                role=Role.ASSISTANT,
                content=content
            )
            self.db.save_message(session_id, assistant_msg)
    
    @abstractmethod
    def _extract_content_from_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Extract content from streaming chunk."""
        pass
    
    @abstractmethod
    def _extract_content_from_response(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract content from non-streaming response."""
        pass
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class MessageConverter:
    """Utility for converting between message formats."""
    
    @staticmethod
    def openai_to_context(msg: Dict[str, Any]) -> ContextMessage:
        """Convert OpenAI message format to ContextMessage."""
        role_map = {
            "system": Role.SYSTEM,
            "user": Role.USER,
            "assistant": Role.ASSISTANT,
            "tool": Role.TOOL
        }
        
        role = role_map.get(msg.get("role", "user"), Role.USER)
        content = msg.get("content", "")
        
        # Handle content array (for vision models)
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        
        return ContextMessage(
            role=role,
            content=str(content),
            metadata={"original": msg}
        )
    
    @staticmethod
    def anthropic_to_context(msg: Dict[str, Any]) -> ContextMessage:
        """Convert Anthropic message format to ContextMessage."""
        role_map = {
            "system": Role.SYSTEM,
            "user": Role.USER,
            "assistant": Role.ASSISTANT
        }
        
        role = role_map.get(msg.get("role", "user"), Role.USER)
        
        # Handle Anthropic content format
        content = msg.get("content", "")
        if isinstance(content, list):
            # Extract text from content blocks
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        texts.append("[image]")
            content = " ".join(texts)
        
        return ContextMessage(
            role=role,
            content=str(content),
            metadata={"original": msg}
        )
    
    @staticmethod
    def ollama_to_context(msg: Dict[str, Any]) -> ContextMessage:
        """Convert Ollama message format to ContextMessage."""
        role_map = {
            "system": Role.SYSTEM,
            "user": Role.USER,
            "assistant": Role.ASSISTANT
        }
        
        role = role_map.get(msg.get("role", "user"), Role.USER)
        
        return ContextMessage(
            role=role,
            content=str(msg.get("content", "")),
            metadata={"original": msg}
        )
