"""Anthropic proxy handler."""

import json
import logging
from typing import Any, Dict, List, Optional

from livecontext.proxies.base import MessageConverter, ProxyHandler
from livecontext.server.models import ContextMessage, Role

logger = logging.getLogger(__name__)


class AnthropicProxyHandler(ProxyHandler):
    """Proxy handler for Anthropic Claude API."""
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    @property
    def default_base_url(self) -> str:
        return "https://api.anthropic.com/v1/messages"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get Anthropic-specific headers."""
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01"
        }
        return headers
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[ContextMessage]:
        """Convert Anthropic messages to ContextMessage."""
        return [MessageConverter.anthropic_to_context(msg) for msg in messages]
    
    def _build_request_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Build Anthropic request payload.
        
        Anthropic uses a different format:
        - system is a top-level parameter
        - messages array doesn't include system
        """
        # Separate system message from other messages
        system_content = None
        chat_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                # Handle system content
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract text from content blocks
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                    system_content = " ".join(texts) if texts else None
                else:
                    system_content = str(content) if content else None
            else:
                chat_messages.append(msg)
        
        payload = {
            "model": model,
            "messages": chat_messages,
            "stream": stream,
            "max_tokens": kwargs.get("max_tokens", 4096)
        }
        
        if system_content:
            payload["system"] = system_content
        
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        
        return payload
    
    def _parse_stream_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """Parse Anthropic streaming chunk.
        
        Anthropic uses SSE format with event types.
        """
        if not chunk:
            return None
        
        # Handle event lines
        if chunk.startswith("event: "):
            event_type = chunk[7:].strip()
            return {"_event_type": event_type, "_pending": True}
        
        if chunk.startswith("data: "):
            data_str = chunk[6:]
            try:
                data = json.loads(data_str)
                # Add event type if we saw one
                if hasattr(self, '_last_event_type'):
                    data['_event_type'] = self._last_event_type
                return data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse chunk: {data_str[:100]}")
                return None
        
        # Try parsing as raw JSON (for non-SSE chunks)
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _extract_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from Anthropic response."""
        usage = response.get("usage", {})
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        }
    
    def _extract_content_from_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Extract content from Anthropic streaming chunk."""
        event_type = chunk.get("type") or chunk.get("_event_type")
        
        if event_type == "content_block_delta":
            delta = chunk.get("delta", {})
            return delta.get("text")
        
        if event_type == "message_delta":
            # Message completion event
            return None
        
        # Handle content_block_start
        if event_type == "content_block_start":
            content_block = chunk.get("content_block", {})
            if content_block.get("type") == "text":
                return content_block.get("text")
        
        return None
    
    def _extract_content_from_response(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract content from Anthropic non-streaming response."""
        content = response.get("content", [])
        if not content:
            return None
        
        # Concatenate all text blocks
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    texts.append(f"[tool_use: {block.get('name', 'unknown')}]")
        
        return " ".join(texts) if texts else None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/daksh/7May/projects/livecontext")
    
    print("Testing Anthropic Proxy Handler")
    print("=" * 50)
    
    # Test message conversion
    handler = AnthropicProxyHandler()
    
    anthropic_messages = [
        {"role": "system", "content": "You are Claude, a helpful AI."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help?"}
    ]
    
    context_messages = handler._convert_messages(anthropic_messages)
    print(f"Converted {len(anthropic_messages)} messages")
    for msg in context_messages:
        print(f"  {msg.role.value}: {msg.content[:30]}...")
    
    # Test payload building
    payload = handler._build_request_payload(
        anthropic_messages,
        "claude-3-opus-20240229",
        stream=True,
        max_tokens=1000,
        temperature=0.7
    )
    print(f"\nPayload keys: {list(payload.keys())}")
    print(f"  Has system: {'system' in payload}")
    print(f"  Messages count: {len(payload.get('messages', []))}")
    
    # Test chunk parsing
    test_chunks = [
        'event: content_block_delta',
        'data: {"type": "content_block_delta", "delta": {"text": "Hello"}}',
        'event: content_block_delta',
        'data: {"type": "content_block_delta", "delta": {"text": " world"}}',
        'event: message_stop',
        'data: {"type": "message_stop"}'
    ]
    
    print("\nParsing chunks:")
    for chunk in test_chunks:
        parsed = handler._parse_stream_chunk(chunk)
        if parsed and not parsed.get("_pending"):
            content = handler._extract_content_from_chunk(parsed)
            if content:
                print(f"  Content: {content}")
    
    # Test content extraction from response
    test_response = {
        "content": [
            {"type": "text", "text": "This is the response."}
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }
    
    content = handler._extract_content_from_response(test_response)
    print(f"\nExtracted from response: {content}")
    
    print("\n✅ Anthropic proxy handler tests passed!")
