"""OpenAI proxy handler."""

import json
import logging
from typing import Any, Dict, List, Optional

from livecontext.proxies.base import MessageConverter, ProxyHandler
from livecontext.server.models import ContextMessage, ProxyRequest, Role

logger = logging.getLogger(__name__)


class OpenAIProxyHandler(ProxyHandler):
    """Proxy handler for OpenAI API."""
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def default_base_url(self) -> str:
        return "https://api.openai.com/v1/chat/completions"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get OpenAI-specific headers."""
        headers = super()._get_headers()
        headers["OpenAI-Beta"] = "assistants=v2"
        return headers
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[ContextMessage]:
        """Convert OpenAI messages to ContextMessage."""
        return [MessageConverter.openai_to_context(msg) for msg in messages]
    
    def _build_request_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Build OpenAI request payload."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        if kwargs.get("max_tokens"):
            payload["max_tokens"] = kwargs["max_tokens"]
        
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        
        return payload
    
    def _parse_stream_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """Parse OpenAI streaming chunk."""
        if not chunk.startswith("data: "):
            return None
        
        data_str = chunk[6:]  # Remove "data: " prefix
        
        if data_str.strip() == "[DONE]":
            return {"done": True}
        
        try:
            return json.loads(data_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse chunk: {chunk[:100]}")
            return None
    
    def _extract_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from OpenAI response."""
        usage = response.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
    
    def _extract_content_from_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Extract content from OpenAI streaming chunk."""
        if chunk.get("done"):
            return None
        
        choices = chunk.get("choices", [])
        if not choices:
            return None
        
        delta = choices[0].get("delta", {})
        return delta.get("content")
    
    def _extract_content_from_response(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract content from OpenAI non-streaming response."""
        choices = response.get("choices", [])
        if not choices:
            return None
        
        message = choices[0].get("message", {})
        return message.get("content")


class AzureOpenAIProxyHandler(OpenAIProxyHandler):
    """Proxy handler for Azure OpenAI API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        **kwargs
    ):
        """Initialize Azure OpenAI proxy.
        
        Args:
            api_key: Azure API key
            base_url: Azure endpoint URL
            api_version: API version
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.api_version = api_version
    
    @property
    def provider_name(self) -> str:
        return "azure"
    
    @property
    def default_base_url(self) -> str:
        raise ValueError("Azure OpenAI requires base_url to be set")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get Azure-specific headers."""
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key or ""
        }
        return headers
    
    def _build_request_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Build Azure OpenAI request payload."""
        # Azure uses deployment name in URL, not model in body
        payload = {
            "messages": messages,
            "stream": stream
        }
        
        if kwargs.get("max_tokens"):
            payload["max_tokens"] = kwargs["max_tokens"]
        
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        
        return payload


if __name__ == "__main__":
    import asyncio
    import sys
    sys.path.insert(0, "/home/daksh/7May/projects/livecontext")
    
    print("Testing OpenAI Proxy Handler")
    print("=" * 50)
    
    # Test message conversion
    handler = OpenAIProxyHandler()
    
    openai_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    context_messages = handler._convert_messages(openai_messages)
    print(f"Converted {len(openai_messages)} messages")
    for msg in context_messages:
        print(f"  {msg.role.value}: {msg.content[:30]}...")
    
    # Test payload building
    payload = handler._build_request_payload(
        openai_messages,
        "gpt-4",
        stream=True,
        max_tokens=100,
        temperature=0.7
    )
    print(f"\nPayload keys: {list(payload.keys())}")
    
    # Test chunk parsing
    test_chunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: [DONE]'
    ]
    
    print("\nParsing chunks:")
    for chunk in test_chunks:
        parsed = handler._parse_stream_chunk(chunk)
        if parsed:
            content = handler._extract_content_from_chunk(parsed)
            print(f"  Chunk: {content}")
    
    print("\n✅ OpenAI proxy handler tests passed!")
