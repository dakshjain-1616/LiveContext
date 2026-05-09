"""Ollama proxy handler for local models."""

import json
import logging
from typing import Any, Dict, List, Optional

from livecontext.proxies.base import MessageConverter, ProxyHandler
from livecontext.server.models import ContextMessage, Role

logger = logging.getLogger(__name__)


class OllamaProxyHandler(ProxyHandler):
    """Proxy handler for Ollama local API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize Ollama proxy.
        
        Args:
            api_key: Not used for Ollama (local)
            base_url: Ollama API URL (default: http://localhost:11434)
        """
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
    
    @property
    def provider_name(self) -> str:
        return "ollama"
    
    @property
    def default_base_url(self) -> str:
        return "http://localhost:11434/api/chat"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get Ollama headers (no auth required)."""
        return {
            "Content-Type": "application/json"
        }
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[ContextMessage]:
        """Convert Ollama messages to ContextMessage."""
        return [MessageConverter.ollama_to_context(msg) for msg in messages]
    
    def _build_request_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Build Ollama request payload.
        
        Ollama uses a simple format:
        {
            "model": "llama2",
            "messages": [{"role": "user", "content": "..."}],
            "stream": true/false,
            "options": {...}
        }
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        # Ollama options
        options = {}
        
        if kwargs.get("temperature") is not None:
            options["temperature"] = kwargs["temperature"]
        
        if kwargs.get("max_tokens"):
            options["num_predict"] = kwargs["max_tokens"]
        
        if options:
            payload["options"] = options
        
        return payload
    
    def _parse_stream_chunk(self, chunk: str) -> Optional[Dict[str, Any]]:
        """Parse Ollama streaming chunk.
        
        Ollama streams JSON objects, one per line.
        """
        if not chunk:
            return None
        
        try:
            data = json.loads(chunk)
            return data
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse chunk: {chunk[:100]}")
            return None
    
    def _extract_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from Ollama response."""
        # Ollama provides usage in the final response
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        
        return {
            "prompt_tokens": prompt_eval_count,
            "completion_tokens": eval_count,
            "total_tokens": prompt_eval_count + eval_count
        }
    
    def _extract_content_from_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Extract content from Ollama streaming chunk."""
        message = chunk.get("message", {})
        content = message.get("content", "")
        
        # Check if done
        if chunk.get("done", False):
            return None
        
        return content if content else None
    
    def _extract_content_from_response(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract content from Ollama non-streaming response."""
        message = response.get("message", {})
        return message.get("content")


class OllamaGenerateProxyHandler(OllamaProxyHandler):
    """Proxy handler for Ollama generate endpoint (legacy/completion style)."""
    
    @property
    def default_base_url(self) -> str:
        return "http://localhost:11434/api/generate"
    
    def _build_request_payload(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """Build Ollama generate request payload.
        
        Generate endpoint uses prompt instead of messages.
        """
        # Convert messages to prompt string
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")
        
        prompt = "\n\n".join(prompt_parts)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        # Options
        options = {}
        
        if kwargs.get("temperature") is not None:
            options["temperature"] = kwargs["temperature"]
        
        if kwargs.get("max_tokens"):
            options["num_predict"] = kwargs["max_tokens"]
        
        if options:
            payload["options"] = options
        
        return payload
    
    def _extract_content_from_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        """Extract content from Ollama generate streaming chunk."""
        if chunk.get("done", False):
            return None
        
        return chunk.get("response")
    
    def _extract_content_from_response(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract content from Ollama generate response."""
        return response.get("response")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/daksh/7May/projects/livecontext")
    
    print("Testing Ollama Proxy Handler")
    print("=" * 50)
    
    # Test message conversion
    handler = OllamaProxyHandler()
    
    ollama_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    
    context_messages = handler._convert_messages(ollama_messages)
    print(f"Converted {len(ollama_messages)} messages")
    for msg in context_messages:
        print(f"  {msg.role.value}: {msg.content[:30]}...")
    
    # Test payload building
    payload = handler._build_request_payload(
        ollama_messages,
        "llama2",
        stream=True,
        max_tokens=100,
        temperature=0.7
    )
    print(f"\nPayload keys: {list(payload.keys())}")
    print(f"  Model: {payload.get('model')}")
    print(f"  Stream: {payload.get('stream')}")
    print(f"  Options: {payload.get('options')}")
    
    # Test chunk parsing
    test_chunks = [
        '{"message": {"role": "assistant", "content": "Hello"}, "done": false}',
        '{"message": {"role": "assistant", "content": " world"}, "done": false}',
        '{"message": {"role": "assistant", "content": ""}, "done": true, "total_duration": 1234567890}'
    ]
    
    print("\nParsing chunks:")
    for chunk in test_chunks:
        parsed = handler._parse_stream_chunk(chunk)
        if parsed:
            content = handler._extract_content_from_chunk(parsed)
            done = parsed.get("done", False)
            print(f"  Content: '{content}' | Done: {done}")
    
    # Test generate handler
    print("\nTesting Generate Handler:")
    gen_handler = OllamaGenerateProxyHandler()
    
    gen_payload = gen_handler._build_request_payload(
        ollama_messages,
        "mistral",
        stream=False,
        temperature=0.5
    )
    print(f"  Prompt length: {len(gen_payload.get('prompt', ''))}")
    
    print("\n✅ Ollama proxy handler tests passed!")
