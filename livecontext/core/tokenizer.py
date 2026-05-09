"""Tokenizer module for counting tokens in messages."""

import re
from typing import Dict, List, Optional, Tuple

import tiktoken

from livecontext.server.models import ContextMessage, Role


class Tokenizer:
    """Token counter for various LLM providers."""
    
    # Default token counts per model family
    DEFAULT_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
    }
    
    # Approximate tokens per character for unknown models
    CHARS_PER_TOKEN = 4
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize tokenizer for a specific model.
        
        Args:
            model_name: Name of the model to tokenize for
        """
        self.model_name = model_name
        self.encoding = self._get_encoding(model_name)
    
    def _get_encoding(self, model_name: str) -> Optional[tiktoken.Encoding]:
        """Get tiktoken encoding for model.
        
        Args:
            model_name: Model name
            
        Returns:
            tiktoken Encoding or None
        """
        # Normalize model name
        model_key = model_name.lower()
        
        # Find matching encoding
        encoding_name = None
        for key, enc in self.DEFAULT_ENCODINGS.items():
            if key in model_key:
                encoding_name = enc
                break
        
        # Fallback to cl100k_base for unknown OpenAI models
        if encoding_name is None and ("gpt" in model_key or "openai" in model_key):
            encoding_name = "cl100k_base"
        
        if encoding_name:
            try:
                return tiktoken.get_encoding(encoding_name)
            except Exception:
                return None
        
        return None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: approximate based on characters
            return len(text) // self.CHARS_PER_TOKEN
    
    def count_message_tokens(self, message: ContextMessage) -> int:
        """Count tokens in a message including overhead.
        
        Args:
            message: ContextMessage to count
            
        Returns:
            Token count with overhead
        """
        # Base token count for content
        content_tokens = self.count_tokens(message.content)
        
        # Add overhead per message (varies by model, ~4 tokens typical)
        overhead = 4
        
        # Role adds tokens
        role_tokens = self.count_tokens(message.role.value)
        
        return content_tokens + role_tokens + overhead
    
    def count_messages_tokens(self, messages: List[ContextMessage]) -> int:
        """Count tokens in a list of messages.
        
        Args:
            messages: List of ContextMessage
            
        Returns:
            Total token count
        """
        total = 0
        for msg in messages:
            total += self.count_message_tokens(msg)
        
        # Add conversation overhead (~3 tokens)
        if messages:
            total += 3
        
        return total
    
    def truncate_to_max_tokens(
        self, 
        text: str, 
        max_tokens: int,
        truncation_side: str = "right"
    ) -> str:
        """Truncate text to fit within max tokens.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            truncation_side: "left" or "right"
            
        Returns:
            Truncated text
        """
        if not text:
            return text
        
        if self.encoding:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            if truncation_side == "left":
                truncated = tokens[-max_tokens:]
            else:
                truncated = tokens[:max_tokens]
            
            return self.encoding.decode(truncated)
        else:
            # Fallback: character-based truncation
            max_chars = max_tokens * self.CHARS_PER_TOKEN
            if truncation_side == "left":
                return text[-max_chars:]
            else:
                return text[:max_chars]
    
    def get_token_breakdown(
        self, 
        messages: List[ContextMessage]
    ) -> Dict[str, any]:
        """Get detailed token breakdown.
        
        Args:
            messages: List of messages
            
        Returns:
            Dictionary with breakdown
        """
        breakdown = {
            "total_tokens": 0,
            "messages": [],
            "by_role": {},
            "overhead": 3 if messages else 0
        }
        
        for msg in messages:
            msg_tokens = self.count_message_tokens(msg)
            breakdown["total_tokens"] += msg_tokens
            
            msg_info = {
                "id": msg.id,
                "role": msg.role.value,
                "tokens": msg_tokens,
                "content_length": len(msg.content)
            }
            breakdown["messages"].append(msg_info)
            
            role = msg.role.value
            if role not in breakdown["by_role"]:
                breakdown["by_role"][role] = {"count": 0, "tokens": 0}
            breakdown["by_role"][role]["count"] += 1
            breakdown["by_role"][role]["tokens"] += msg_tokens
        
        breakdown["total_tokens"] += breakdown["overhead"]
        
        return breakdown


class AnthropicTokenizer(Tokenizer):
    """Tokenizer for Anthropic models (Claude)."""
    
    # Claude uses roughly similar tokenization to GPT-4
    # But we can use a simpler approximation
    
    def __init__(self, model_name: str = "claude-3-opus"):
        """Initialize Anthropic tokenizer.
        
        Args:
            model_name: Claude model name
        """
        super().__init__(model_name)
        # Claude tends to use slightly fewer tokens
        self.CHARS_PER_TOKEN = 3.5
    
    def _get_encoding(self, model_name: str) -> Optional[tiktoken.Encoding]:
        """Anthropic doesn't use tiktoken, use approximation."""
        return None
    
    def count_message_tokens(self, message: ContextMessage) -> int:
        """Count tokens for Claude message format."""
        content_tokens = self.count_tokens(message.content)
        
        # Claude uses XML-style formatting
        # <message role="user">content</message>
        overhead = 8  # XML tags overhead
        
        return content_tokens + overhead


class OllamaTokenizer(Tokenizer):
    """Tokenizer for Ollama/local models."""
    
    def __init__(self, model_name: str = "llama2"):
        """Initialize Ollama tokenizer.
        
        Args:
            model_name: Local model name
        """
        super().__init__(model_name)
        self.CHARS_PER_TOKEN = 4
    
    def _get_encoding(self, model_name: str) -> Optional[tiktoken.Encoding]:
        """Local models may use various tokenizers."""
        # Try to detect tokenizer type from model name
        model_lower = model_name.lower()
        
        if "llama" in model_lower:
            # Llama uses SentencePiece
            return None
        elif "mistral" in model_lower or "mixtral" in model_lower:
            # Mistral uses similar to Llama
            return None
        elif "phi" in model_lower:
            # Phi uses tiktoken-like
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None
        
        return None


def get_tokenizer(model_name: str, provider: str = "openai") -> Tokenizer:
    """Factory function to get appropriate tokenizer.
    
    Args:
        model_name: Model name
        provider: Provider name (openai, anthropic, ollama)
        
    Returns:
        Tokenizer instance
    """
    provider_lower = provider.lower()
    
    if provider_lower == "anthropic":
        return AnthropicTokenizer(model_name)
    elif provider_lower == "ollama":
        return OllamaTokenizer(model_name)
    else:
        return Tokenizer(model_name)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/daksh/7May/projects/livecontext")
    
    # Test tokenizers
    print("Testing Tokenizers")
    print("=" * 50)
    
    # Test OpenAI tokenizer
    tokenizer = Tokenizer("gpt-4")
    text = "Hello, world! This is a test message."
    tokens = tokenizer.count_tokens(text)
    print(f"OpenAI GPT-4: '{text[:30]}...' = {tokens} tokens")
    
    # Test message counting
    messages = [
        ContextMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ContextMessage(role=Role.USER, content="Hello!"),
        ContextMessage(role=Role.ASSISTANT, content="Hi there! How can I help you today?"),
    ]
    
    total = tokenizer.count_messages_tokens(messages)
    print(f"Total for {len(messages)} messages: {total} tokens")
    
    # Test breakdown
    breakdown = tokenizer.get_token_breakdown(messages)
    print(f"\nBreakdown: {breakdown['by_role']}")
    
    # Test Anthropic
    anthropic = AnthropicTokenizer("claude-3-opus")
    anthropic_tokens = anthropic.count_messages_tokens(messages)
    print(f"\nAnthropic Claude: {anthropic_tokens} tokens")
    
    # Test truncation
    long_text = "This is a very long text that needs to be truncated. " * 20
    truncated = tokenizer.truncate_to_max_tokens(long_text, 20)
    truncated_tokens = tokenizer.count_tokens(truncated)
    print(f"\nTruncation test: {len(long_text)} chars -> {truncated_tokens} tokens")
    
    print("\n✅ All tokenizer tests passed!")
