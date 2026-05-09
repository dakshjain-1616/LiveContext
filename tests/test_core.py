"""Tests for LiveContext core modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from livecontext.core.tokenizer import Tokenizer, get_tokenizer
from livecontext.core.embedder import Embedder, get_embedder
from livecontext.core.eviction import (
    EvictionManager,
    EvictionResult,
    create_eviction_manager,
)
from livecontext.server.models import ContextMessage, Role, EvictionStrategy


class TestTokenizer:
    """Test tokenizer functionality."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer can be initialized."""
        tokenizer = Tokenizer()
        assert tokenizer is not None
    
    def test_count_tokens_simple(self):
        """Test token counting for simple text."""
        tokenizer = Tokenizer()
        text = "Hello world"
        count = tokenizer.count_tokens(text)
        assert isinstance(count, int)
        assert count > 0
    
    def test_count_tokens_empty(self):
        """Test token counting for empty string."""
        tokenizer = Tokenizer()
        count = tokenizer.count_tokens("")
        assert count == 0
    
    def test_count_tokens_long(self):
        """Test token counting for longer text."""
        tokenizer = Tokenizer()
        text = "This is a longer piece of text that should have more tokens. " * 10
        count = tokenizer.count_tokens(text)
        assert count > 50  # Should have many tokens
    
    def test_get_tokenizer_factory(self):
        """Test get_tokenizer factory function."""
        tokenizer = get_tokenizer("gpt-4", "openai")
        assert isinstance(tokenizer, Tokenizer)
    
    def test_count_messages_tokens(self):
        """Test counting tokens for multiple messages."""
        tokenizer = Tokenizer()
        messages = [
            ContextMessage(role=Role.USER, content="Hello", token_count=1),
            ContextMessage(role=Role.ASSISTANT, content="Hi there!", token_count=2),
        ]
        total = tokenizer.count_messages_tokens(messages)
        # Should sum the token_count fields
        assert total >= 3


class TestEmbedder:
    """Test embedder functionality."""
    
    @pytest.fixture
    def embedder(self):
        """Create embedder fixture."""
        return get_embedder(mock=True)  # Use mock for faster tests
    
    def test_embedder_initialization(self, embedder):
        """Test embedder can be initialized."""
        assert embedder is not None
    
    def test_embed_single(self, embedder):
        """Test embedding a single text."""
        text = "This is a test sentence."
        embedding = embedder.embed(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
    
    def test_embed_batch(self, embedder):
        """Test embedding multiple texts."""
        from livecontext.core.embedder import MockEmbedder
        embedder = MockEmbedder()
        
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
        ]
        embeddings = [embedder.embed(t) for t in texts]
        
        assert len(embeddings) == 3
        assert all(isinstance(e, list) for e in embeddings)
    
    def test_cosine_similarity(self, embedder):
        """Test cosine similarity calculation."""
        emb1 = embedder.embed("Test sentence")
        emb2 = embedder.embed("Test sentence")
        
        similarity = embedder.cosine_similarity(emb1, emb2)
        assert 0 <= similarity <= 1
        assert similarity > 0.99  # Same text should be very similar
    
    def test_similarity_different_texts(self, embedder):
        """Test similarity of different texts."""
        emb1 = embedder.embed("The weather is nice today.")
        emb2 = embedder.embed("Machine learning is fascinating.")
        
        similarity = embedder.cosine_similarity(emb1, emb2)
        assert 0 <= similarity <= 1


class TestEvictionManager:
    """Test eviction manager functionality."""
    
    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            ContextMessage(
                id="msg1",
                role=Role.SYSTEM,
                content="System prompt",
                token_count=10,
                timestamp="2024-01-01T10:00:00",
                importance_score=1.0
            ),
            ContextMessage(
                id="msg2",
                role=Role.USER,
                content="User message 1",
                token_count=5,
                timestamp="2024-01-01T10:01:00",
                importance_score=0.9
            ),
            ContextMessage(
                id="msg3",
                role=Role.ASSISTANT,
                content="Assistant response 1",
                token_count=20,
                timestamp="2024-01-01T10:02:00",
                importance_score=0.8
            ),
            ContextMessage(
                id="msg4",
                role=Role.USER,
                content="User message 2",
                token_count=5,
                timestamp="2024-01-01T10:03:00",
                importance_score=0.7
            ),
        ]
    
    def test_eviction_manager_initialization(self):
        """Test eviction manager can be initialized."""
        manager = create_eviction_manager("gpt-4", "openai", max_tokens=100)
        assert manager is not None
        assert isinstance(manager, EvictionManager)
    
    def test_eviction_manager_evict(self, sample_messages):
        """Test eviction manager can process messages."""
        manager = create_eviction_manager("gpt-4", "openai", max_tokens=100)
        
        result = manager.evict(sample_messages, strategy=EvictionStrategy.SLIDING_WINDOW)
        assert result is not None
        assert isinstance(result, EvictionResult)
    
    def test_eviction_result_structure(self, sample_messages):
        """Test eviction result has expected structure."""
        manager = create_eviction_manager("gpt-4", "openai", max_tokens=100)
        
        result = manager.evict(sample_messages, strategy=EvictionStrategy.SLIDING_WINDOW)
        
        assert hasattr(result, 'kept_messages')
        assert hasattr(result, 'evicted_messages')
        assert hasattr(result, 'evictions')
        assert hasattr(result, 'tokens_before')
        assert hasattr(result, 'tokens_after')
        assert hasattr(result, 'tokens_saved')
    
    def test_check_overflow(self, sample_messages):
        """Test overflow detection."""
        manager = create_eviction_manager("gpt-4", "openai", max_tokens=50)
        
        # Check with current messages
        would_overflow, total_tokens = manager.check_overflow(sample_messages)
        assert isinstance(total_tokens, int)
        assert isinstance(would_overflow, bool)
    
    def test_system_message_protected(self, sample_messages):
        """Test that system messages are protected from eviction."""
        manager = create_eviction_manager("gpt-4", "openai", max_tokens=30)
        
        result = manager.evict(sample_messages, strategy=EvictionStrategy.SLIDING_WINDOW)
        
        # System message should be in kept messages
        system_kept = any(m.role == Role.SYSTEM for m in result.kept_messages)
        assert system_kept


class TestEvictionResult:
    """Test eviction result functionality."""
    
    def test_eviction_result_creation(self):
        """Test eviction result can be created."""
        result = EvictionResult(
            kept_messages=[],
            evicted_messages=[],
            evictions=[],
            tokens_before=100,
            tokens_after=50
        )
        
        assert result.tokens_saved == 50
    
    def test_eviction_result_to_snapshot(self):
        """Test converting result to snapshot."""
        from livecontext.server.models import ContextSnapshot
        
        msg = ContextMessage(
            id="test",
            role=Role.USER,
            content="Test",
            token_count=10
        )
        
        result = EvictionResult(
            kept_messages=[msg],
            evicted_messages=[],
            evictions=[],
            tokens_before=100,
            tokens_after=10
        )
        
        snapshot = result.to_snapshot("session-123", "gpt-4", "openai")
        
        assert isinstance(snapshot, ContextSnapshot)
        assert snapshot.session_id == "session-123"
        assert snapshot.total_tokens == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
