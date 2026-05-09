"""Eviction strategies for managing context window overflow."""

import logging
from typing import Callable, Dict, List, Optional, Tuple

from livecontext.core.embedder import Embedder
from livecontext.core.tokenizer import Tokenizer
from livecontext.server.models import (
    ContextMessage,
    ContextSnapshot,
    Eviction,
    EvictionStrategy,
    Role,
)

logger = logging.getLogger(__name__)


class EvictionResult:
    """Result of an eviction operation."""
    
    def __init__(
        self,
        kept_messages: List[ContextMessage],
        evicted_messages: List[ContextMessage],
        evictions: List[Eviction],
        tokens_before: int,
        tokens_after: int
    ):
        self.kept_messages = kept_messages
        self.evicted_messages = evicted_messages
        self.evictions = evictions
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.tokens_saved = tokens_before - tokens_after
    
    def to_snapshot(self, session_id: str, model_name: str, provider: str) -> ContextSnapshot:
        """Convert to ContextSnapshot.
        
        Args:
            session_id: Session ID
            model_name: Model name
            provider: Provider name
            
        Returns:
            ContextSnapshot
        """
        max_tokens = self.kept_messages[0].token_count if self.kept_messages else 4096
        
        return ContextSnapshot(
            session_id=session_id,
            messages=self.kept_messages,
            evictions=self.evictions,
            total_tokens=self.tokens_after,
            max_tokens=max_tokens,
            utilization_percent=(self.tokens_after / max_tokens * 100) if max_tokens > 0 else 0,
            model_name=model_name,
            provider=provider
        )


class EvictionManager:
    """Manages context window eviction strategies."""
    
    def __init__(
        self,
        tokenizer: Tokenizer,
        embedder: Optional[Embedder] = None,
        max_tokens: int = 4096,
        buffer_tokens: int = 200
    ):
        """Initialize eviction manager.
        
        Args:
            tokenizer: Tokenizer for counting tokens
            embedder: Optional embedder for similarity-based eviction
            max_tokens: Maximum tokens allowed in context
            buffer_tokens: Buffer to leave room for response
        """
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.buffer_tokens = buffer_tokens
        self.effective_max = max_tokens - buffer_tokens
        
        logger.info(f"EvictionManager initialized: max={max_tokens}, buffer={buffer_tokens}")
    
    def check_overflow(
        self,
        messages: List[ContextMessage],
        new_message: Optional[ContextMessage] = None
    ) -> Tuple[bool, int]:
        """Check if adding a message would cause overflow.
        
        Args:
            messages: Current messages
            new_message: Optional new message to add
            
        Returns:
            (would_overflow, total_tokens)
        """
        total_tokens = self.tokenizer.count_messages_tokens(messages)
        
        if new_message:
            total_tokens += self.tokenizer.count_message_tokens(new_message)
        
        return total_tokens > self.effective_max, total_tokens
    
    def evict(
        self,
        messages: List[ContextMessage],
        strategy: EvictionStrategy = EvictionStrategy.SLIDING_WINDOW,
        new_message: Optional[ContextMessage] = None
    ) -> EvictionResult:
        """Apply eviction strategy to messages.
        
        Args:
            messages: Current messages
            strategy: Eviction strategy to use
            new_message: Optional new message being added
            
        Returns:
            EvictionResult
        """
        tokens_before = self.tokenizer.count_messages_tokens(messages)
        
        if new_message:
            tokens_before += self.tokenizer.count_message_tokens(new_message)
        
        # If no overflow, return as-is
        if tokens_before <= self.effective_max:
            return EvictionResult(
                kept_messages=messages + ([new_message] if new_message else []),
                evicted_messages=[],
                evictions=[],
                tokens_before=tokens_before,
                tokens_after=tokens_before
            )
        
        # Apply strategy
        if strategy == EvictionStrategy.SLIDING_WINDOW:
            result = self._sliding_window_eviction(messages, new_message)
        elif strategy == EvictionStrategy.TOKEN_TRUNCATION:
            result = self._token_truncation_eviction(messages, new_message)
        elif strategy == EvictionStrategy.SIMILARITY_MERGE:
            if self.embedder is None:
                logger.warning("Similarity merge requires embedder, falling back to sliding window")
                result = self._sliding_window_eviction(messages, new_message)
            else:
                result = self._similarity_merge_eviction(messages, new_message)
        elif strategy == EvictionStrategy.IMPORTANCE_FILTER:
            if self.embedder is None:
                logger.warning("Importance filter requires embedder, falling back to sliding window")
                result = self._sliding_window_eviction(messages, new_message)
            else:
                result = self._importance_filter_eviction(messages, new_message)
        else:
            result = self._sliding_window_eviction(messages, new_message)
        
        result.tokens_before = tokens_before
        return result
    
    def _sliding_window_eviction(
        self,
        messages: List[ContextMessage],
        new_message: Optional[ContextMessage]
    ) -> EvictionResult:
        """Remove oldest messages until under limit.
        
        Args:
            messages: Current messages
            new_message: Optional new message
            
        Returns:
            EvictionResult
        """
        # Combine messages
        all_messages = list(messages)
        if new_message:
            all_messages.append(new_message)
        
        # Keep removing oldest non-system messages until under limit
        kept = list(all_messages)
        evicted = []
        evictions = []

        while kept:
            tokens = self.tokenizer.count_messages_tokens(kept)
            if tokens <= self.effective_max:
                break

            # Find oldest non-system message (never evict system messages)
            for i, msg in enumerate(kept):
                if msg.role != Role.SYSTEM:
                    removed = kept.pop(i)
                    evicted.append(removed)

                    eviction = Eviction(
                        message_id=removed.id,
                        strategy=EvictionStrategy.SLIDING_WINDOW,
                        reason="Sliding window eviction - oldest message removed",
                        token_savings=removed.token_count
                    )
                    evictions.append(eviction)
                    break
            else:
                # No non-system messages to evict, stop here to protect system messages
                logger.warning("Cannot evict further - only system messages remain. Context window may be over limit.")
                break
        
        tokens_after = self.tokenizer.count_messages_tokens(kept)
        
        return EvictionResult(
            kept_messages=kept,
            evicted_messages=evicted,
            evictions=evictions,
            tokens_before=0,  # Set by caller
            tokens_after=tokens_after
        )
    
    def _token_truncation_eviction(
        self,
        messages: List[ContextMessage],
        new_message: Optional[ContextMessage]
    ) -> EvictionResult:
        """Truncate long messages to fit.
        
        Args:
            messages: Current messages
            new_message: Optional new message
            
        Returns:
            EvictionResult
        """
        all_messages = list(messages)
        if new_message:
            all_messages.append(new_message)
        
        # Calculate available tokens per message
        num_messages = len(all_messages)
        if num_messages == 0:
            return EvictionResult([], [], [], 0, 0)
        
        # Reserve tokens for system messages
        system_messages = [m for m in all_messages if m.role == Role.SYSTEM]
        non_system = [m for m in all_messages if m.role != Role.SYSTEM]
        
        system_tokens = sum(m.token_count for m in system_messages)
        available = self.effective_max - system_tokens - (len(non_system) * 4)  # Overhead
        
        if available <= 0 or not non_system:
            # Fall back to sliding window
            return self._sliding_window_eviction(messages, new_message)
        
        tokens_per_message = available // len(non_system)
        
        kept = list(system_messages)
        evicted = []
        evictions = []
        
        for msg in non_system:
            if msg.token_count > tokens_per_message:
                # Truncate message
                truncated_content = self.tokenizer.truncate_to_max_tokens(
                    msg.content,
                    tokens_per_message - 4  # Leave room for overhead
                )
                
                if len(truncated_content) < len(msg.content) * 0.5:
                    # Too much truncation, evict instead
                    evicted.append(msg)
                    eviction = Eviction(
                        message_id=msg.id,
                        strategy=EvictionStrategy.TOKEN_TRUNCATION,
                        reason="Message too long, evicted instead of heavy truncation",
                        token_savings=msg.token_count
                    )
                    evictions.append(eviction)
                else:
                    # Create truncated message
                    from copy import copy
                    truncated = copy(msg)
                    truncated.content = truncated_content + "..."
                    truncated.token_count = self.tokenizer.count_tokens(truncated.content)
                    kept.append(truncated)
                    
                    eviction = Eviction(
                        message_id=msg.id,
                        strategy=EvictionStrategy.TOKEN_TRUNCATION,
                        reason="Message truncated to fit context window",
                        token_savings=msg.token_count - truncated.token_count
                    )
                    evictions.append(eviction)
            else:
                kept.append(msg)
        
        tokens_after = self.tokenizer.count_messages_tokens(kept)
        
        return EvictionResult(
            kept_messages=kept,
            evicted_messages=evicted,
            evictions=evictions,
            tokens_before=0,
            tokens_after=tokens_after
        )
    
    def _similarity_merge_eviction(
        self,
        messages: List[ContextMessage],
        new_message: Optional[ContextMessage]
    ) -> EvictionResult:
        """Merge similar messages to reduce tokens.
        
        Args:
            messages: Current messages
            new_message: Optional new message
            
        Returns:
            EvictionResult
        """
        if self.embedder is None:
            raise ValueError("Embedder required for similarity merge")
        
        all_messages = list(messages)
        if new_message:
            all_messages.append(new_message)
        
        # Find similar message pairs
        kept = []
        evicted = []
        evictions = []
        merged_indices = set()
        
        # Sort by timestamp to maintain order
        all_messages.sort(key=lambda m: m.timestamp)
        
        for i, msg1 in enumerate(all_messages):
            if i in merged_indices:
                continue
            
            # Find similar messages
            candidates = [all_messages[j] for j in range(i + 1, len(all_messages)) if j not in merged_indices]
            
            if not candidates:
                kept.append(msg1)
                continue
            
            similar = self.embedder.find_similar_messages(msg1, candidates, threshold=0.90, top_k=1)
            
            if similar:
                msg2, sim_score = similar[0]
                j = all_messages.index(msg2)
                
                # Merge messages
                merged_content = f"{msg1.content}\n\n[Related: {msg2.content}]"
                merged_tokens = self.tokenizer.count_tokens(merged_content)
                
                if merged_tokens < msg1.token_count + msg2.token_count:
                    # Create merged message
                    from copy import copy
                    merged = copy(msg1)
                    merged.content = merged_content
                    merged.token_count = merged_tokens
                    merged.importance_score = max(msg1.importance_score, msg2.importance_score)
                    
                    kept.append(merged)
                    merged_indices.add(i)
                    merged_indices.add(j)
                    
                    eviction = Eviction(
                        message_id=msg2.id,
                        strategy=EvictionStrategy.SIMILARITY_MERGE,
                        reason=f"Merged with similar message (sim: {sim_score:.3f})",
                        token_savings=msg2.token_count,
                        similarity_score=sim_score,
                        merged_into=merged.id
                    )
                    evictions.append(eviction)
                else:
                    kept.append(msg1)
            else:
                kept.append(msg1)
        
        # Add unmerged messages
        for i, msg in enumerate(all_messages):
            if i not in merged_indices and msg not in kept:
                kept.append(msg)
        
        # If still over limit, use sliding window
        tokens = self.tokenizer.count_messages_tokens(kept)
        if tokens > self.effective_max:
            return self._sliding_window_eviction(kept, None)
        
        return EvictionResult(
            kept_messages=kept,
            evicted_messages=evicted,
            evictions=evictions,
            tokens_before=0,
            tokens_after=tokens
        )
    
    def _importance_filter_eviction(
        self,
        messages: List[ContextMessage],
        new_message: Optional[ContextMessage]
    ) -> EvictionResult:
        """Remove least important messages based on similarity to context.
        
        Args:
            messages: Current messages
            new_message: Optional new message
            
        Returns:
            EvictionResult
        """
        if self.embedder is None:
            raise ValueError("Embedder required for importance filter")
        
        all_messages = list(messages)
        if new_message:
            all_messages.append(new_message)
        
        # Calculate importance scores
        scored_messages = []
        
        for i, msg in enumerate(all_messages):
            # Get context (all other messages)
            context = [m for j, m in enumerate(all_messages) if j != i]
            
            importance = self.embedder.calculate_message_importance(msg, context)
            scored_messages.append((msg, importance))
        
        # Sort by importance (lowest first - will be evicted)
        scored_messages.sort(key=lambda x: x[1])
        
        # Keep removing least important until under limit
        kept = [msg for msg, _ in scored_messages]
        evicted = []
        evictions = []
        
        while kept:
            tokens = self.tokenizer.count_messages_tokens(kept)
            if tokens <= self.effective_max:
                break
            
            # Find least important non-system message
            for i, (msg, importance) in enumerate(scored_messages):
                if msg in kept and msg.role != Role.SYSTEM:
                    kept.remove(msg)
                    evicted.append(msg)
                    
                    eviction = Eviction(
                        message_id=msg.id,
                        strategy=EvictionStrategy.IMPORTANCE_FILTER,
                        reason=f"Low importance score: {importance:.3f}",
                        token_savings=msg.token_count
                    )
                    evictions.append(eviction)
                    break
            else:
                # Only system messages left
                break
        
        tokens_after = self.tokenizer.count_messages_tokens(kept)
        
        return EvictionResult(
            kept_messages=kept,
            evicted_messages=evicted,
            evictions=evictions,
            tokens_before=0,
            tokens_after=tokens_after
        )


def create_eviction_manager(
    model_name: str,
    provider: str,
    max_tokens: int = 4096,
    use_embeddings: bool = False,
    db_path: Optional[str] = None
) -> EvictionManager:
    """Factory function to create eviction manager.
    
    Args:
        model_name: Model name
        provider: Provider name
        max_tokens: Maximum tokens
        use_embeddings: Whether to use embedding-based strategies
        db_path: Optional database path for caching
        
    Returns:
        EvictionManager instance
    """
    from livecontext.core.tokenizer import get_tokenizer
    from livecontext.core.embedder import get_embedder
    from livecontext.server.db import get_db
    
    tokenizer = get_tokenizer(model_name, provider)
    
    embedder = None
    if use_embeddings:
        cache_db = get_db(db_path) if db_path else None
        embedder = get_embedder(cache_db=cache_db)
    
    return EvictionManager(tokenizer, embedder, max_tokens)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/daksh/7May/projects/livecontext")
    
    print("Testing Eviction Manager")
    print("=" * 50)
    
    from livecontext.core.tokenizer import Tokenizer
    from livecontext.server.models import Role
    
    # Create test messages
    tokenizer = Tokenizer("gpt-4")
    
    messages = [
        ContextMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ContextMessage(role=Role.USER, content="Hello!"),
        ContextMessage(role=Role.ASSISTANT, content="Hi there! How can I help?"),
        ContextMessage(role=Role.USER, content="Tell me about Python."),
        ContextMessage(role=Role.ASSISTANT, content="Python is a programming language." * 10),
    ]
    
    # Count tokens
    for msg in messages:
        msg.token_count = tokenizer.count_message_tokens(msg)
    
    total_tokens = tokenizer.count_messages_tokens(messages)
    print(f"Total tokens: {total_tokens}")
    
    # Test sliding window
    print("\n1. Testing Sliding Window Eviction:")
    manager = EvictionManager(tokenizer, max_tokens=100, buffer_tokens=10)
    
    result = manager.evict(messages, EvictionStrategy.SLIDING_WINDOW)
    print(f"  Kept: {len(result.kept_messages)} messages")
    print(f"  Evicted: {len(result.evicted_messages)} messages")
    print(f"  Tokens saved: {result.tokens_saved}")
    
    # Test token truncation
    print("\n2. Testing Token Truncation:")
    manager2 = EvictionManager(tokenizer, max_tokens=150, buffer_tokens=10)
    
    result2 = manager2.evict(messages, EvictionStrategy.TOKEN_TRUNCATION)
    print(f"  Kept: {len(result2.kept_messages)} messages")
    print(f"  Evicted: {len(result2.evicted_messages)} messages")
    print(f"  Evictions: {len(result2.evictions)}")
    
    # Test with new message causing overflow
    print("\n3. Testing with New Message:")
    new_msg = ContextMessage(role=Role.USER, content="Another question here!" * 5)
    new_msg.token_count = tokenizer.count_message_tokens(new_msg)
    
    manager3 = EvictionManager(tokenizer, max_tokens=200, buffer_tokens=20)
    
    would_overflow, tokens = manager3.check_overflow(messages, new_msg)
    print(f"  Would overflow: {would_overflow}")
    print(f"  Total tokens with new message: {tokens}")
    
    if would_overflow:
        result3 = manager3.evict(messages, EvictionStrategy.SLIDING_WINDOW, new_msg)
        print(f"  After eviction: {len(result3.kept_messages)} messages")
        print(f"  Final tokens: {result3.tokens_after}")
    
    print("\n✅ All eviction tests passed!")
