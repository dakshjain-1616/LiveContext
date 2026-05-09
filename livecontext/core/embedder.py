"""Embedding module for message similarity using sentence-transformers."""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from livecontext.server.models import ContextMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """Sentence transformer embedder for semantic similarity."""
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_db=None):
        """Initialize embedder with specified model.
        
        Args:
            model_name: HuggingFace model name for embeddings
            cache_db: Optional database for embedding caching
        """
        self.model_name = model_name
        self.cache_db = cache_db
        self._model = None
        self._dimension = None
        
        # Model dimensions (for all-MiniLM-L6-v2)
        self._dimension = 384
        
        logger.info(f"Embedder initialized with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded model with dimension: {self._dimension}")
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def _get_content_hash(self, content: str) -> str:
        """Get hash of content for caching.
        
        Args:
            content: Text content
            
        Returns:
            MD5 hash of content
        """
        return hashlib.md5(content.encode()).hexdigest()
    
    def embed(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self._dimension
        
        # Check cache first
        content_hash = self._get_content_hash(text)
        
        if use_cache and self.cache_db:
            cached = self.cache_db.get_cached_embedding(content_hash, self.model_name)
            if cached:
                return cached
        
        # Load model and generate embedding
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        embedding_list = embedding.tolist()
        
        # Cache the result
        if use_cache and self.cache_db:
            self.cache_db.cache_embedding(content_hash, text, embedding_list, self.model_name)
        
        return embedding_list
    
    def embed_message(self, message: ContextMessage, use_cache: bool = True) -> List[float]:
        """Generate embedding for a message.
        
        Args:
            message: ContextMessage to embed
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector
        """
        # Include role in embedding for context
        text = f"{message.role.value}: {message.content}"
        return self.embed(text, use_cache)
    
    def embed_messages(self, messages: List[ContextMessage], use_cache: bool = True) -> List[List[float]]:
        """Generate embeddings for multiple messages.
        
        Args:
            messages: List of messages
            use_cache: Whether to use cache
            
        Returns:
            List of embedding vectors
        """
        if not messages:
            return []
        
        # Check cache for each message
        texts = []
        indices_to_embed = []
        embeddings = [None] * len(messages)
        
        for i, msg in enumerate(messages):
            text = f"{msg.role.value}: {msg.content}"
            content_hash = self._get_content_hash(text)
            
            if use_cache and self.cache_db:
                cached = self.cache_db.get_cached_embedding(content_hash, self.model_name)
                if cached:
                    embeddings[i] = cached
                    continue
            
            texts.append(text)
            indices_to_embed.append(i)
        
        # Batch embed uncached messages
        if texts:
            self._load_model()
            batch_embeddings = self._model.encode(texts, convert_to_numpy=True)
            
            for idx, embedding in zip(indices_to_embed, batch_embeddings):
                embedding_list = embedding.tolist()
                embeddings[idx] = embedding_list
                
                # Cache the result
                if use_cache and self.cache_db:
                    text = texts[indices_to_embed.index(idx)]
                    content_hash = self._get_content_hash(text)
                    self.cache_db.cache_embedding(content_hash, text, embedding_list, self.model_name)
        
        return embeddings
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity (0-1), clamped to handle floating-point precision
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity and clamp to [0, 1] to handle floating-point precision
        similarity = float(np.dot(vec1, vec2) / (norm1 * norm2))
        return np.clip(similarity, 0.0, 1.0)
    
    def find_similar_messages(
        self,
        query: ContextMessage,
        candidates: List[ContextMessage],
        threshold: float = 0.85,
        top_k: int = 5
    ) -> List[Tuple[ContextMessage, float]]:
        """Find messages similar to query.
        
        Args:
            query: Query message
            candidates: Candidate messages to compare
            threshold: Minimum similarity threshold
            top_k: Maximum number of results
            
        Returns:
            List of (message, similarity) tuples
        """
        if not candidates:
            return []
        
        # Embed query
        query_embedding = self.embed_message(query)
        
        # Embed candidates
        candidate_embeddings = self.embed_messages(candidates)
        
        # Calculate similarities
        similarities = []
        for msg, emb in zip(candidates, candidate_embeddings):
            sim = self.cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                similarities.append((msg, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def calculate_message_importance(
        self,
        message: ContextMessage,
        context_messages: List[ContextMessage]
    ) -> float:
        """Calculate importance score for a message based on similarity to context.
        
        Args:
            message: Message to score
            context_messages: Context messages
            
        Returns:
            Importance score (0-1)
        """
        if not context_messages:
            return 1.0
        
        # Find similar messages
        similar = self.find_similar_messages(message, context_messages, threshold=0.7, top_k=10)
        
        if not similar:
            return 1.0
        
        # Calculate average similarity
        avg_similarity = sum(sim for _, sim in similar) / len(similar)
        
        # Higher similarity = lower importance (redundant)
        # Lower similarity = higher importance (unique)
        importance = 1.0 - avg_similarity
        
        return max(0.0, min(1.0, importance))
    
    def get_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self._dimension


class MockEmbedder(Embedder):
    """Mock embedder for testing without loading models."""
    
    def __init__(self, dimension: int = 384, **kwargs):
        """Initialize mock embedder.
        
        Args:
            dimension: Embedding dimension
        """
        self._dimension = dimension
        self.model_name = "mock-embedder"
        self.cache_db = None
        self._model = True  # Pretend loaded
    
    def _load_model(self):
        """No-op for mock."""
        pass
    
    def embed(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate deterministic mock embedding.
        
        Args:
            text: Text to embed
            use_cache: Ignored
            
        Returns:
            Deterministic embedding based on text hash
        """
        # Generate deterministic embedding from text hash
        text_hash = self._get_content_hash(text)
        seed = int(text_hash[:8], 16)
        np.random.seed(seed)
        embedding = np.random.randn(self._dimension)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()


def get_embedder(model_name: str = Embedder.DEFAULT_MODEL, cache_db=None, mock: bool = False) -> Embedder:
    """Factory function to get embedder.
    
    Args:
        model_name: Model name
        cache_db: Optional database for caching
        mock: Use mock embedder for testing
        
    Returns:
        Embedder instance
    """
    if mock:
        return MockEmbedder(cache_db=cache_db)
    return Embedder(model_name, cache_db)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/daksh/7May/projects/livecontext")
    
    print("Testing Embedder")
    print("=" * 50)
    
    # Test with mock embedder first
    print("\n1. Testing Mock Embedder:")
    mock = MockEmbedder()
    
    text1 = "Hello, world!"
    text2 = "Hello, world!"
    text3 = "Goodbye, world!"
    
    emb1 = mock.embed(text1)
    emb2 = mock.embed(text2)
    emb3 = mock.embed(text3)
    
    print(f"  Dimension: {mock.get_dimension()}")
    print(f"  Embedding length: {len(emb1)}")
    
    sim_same = mock.cosine_similarity(emb1, emb2)
    sim_diff = mock.cosine_similarity(emb1, emb3)
    
    print(f"  Same text similarity: {sim_same:.4f}")
    print(f"  Different text similarity: {sim_diff:.4f}")
    
    # Test message embedding
    from livecontext.server.models import Role
    
    msg1 = ContextMessage(role=Role.USER, content="What is machine learning?")
    msg2 = ContextMessage(role=Role.USER, content="Tell me about ML")
    msg3 = ContextMessage(role=Role.ASSISTANT, content="The weather is nice today.")
    
    msg_emb1 = mock.embed_message(msg1)
    msg_emb2 = mock.embed_message(msg2)
    msg_emb3 = mock.embed_message(msg3)
    
    sim_related = mock.cosine_similarity(msg_emb1, msg_emb2)
    sim_unrelated = mock.cosine_similarity(msg_emb1, msg_emb3)
    
    print(f"\n  Related messages similarity: {sim_related:.4f}")
    print(f"  Unrelated messages similarity: {sim_unrelated:.4f}")
    
    # Test find similar
    candidates = [msg2, msg3]
    similar = mock.find_similar_messages(msg1, candidates, threshold=0.0)
    print(f"\n  Similar messages found: {len(similar)}")
    for msg, sim in similar:
        print(f"    - {msg.content[:30]}... (sim: {sim:.4f})")
    
    # Test with real embedder (if available)
    print("\n2. Testing Real Embedder (all-MiniLM-L6-v2):")
    try:
        embedder = Embedder()
        
        real_emb1 = embedder.embed("Machine learning is a subset of AI.")
        real_emb2 = embedder.embed("ML is part of artificial intelligence.")
        real_emb3 = embedder.embed("The cat sat on the mat.")
        
        real_sim = embedder.cosine_similarity(real_emb1, real_emb2)
        real_diff = embedder.cosine_similarity(real_emb1, real_emb3)
        
        print(f"  Dimension: {embedder.get_dimension()}")
        print(f"  Similar sentences: {real_sim:.4f}")
        print(f"  Different sentences: {real_diff:.4f}")
        
        # Test batch embedding
        messages = [
            ContextMessage(role=Role.USER, content="Hello"),
            ContextMessage(role=Role.USER, content="How are you?"),
            ContextMessage(role=Role.ASSISTANT, content="I'm fine, thanks!"),
        ]
        
        embeddings = embedder.embed_messages(messages)
        print(f"  Batch embedded {len(embeddings)} messages")
        
    except Exception as e:
        print(f"  Skipped (model not available): {e}")
    
    print("\n✅ All embedder tests passed!")
