"""SQLite database layer for persistence and embedding caching."""

import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Handle imports when running directly vs as module
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from livecontext.server.models import (
    ContextMessage,
    ContextSnapshot,
    Eviction,
    EvictionStrategy,
    Role,
    SessionInfo,
)


class Database:
    """SQLite database manager for LiveContext."""
    
    def __init__(self, db_path: str = "data/livecontext.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT NOT NULL DEFAULT 'unknown',
                    provider TEXT NOT NULL DEFAULT 'unknown',
                    max_tokens INTEGER NOT NULL DEFAULT 4096,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    total_evictions INTEGER NOT NULL DEFAULT 0,
                    is_active BOOLEAN NOT NULL DEFAULT 1
                )
            """)
            
            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    embedding BLOB,
                    importance_score REAL NOT NULL DEFAULT 1.0,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            
            # Snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    max_tokens INTEGER NOT NULL DEFAULT 4096,
                    utilization_percent REAL NOT NULL DEFAULT 0.0,
                    model_name TEXT NOT NULL DEFAULT 'unknown',
                    provider TEXT NOT NULL DEFAULT 'unknown',
                    message_ids TEXT,
                    eviction_ids TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            
            # Evictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evictions (
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT,
                    token_savings INTEGER NOT NULL DEFAULT 0,
                    similarity_score REAL,
                    merged_into TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)
            
            # Embeddings cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_cache (
                    content_hash TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER NOT NULL DEFAULT 1,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session 
                ON messages(session_id, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_session 
                ON snapshots(session_id, timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_evictions_session 
                ON evictions(session_id, timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_model 
                ON embeddings_cache(model_name, content_hash)
            """)
            
            conn.commit()
    
    # Session operations
    def create_session(self, session: SessionInfo) -> str:
        """Create a new session.
        
        Args:
            session: SessionInfo object
            
        Returns:
            Session ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (id, created_at, updated_at, model_name, provider, 
                                    max_tokens, message_count, total_evictions, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.model_name,
                session.provider,
                session.max_tokens,
                session.message_count,
                session.total_evictions,
                session.is_active
            ))
            return session.id
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            SessionInfo or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return SessionInfo(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                model_name=row["model_name"],
                provider=row["provider"],
                max_tokens=row["max_tokens"],
                message_count=row["message_count"],
                total_evictions=row["total_evictions"],
                is_active=bool(row["is_active"])
            )
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """Update session fields.
        
        Args:
            session_id: Session ID
            **kwargs: Fields to update
            
        Returns:
            True if updated
        """
        allowed_fields = {"model_name", "provider", "max_tokens", "message_count", 
                         "total_evictions", "is_active", "updated_at"}
        
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return False
        
        if "updated_at" not in updates:
            updates["updated_at"] = datetime.utcnow().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values()) + [session_id]
            
            cursor.execute(f"""
                UPDATE sessions 
                SET {set_clause}
                WHERE id = ?
            """, values)
            
            return cursor.rowcount > 0
    
    def list_sessions(self, active_only: bool = False) -> List[SessionInfo]:
        """List all sessions.
        
        Args:
            active_only: Only return active sessions
            
        Returns:
            List of SessionInfo
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if active_only:
                cursor.execute("SELECT * FROM sessions WHERE is_active = 1 ORDER BY updated_at DESC")
            else:
                cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
            
            rows = cursor.fetchall()
            
            return [
                SessionInfo(
                    id=row["id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    model_name=row["model_name"],
                    provider=row["provider"],
                    max_tokens=row["max_tokens"],
                    message_count=row["message_count"],
                    total_evictions=row["total_evictions"],
                    is_active=bool(row["is_active"])
                )
                for row in rows
            ]
    
    # Message operations
    def save_message(self, session_id: str, message: ContextMessage) -> str:
        """Save a message.
        
        Args:
            session_id: Session ID
            message: ContextMessage object
            
        Returns:
            Message ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            embedding_bytes = None
            if message.embedding:
                embedding_bytes = json.dumps(message.embedding).encode()
            
            cursor.execute("""
                INSERT INTO messages (id, session_id, role, content, timestamp, 
                                    token_count, embedding, importance_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message.id,
                session_id,
                message.role.value,
                message.content,
                message.timestamp.isoformat(),
                message.token_count,
                embedding_bytes,
                message.importance_score,
                json.dumps(message.metadata) if message.metadata else None
            ))
            
            return message.id
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[ContextMessage]:
        """Get messages for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of messages
            
        Returns:
            List of ContextMessage
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp"
            params = [session_id]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                embedding = None
                if row["embedding"]:
                    embedding = json.loads(row["embedding"].decode())
                
                messages.append(ContextMessage(
                    id=row["id"],
                    role=Role(row["role"]),
                    content=row["content"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    token_count=row["token_count"],
                    embedding=embedding,
                    importance_score=row["importance_score"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                ))
            
            return messages
    
    # Snapshot operations
    def save_snapshot(self, snapshot: ContextSnapshot) -> str:
        """Save a snapshot.
        
        Args:
            snapshot: ContextSnapshot object
            
        Returns:
            Snapshot ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            message_ids = json.dumps([m.id for m in snapshot.messages])
            eviction_ids = json.dumps([e.id for e in snapshot.evictions])
            
            cursor.execute("""
                INSERT INTO snapshots (id, session_id, timestamp, total_tokens, max_tokens,
                                     utilization_percent, model_name, provider, message_ids, eviction_ids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot.id,
                snapshot.session_id,
                snapshot.timestamp.isoformat(),
                snapshot.total_tokens,
                snapshot.max_tokens,
                snapshot.utilization_percent,
                snapshot.model_name,
                snapshot.provider,
                message_ids,
                eviction_ids
            ))
            
            return snapshot.id
    
    def get_snapshot(self, snapshot_id: str) -> Optional[ContextSnapshot]:
        """Get snapshot by ID.
        
        Args:
            snapshot_id: Snapshot ID
            
        Returns:
            ContextSnapshot or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Load messages and evictions
            message_ids = json.loads(row["message_ids"]) if row["message_ids"] else []
            eviction_ids = json.loads(row["eviction_ids"]) if row["eviction_ids"] else []
            
            messages = []
            for msg_id in message_ids:
                cursor.execute("SELECT * FROM messages WHERE id = ?", (msg_id,))
                msg_row = cursor.fetchone()
                if msg_row:
                    embedding = None
                    if msg_row["embedding"]:
                        embedding = json.loads(msg_row["embedding"].decode())
                    messages.append(ContextMessage(
                        id=msg_row["id"],
                        role=Role(msg_row["role"]),
                        content=msg_row["content"],
                        timestamp=datetime.fromisoformat(msg_row["timestamp"]),
                        token_count=msg_row["token_count"],
                        embedding=embedding,
                        importance_score=msg_row["importance_score"],
                        metadata=json.loads(msg_row["metadata"]) if msg_row["metadata"] else {}
                    ))
            
            evictions = []
            for evict_id in eviction_ids:
                cursor.execute("SELECT * FROM evictions WHERE id = ?", (evict_id,))
                ev_row = cursor.fetchone()
                if ev_row:
                    evictions.append(Eviction(
                        id=ev_row["id"],
                        message_id=ev_row["message_id"],
                        strategy=EvictionStrategy(ev_row["strategy"]),
                        timestamp=datetime.fromisoformat(ev_row["timestamp"]),
                        reason=ev_row["reason"] or "",
                        token_savings=ev_row["token_savings"],
                        similarity_score=ev_row["similarity_score"],
                        merged_into=ev_row["merged_into"]
                    ))
            
            return ContextSnapshot(
                id=row["id"],
                session_id=row["session_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                messages=messages,
                evictions=evictions,
                total_tokens=row["total_tokens"],
                max_tokens=row["max_tokens"],
                utilization_percent=row["utilization_percent"],
                model_name=row["model_name"],
                provider=row["provider"]
            )
    
    def get_snapshots(self, session_id: str, limit: int = 100) -> List[ContextSnapshot]:
        """Get snapshots for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of snapshots
            
        Returns:
            List of ContextSnapshot
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM snapshots 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (session_id, limit))
            
            rows = cursor.fetchall()
            snapshots = []
            
            for row in rows:
                snapshot = self.get_snapshot(row["id"])
                if snapshot:
                    snapshots.append(snapshot)
            
            return snapshots
    
    # Eviction operations
    def save_eviction(self, session_id: str, eviction: Eviction) -> str:
        """Save an eviction record.
        
        Args:
            session_id: Session ID
            eviction: Eviction object
            
        Returns:
            Eviction ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evictions (id, message_id, session_id, strategy, timestamp,
                                     reason, token_savings, similarity_score, merged_into)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eviction.id,
                eviction.message_id,
                session_id,
                eviction.strategy.value,
                eviction.timestamp.isoformat(),
                eviction.reason,
                eviction.token_savings,
                eviction.similarity_score,
                eviction.merged_into
            ))
            
            return eviction.id
    
    # Embedding cache operations
    def get_cached_embedding(self, content_hash: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding.
        
        Args:
            content_hash: Hash of content
            model_name: Model used for embedding
            
        Returns:
            Embedding vector or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT embedding FROM embeddings_cache 
                WHERE content_hash = ? AND model_name = ?
            """, (content_hash, model_name))
            
            row = cursor.fetchone()
            if row:
                # Update access stats
                cursor.execute("""
                    UPDATE embeddings_cache 
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE content_hash = ? AND model_name = ?
                """, (datetime.utcnow().isoformat(), content_hash, model_name))
                
                return json.loads(row["embedding"].decode())
            
            return None
    
    def cache_embedding(self, content_hash: str, content: str, embedding: List[float], 
                       model_name: str) -> None:
        """Cache an embedding.
        
        Args:
            content_hash: Hash of content
            content: Original content
            embedding: Embedding vector
            model_name: Model used for embedding
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings_cache 
                (content_hash, content, embedding, model_name, created_at, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, 1, ?)
            """, (
                content_hash,
                content,
                json.dumps(embedding).encode(),
                model_name,
                datetime.utcnow().isoformat(),
                datetime.utcnow().isoformat()
            ))
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """Clear embedding cache.
        
        Args:
            older_than_days: Only clear entries older than N days
            
        Returns:
            Number of entries cleared
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if older_than_days:
                from datetime import timedelta
                cutoff = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()
                cursor.execute("DELETE FROM embeddings_cache WHERE last_accessed < ?", (cutoff,))
            else:
                cursor.execute("DELETE FROM embeddings_cache")
            
            return cursor.rowcount
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with stats
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            for table in ["sessions", "messages", "snapshots", "evictions", "embeddings_cache"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(access_count) FROM embeddings_cache")
            result = cursor.fetchone()[0]
            stats["embedding_cache_hits"] = result or 0
            
            return stats


# Global database instance
_db_instance: Optional[Database] = None


def get_db(db_path: str = "data/livecontext.db") -> Database:
    """Get or create database instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        Database instance
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(db_path)
    return _db_instance


def reset_db() -> None:
    """Reset global database instance."""
    global _db_instance
    _db_instance = None


if __name__ == "__main__":
    # Test database
    db = Database("data/test.db")
    print("✅ Database initialized successfully")
    print(f"Database path: {db.db_path}")
    
    stats = db.get_stats()
    print(f"Stats: {stats}")
