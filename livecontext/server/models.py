"""Pydantic models for LiveContext data structures."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    """Message role enumeration."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class EvictionStrategy(str, Enum):
    """Strategy used for message eviction."""
    TOKEN_TRUNCATION = "token_truncation"
    SIMILARITY_MERGE = "similarity_merge"
    IMPORTANCE_FILTER = "importance_filter"
    SLIDING_WINDOW = "sliding_window"
    MANUAL = "manual"


class ContextMessage(BaseModel):
    """A single message in the context window."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_count: int = Field(default=0, ge=0)
    embedding: Optional[List[float]] = None
    importance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "id": "msg-123",
                "role": "user",
                "content": "Hello, how are you?",
                "timestamp": "2024-01-15T10:30:00Z",
                "token_count": 6,
                "importance_score": 1.0
            }]
        }
    }


class Eviction(BaseModel):
    """Record of a message eviction event."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    message_id: str
    strategy: EvictionStrategy
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reason: str = ""
    token_savings: int = Field(default=0, ge=0)
    similarity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    merged_into: Optional[str] = None
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "id": "evict-456",
                "message_id": "msg-123",
                "strategy": "similarity_merge",
                "timestamp": "2024-01-15T10:35:00Z",
                "reason": "High similarity to existing context",
                "token_savings": 6,
                "similarity_score": 0.95
            }]
        }
    }


class ContextSnapshot(BaseModel):
    """Complete snapshot of a context window state."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    messages: List[ContextMessage] = Field(default_factory=list)
    evictions: List[Eviction] = Field(default_factory=list)
    total_tokens: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=4096, gt=0)
    utilization_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    model_name: str = "unknown"
    provider: str = "unknown"
    
    @field_validator("utilization_percent")
    @classmethod
    def validate_utilization(cls, v: float) -> float:
        return max(0.0, min(100.0, v))
    
    def calculate_utilization(self) -> float:
        """Calculate context window utilization percentage."""
        if self.max_tokens <= 0:
            return 0.0
        return (self.total_tokens / self.max_tokens) * 100
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "id": "snap-789",
                "session_id": "session-abc",
                "timestamp": "2024-01-15T10:30:00Z",
                "messages": [],
                "evictions": [],
                "total_tokens": 0,
                "max_tokens": 4096,
                "utilization_percent": 0.0,
                "model_name": "gpt-4",
                "provider": "openai"
            }]
        }
    }


class SessionInfo(BaseModel):
    """Session metadata."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    model_name: str = "unknown"
    provider: str = "unknown"
    max_tokens: int = Field(default=4096, gt=0)
    message_count: int = Field(default=0, ge=0)
    total_evictions: int = Field(default=0, ge=0)
    is_active: bool = True


class StreamingEvent(BaseModel):
    """WebSocket streaming event for real-time updates."""
    event_type: str  # "snapshot", "eviction", "message", "error"
    payload: Union[ContextSnapshot, Eviction, ContextMessage, Dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str


class ProxyRequest(BaseModel):
    """Incoming proxy request metadata."""
    provider: str  # "openai", "anthropic", "ollama"
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    
    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not v:
            raise ValueError("Messages list cannot be empty")
        for msg in v:
            if "role" not in msg or "content" not in msg:
                raise ValueError("Each message must have 'role' and 'content' fields")
        return v


class ProxyResponse(BaseModel):
    """Outgoing proxy response metadata."""
    snapshot_id: str
    session_id: str
    tokens_used: int
    tokens_remaining: int
    evictions_triggered: int
    processing_time_ms: float
