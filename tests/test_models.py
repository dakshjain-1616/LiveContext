"""Test script for Pydantic models."""

import sys
sys.path.insert(0, '/home/daksh/7May/projects/livecontext')

from datetime import datetime
from livecontext.server.models import (
    ContextMessage,
    ContextSnapshot,
    Eviction,
    EvictionStrategy,
    Role,
    SessionInfo,
    StreamingEvent,
    ProxyRequest,
    ProxyResponse
)


def test_context_message():
    """Test ContextMessage model."""
    print("Testing ContextMessage...")
    msg = ContextMessage(
        role=Role.USER,
        content="Hello, world!",
        token_count=3
    )
    assert msg.role == Role.USER
    assert msg.content == "Hello, world!"
    assert msg.token_count == 3
    assert msg.id is not None
    print(f"  ✓ Created message: {msg.id[:8]}...")
    
    # Test validation
    try:
        ContextMessage(role=Role.USER, content="")
        assert False, "Should have raised validation error"
    except ValueError as e:
        print(f"  ✓ Empty content validation: {e}")
    
    return msg


def test_eviction():
    """Test Eviction model."""
    print("Testing Eviction...")
    eviction = Eviction(
        message_id="msg-123",
        strategy=EvictionStrategy.SIMILARITY_MERGE,
        reason="High similarity to existing context",
        token_savings=50,
        similarity_score=0.95
    )
    assert eviction.strategy == EvictionStrategy.SIMILARITY_MERGE
    assert eviction.token_savings == 50
    print(f"  ✓ Created eviction: {eviction.id[:8]}...")
    return eviction


def test_context_snapshot():
    """Test ContextSnapshot model."""
    print("Testing ContextSnapshot...")
    
    msg1 = ContextMessage(role=Role.SYSTEM, content="You are a helpful assistant.", token_count=6)
    msg2 = ContextMessage(role=Role.USER, content="Hello!", token_count=1)
    
    snapshot = ContextSnapshot(
        session_id="session-abc",
        messages=[msg1, msg2],
        total_tokens=7,
        max_tokens=4096,
        model_name="gpt-4",
        provider="openai"
    )
    
    # Calculate utilization
    snapshot.utilization_percent = snapshot.calculate_utilization()
    
    assert len(snapshot.messages) == 2
    assert snapshot.total_tokens == 7
    assert snapshot.utilization_percent > 0
    print(f"  ✓ Created snapshot: {snapshot.id[:8]}...")
    print(f"  ✓ Utilization: {snapshot.utilization_percent:.2f}%")
    
    return snapshot


def test_session_info():
    """Test SessionInfo model."""
    print("Testing SessionInfo...")
    session = SessionInfo(
        model_name="claude-3-opus",
        provider="anthropic",
        max_tokens=200000,
        message_count=10
    )
    assert session.model_name == "claude-3-opus"
    assert session.max_tokens == 200000
    print(f"  ✓ Created session: {session.id[:8]}...")
    return session


def test_streaming_event():
    """Test StreamingEvent model."""
    print("Testing StreamingEvent...")
    snapshot = test_context_snapshot()
    
    event = StreamingEvent(
        event_type="snapshot",
        payload=snapshot,
        session_id="session-abc"
    )
    assert event.event_type == "snapshot"
    assert event.session_id == "session-abc"
    print(f"  ✓ Created streaming event")
    return event


def test_proxy_request():
    """Test ProxyRequest model."""
    print("Testing ProxyRequest...")
    request = ProxyRequest(
        provider="openai",
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ],
        stream=True,
        max_tokens=100
    )
    assert request.provider == "openai"
    assert len(request.messages) == 2
    print(f"  ✓ Created proxy request")
    
    # Test validation
    try:
        ProxyRequest(provider="openai", model="gpt-4", messages=[])
        assert False, "Should have raised validation error"
    except ValueError as e:
        print(f"  ✓ Empty messages validation: {e}")
    
    return request


def test_proxy_response():
    """Test ProxyResponse model."""
    print("Testing ProxyResponse...")
    response = ProxyResponse(
        snapshot_id="snap-123",
        session_id="session-abc",
        tokens_used=50,
        tokens_remaining=3950,
        evictions_triggered=0,
        processing_time_ms=150.5
    )
    assert response.tokens_used == 50
    assert response.tokens_remaining == 3950
    print(f"  ✓ Created proxy response")
    return response


def main():
    """Run all model tests."""
    print("=" * 50)
    print("Testing LiveContext Pydantic Models")
    print("=" * 50)
    
    try:
        test_context_message()
        test_eviction()
        test_context_snapshot()
        test_session_info()
        test_streaming_event()
        test_proxy_request()
        test_proxy_response()
        
        print("\n" + "=" * 50)
        print("✅ All model tests passed!")
        print("=" * 50)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
