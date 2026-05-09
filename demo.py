#!/usr/bin/env python3
"""LiveContext Demo - Generate synthetic data and demonstrate the system."""

import asyncio
import json
import logging
import random
import sys
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

# Add project to path
sys.path.insert(0, "/home/daksh/7May/projects/livecontext")

from livecontext.core.embedder import Embedder
from livecontext.core.tokenizer import Tokenizer
from livecontext.server.db import Database, get_db
from livecontext.server.models import (
    ContextMessage,
    ContextSnapshot,
    Eviction,
    EvictionStrategy,
    Role,
    SessionInfo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DemoDataGenerator:
    """Generate realistic synthetic data for LiveContext demo."""
    
    SAMPLE_CONVERSATIONS = [
        {
            "system": "You are a helpful coding assistant. Help the user with their programming questions.",
            "turns": [
                ("user", "How do I reverse a list in Python?"),
                ("assistant", "You can reverse a list in Python using several methods:\n\n1. Using the reverse() method:\n```python\nmy_list = [1, 2, 3, 4, 5]\nmy_list.reverse()\nprint(my_list)  # [5, 4, 3, 2, 1]\n```\n\n2. Using slicing:\n```python\nmy_list = [1, 2, 3, 4, 5]\nreversed_list = my_list[::-1]\nprint(reversed_list)  # [5, 4, 3, 2, 1]\n```\n\n3. Using reversed() function:\n```python\nmy_list = [1, 2, 3, 4, 5]\nreversed_list = list(reversed(my_list))\nprint(reversed_list)  # [5, 4, 3, 2, 1]\n```\n\nThe slicing method [::-1] creates a new reversed list without modifying the original, while reverse() modifies the list in place."),
                ("user", "Which method is most efficient?"),
                ("assistant", "For most use cases, the slicing method `[::-1]` is the most efficient because:\n\n1. It's implemented in C and very fast\n2. It works on any sequence type\n3. It's concise and readable\n\nHowever, if you need to reverse a list in-place (to save memory), use `reverse()`. The `reversed()` function is useful when you just need to iterate backwards without creating a new list."),
            ]
        },
        {
            "system": "You are a data science assistant. Help with data analysis and machine learning questions.",
            "turns": [
                ("user", "What's the difference between precision and recall?"),
                ("assistant", "Precision and recall are two important metrics in classification:\n\n**Precision**: Of all positive predictions, how many were actually correct?\n- Formula: True Positives / (True Positives + False Positives)\n- Focus: Quality of positive predictions\n- High when: Few false alarms\n\n**Recall**: Of all actual positives, how many did we correctly identify?\n- Formula: True Positives / (True Positives + False Negatives)\n- Focus: Coverage of positive class\n- High when: Few missed positives\n\n**Trade-off**: Usually, increasing precision decreases recall and vice versa. The F1 score balances both."),
                ("user", "When would I prioritize one over the other?"),
                ("assistant", "Choose based on your use case:\n\n**Prioritize Precision when:**\n- False positives are costly (e.g., spam detection - don't mark important emails as spam)\n- You want to be confident in positive predictions\n- Example: Medical diagnosis confirmation\n\n**Prioritize Recall when:**\n- False negatives are costly (e.g., fraud detection - don't miss actual fraud)\n- Missing positives is dangerous\n- Example: Cancer screening\n\n**Balanced approach:**\n- Use F1 score when both matter equally\n- Use ROC-AUC for overall model performance\n- Adjust classification threshold based on business needs"),
            ]
        },
        {
            "system": "You are a DevOps assistant. Help with infrastructure and deployment questions.",
            "turns": [
                ("user", "What's the difference between Docker and Kubernetes?"),
                ("assistant", "Docker and Kubernetes serve different but complementary purposes:\n\n**Docker**\n- Container runtime that packages applications\n- Creates isolated environments for apps\n- Runs on a single machine\n- Analogous to: A shipping container\n\n**Kubernetes**\n- Container orchestration platform\n- Manages multiple containers across machines\n- Handles scaling, load balancing, self-healing\n- Analogous to: A port authority managing thousands of containers\n\n**Relationship**: You use Docker to create containers, then Kubernetes to manage them at scale."),
                ("user", "Do I need Kubernetes for a small project?"),
                ("assistant", "For small projects, you probably don't need Kubernetes:\n\n**Use Docker alone if:**\n- Single server deployment\n- Few services (< 5)\n- Simple scaling needs\n- Small team\n- Use Docker Compose for multi-container apps\n\n**Consider Kubernetes when:**\n- Multiple servers/clusters\n- Complex microservices\n- Need auto-scaling, self-healing\n- Multi-environment management\n- Large team with DevOps expertise\n\n**Alternatives**: Docker Swarm, AWS ECS, Google Cloud Run, Heroku"),
            ]
        },
        {
            "system": "You are a React expert. Help with frontend development questions.",
            "turns": [
                ("user", "When should I use useEffect vs useLayoutEffect?"),
                ("assistant", "Both hooks run side effects, but at different times:\n\n**useEffect**\n- Runs after browser paint\n- Non-blocking, doesn't delay rendering\n- Use for: Data fetching, subscriptions, manual DOM changes\n- Most common choice\n\n**useLayoutEffect**\n- Runs before browser paint (synchronous)\n- Blocking, can delay rendering\n- Use for: Measuring DOM, preventing visual flicker, animations\n- Rarely needed\n\n**Rule of thumb**: Start with useEffect. Only switch to useLayoutEffect if you see visual inconsistencies."),
                ("user", "Can you show an example of useLayoutEffect?"),
                ("assistant", "Here's a useLayoutEffect example for measuring DOM elements:\n\n```jsx\nfunction Tooltip({ children, targetRef }) {\n  const tooltipRef = useRef();\n  const [position, setPosition] = useState({ x: 0, y: 0 });\n\n  useLayoutEffect(() => {\n    // Measure before paint to prevent flicker\n    const targetRect = targetRef.current.getBoundingClientRect();\n    const tooltipRect = tooltipRef.current.getBoundingClientRect();\n    \n    setPosition({\n      x: targetRect.left + (targetRect.width - tooltipRect.width) / 2,\n      y: targetRect.top - tooltipRect.height - 8\n    });\n  }, [targetRef]);\n\n  return (\n    <div\n      ref={tooltipRef}\n      style={{ position: 'absolute', left: position.x, top: position.y }}\n    >\n      {children}\n    </div>\n  );\n}\n```\n\nWithout useLayoutEffect, the tooltip would briefly appear at the wrong position before correcting."),
            ]
        },
        {
            "system": "You are a SQL expert. Help with database design and queries.",
            "turns": [
                ("user", "What's the difference between INNER JOIN and LEFT JOIN?"),
                ("assistant", "JOIN types determine which rows are returned:\n\n**INNER JOIN**\n- Returns only matching rows from both tables\n- Excludes rows without matches\n- Most restrictive\n\n**LEFT JOIN**\n- Returns all rows from left table\n- Matching rows from right table\n- NULL for non-matching right rows\n- More inclusive\n\n**Example:**\n```sql\n-- Only users with orders\nSELECT u.name, o.total\nFROM users u\nINNER JOIN orders o ON u.id = o.user_id;\n\n-- All users, NULL if no orders\nSELECT u.name, o.total\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id;\n```"),
                ("user", "When should I use RIGHT JOIN or FULL JOIN?"),
                ("assistant", "These joins are less common but useful in specific cases:\n\n**RIGHT JOIN**\n- Returns all rows from right table\n- Rarely used (just swap table order for LEFT JOIN)\n- Example: When you can't change table order in generated SQL\n\n**FULL OUTER JOIN**\n- Returns all rows from both tables\n- NULL where no match exists\n- Use cases:\n  - Data reconciliation (find mismatches)\n  - Combining datasets\n  - Reporting on all entities\n\n**Example:**\n```sql\n-- Find all customers and orders, even unmatched\nSELECT c.name, o.id\nFROM customers c\nFULL OUTER JOIN orders o ON c.id = o.customer_id;\n```\n\nNote: MySQL doesn't support FULL JOIN directly (use UNION)."),
            ]
        },
    ]
    
    def __init__(self, db: Optional[Database] = None):
        """Initialize generator."""
        self.db = db or get_db()
        self.tokenizer = Tokenizer()
        self.embedder = Embedder()
        
    def generate_session(
        self,
        model_name: str = "gpt-4",
        provider: str = "openai",
        max_tokens: int = 4096
    ) -> SessionInfo:
        """Generate a new session."""
        session = SessionInfo(
            model_name=model_name,
            provider=provider,
            max_tokens=max_tokens
        )
        self.db.create_session(session)
        logger.info(f"Created session: {session.id}")
        return session
    
    def generate_conversation(
        self,
        session_id: str,
        conversation_idx: int = 0,
        num_turns: Optional[int] = None
    ) -> List[ContextMessage]:
        """Generate a conversation for a session."""
        conv = self.SAMPLE_CONVERSATIONS[conversation_idx % len(self.SAMPLE_CONVERSATIONS)]
        
        messages = []
        
        # Add system message
        system_msg = ContextMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=Role.SYSTEM,
            content=conv["system"],
            timestamp=datetime.now() - timedelta(minutes=30),
            token_count=self.tokenizer.count_tokens(conv["system"]),
            importance_score=1.0
        )
        self.db.save_message(session_id, system_msg)
        messages.append(system_msg)
        
        # Add conversation turns
        turns = conv["turns"][:num_turns] if num_turns else conv["turns"]
        base_time = datetime.now() - timedelta(minutes=20)
        
        for i, (role, content) in enumerate(turns):
            msg = ContextMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role=Role(role),
                content=content,
                timestamp=base_time + timedelta(minutes=i * 2),
                token_count=self.tokenizer.count_tokens(content),
                importance_score=random.uniform(0.7, 1.0)
            )
            self.db.save_message(session_id, msg)
            messages.append(msg)
        
        return messages
    
    def generate_snapshot(
        self,
        session_id: str,
        messages: List[ContextMessage],
        max_tokens: int = 4096
    ) -> ContextSnapshot:
        """Generate a snapshot from messages."""
        total_tokens = sum(m.token_count for m in messages)
        utilization = (total_tokens / max_tokens) * 100 if max_tokens > 0 else 0
        
        snapshot = ContextSnapshot(
            id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(),
            total_tokens=total_tokens,
            max_tokens=max_tokens,
            utilization_percent=utilization,
            messages=messages,
            evictions=[]
        )
        
        self.db.save_snapshot(snapshot)
        return snapshot
    
    def generate_eviction(
        self,
        session_id: str,
        message: ContextMessage,
        strategy: EvictionStrategy = EvictionStrategy.SLIDING_WINDOW,
        reason: str = "Token budget exceeded"
    ) -> Eviction:
        """Generate an eviction record."""
        eviction = Eviction(
            id=str(uuid.uuid4()),
            session_id=session_id,
            message_id=message.id,
            strategy=strategy,
            reason=reason,
            token_savings=message.token_count,
            timestamp=datetime.now(),
            similarity_score=random.uniform(0.5, 0.9) if strategy == EvictionStrategy.SIMILARITY_MERGE else None
        )
        
        self.db.save_eviction(session_id, eviction)
        return eviction
    
    def generate_full_demo(
        self,
        num_sessions: int = 3,
        num_conversations: int = 5
    ) -> dict:
        """Generate a complete demo dataset."""
        logger.info("=" * 60)
        logger.info("GENERATING LIVECONTEXT DEMO DATA")
        logger.info("=" * 60)
        
        sessions = []
        all_messages = []
        all_snapshots = []
        all_evictions = []
        
        # Generate sessions
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]
        providers = ["openai", "openai", "anthropic", "anthropic"]
        
        for i in range(num_sessions):
            session = self.generate_session(
                model_name=models[i % len(models)],
                provider=providers[i % len(providers)],
                max_tokens=random.choice([4096, 8192, 16384, 32768])
            )
            sessions.append(session)
            
            # Generate conversations for this session
            for j in range(num_conversations):
                messages = self.generate_conversation(
                    session.id,
                    conversation_idx=j,
                    num_turns=random.randint(3, 6)
                )
                all_messages.extend(messages)
                
                # Generate snapshot
                snapshot = self.generate_snapshot(
                    session.id,
                    messages,
                    session.max_tokens
                )
                all_snapshots.append(snapshot)
                
                # Maybe generate some evictions
                if random.random() < 0.3 and len(messages) > 3:
                    evicted_msg = random.choice(messages[1:])  # Don't evict system
                    eviction = self.generate_eviction(
                        session.id,
                        evicted_msg,
                        strategy=random.choice(list(EvictionStrategy)),
                        reason=random.choice([
                            "Token budget exceeded",
                            "Similar message found",
                            "Low importance score",
                            "Sliding window eviction"
                        ])
                    )
                    all_evictions.append(eviction)
        
        result = {
            "sessions": sessions,
            "messages": all_messages,
            "snapshots": all_snapshots,
            "evictions": all_evictions
        }
        
        logger.info("=" * 60)
        logger.info("DEMO DATA GENERATION COMPLETE")
        logger.info(f"  Sessions: {len(sessions)}")
        logger.info(f"  Messages: {len(all_messages)}")
        logger.info(f"  Snapshots: {len(all_snapshots)}")
        logger.info(f"  Evictions: {len(all_evictions)}")
        logger.info("=" * 60)
        
        return result


def print_demo_summary(data: dict):
    """Print a summary of the generated demo data."""
    print("\n" + "=" * 60)
    print("LIVECONTEXT DEMO DATA SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Generated {len(data['sessions'])} sessions:")
    for session in data['sessions']:
        print(f"   • {session.id[:16]}... ({session.model_name})")
    
    print(f"\n💬 Total messages: {len(data['messages'])}")
    
    by_role = {}
    for msg in data['messages']:
        role = msg.role.value
        by_role[role] = by_role.get(role, 0) + 1
    
    for role, count in sorted(by_role.items()):
        print(f"   • {role}: {count}")
    
    print(f"\n📸 Total snapshots: {len(data['snapshots'])}")
    
    total_tokens = sum(s.total_tokens for s in data['snapshots'])
    avg_utilization = sum(s.utilization_percent for s in data['snapshots']) / len(data['snapshots']) if data['snapshots'] else 0
    
    print(f"   • Total tokens tracked: {total_tokens:,}")
    print(f"   • Average utilization: {avg_utilization:.1f}%")
    
    print(f"\n🗑️  Total evictions: {len(data['evictions'])}")
    
    total_saved = sum(e.token_savings for e in data['evictions'])
    print(f"   • Tokens saved: {total_saved:,}")
    
    print("\n" + "=" * 60)
    print("✅ Demo data ready! Start the server with:")
    print("   python -m livecontext.cli serve")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LiveContext Demo Data Generator")
    parser.add_argument("--sessions", "-s", type=int, default=3, help="Number of sessions")
    parser.add_argument("--conversations", "-c", type=int, default=5, help="Conversations per session")
    parser.add_argument("--clean", action="store_true", help="Clean existing data first")
    
    args = parser.parse_args()
    
    # Initialize database
    db = get_db()
    
    if args.clean:
        print("🧹 Cleaning existing data...")
        # Note: In production, you'd have a proper cleanup method
        import os
        db_path = os.environ.get("LIVECONTEXT_DB_PATH", "data/livecontext.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"   Removed {db_path}")
        db = get_db()  # Re-initialize
    
    # Generate demo data
    generator = DemoDataGenerator(db)
    data = generator.generate_full_demo(
        num_sessions=args.sessions,
        num_conversations=args.conversations
    )
    
    # Print summary
    print_demo_summary(data)
    
    # Show sample data
    print("\n📋 Sample Session Details:")
    print("-" * 60)
    
    for session in data['sessions'][:2]:
        print(f"\nSession: {session.id}")
        print(f"  Model: {session.model_name}")
        print(f"  Provider: {session.provider}")
        print(f"  Max Tokens: {session.max_tokens:,}")
        print(f"  Created: {session.created_at}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
