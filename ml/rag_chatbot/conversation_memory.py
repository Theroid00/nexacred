"""
Conversation Memory Manager with Local File Storage
=================================================

Manages conversation history and context for the RAG chatbot system.
Stores conversations in local JSON files instead of MongoDB for persistence.
"""

import logging
import uuid
import json
import os
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float  # Use float timestamp for simplicity
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create from dictionary (JSON retrieval)."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {})
        )


@dataclass
class Conversation:
    """Represents a complete conversation with metadata."""
    session_id: str
    user_id: str
    messages: List[ChatMessage]
    created_at: float
    updated_at: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create from dictionary (JSON retrieval)."""
        messages = [ChatMessage.from_dict(msg) for msg in data.get("messages", [])]

        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            messages=messages,
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=data.get("metadata", {})
        )


class ConversationMemory:
    """Manages conversation history with local file storage."""

    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize conversation memory with local storage."""
        self.storage_dir = Path(storage_dir) if storage_dir else Path(__file__).parent / "conversation_storage"
        self.storage_dir.mkdir(exist_ok=True)
        logger.info(f"Conversation storage directory: {self.storage_dir}")

    def _get_conversation_file(self, session_id: str) -> Path:
        """Get the file path for a conversation."""
        return self.storage_dir / f"{session_id}.json"

    def start_conversation(self, user_id: str) -> str:
        """Start a new conversation and return session ID."""
        import time
        session_id = str(uuid.uuid4())
        current_time = time.time()

        conversation = Conversation(
            session_id=session_id,
            user_id=user_id,
            messages=[],
            created_at=current_time,
            updated_at=current_time
        )

        self._save_conversation(conversation)
        logger.info(f"Started new conversation: {session_id} for user: {user_id}")
        return session_id

    def add_message(self, session_id: str, message: ChatMessage) -> bool:
        """Add a message to the conversation."""
        try:
            conversation = self.get_conversation(session_id)
            if not conversation:
                logger.warning(f"Conversation {session_id} not found")
                return False

            conversation.messages.append(message)
            conversation.updated_at = message.timestamp

            self._save_conversation(conversation)
            logger.debug(f"Added {message.role} message to conversation {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding message to conversation {session_id}: {e}")
            return False

    def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Retrieve a conversation by session ID."""
        try:
            file_path = self._get_conversation_file(session_id)
            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Conversation.from_dict(data)

        except Exception as e:
            logger.error(f"Error loading conversation {session_id}: {e}")
            return None

    def _save_conversation(self, conversation: Conversation) -> bool:
        """Save conversation to file."""
        try:
            file_path = self._get_conversation_file(conversation.session_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)
            return True

        except Exception as e:
            logger.error(f"Error saving conversation {conversation.session_id}: {e}")
            return False

    def get_user_conversations(self, user_id: str) -> List[str]:
        """Get all conversation session IDs for a user."""
        session_ids = []
        try:
            for file_path in self.storage_dir.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if data.get("user_id") == user_id:
                            session_ids.append(data["session_id"])
                except Exception as e:
                    logger.warning(f"Error reading conversation file {file_path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error listing conversations for user {user_id}: {e}")

        return session_ids

    def clear_conversation(self, session_id: str) -> bool:
        """Clear/delete a conversation."""
        try:
            file_path = self._get_conversation_file(session_id)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Cleared conversation: {session_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error clearing conversation {session_id}: {e}")
            return False

    def get_recent_context(self, session_id: str, max_messages: int = 10) -> str:
        """Get recent conversation context as formatted string."""
        conversation = self.get_conversation(session_id)
        if not conversation or not conversation.messages:
            return ""

        recent_messages = conversation.messages[-max_messages:]
        context_parts = []

        for msg in recent_messages:
            role_name = "User" if msg.role == "user" else "Assistant"
            context_parts.append(f"{role_name}: {msg.content}")

        return "\n".join(context_parts)


# Global conversation memory instance
_global_conversation_memory = None


def get_conversation_memory() -> ConversationMemory:
    """Get the global conversation memory instance."""
    global _global_conversation_memory
    if _global_conversation_memory is None:
        _global_conversation_memory = ConversationMemory()
    return _global_conversation_memory


if __name__ == "__main__":
    """Test conversation memory functionality with local file storage."""
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🧠 Testing Conversation Memory (Local File Storage)...")

    # Test conversation memory
    memory = ConversationMemory(enable_persistence=True)

    # Start a conversation
    session_id = memory.start_conversation("test_user")
    print(f"✅ Started session: {session_id}")

    # Add some messages
    memory.add_message(session_id, "user", "What is a credit score?")
    memory.add_message(session_id, "assistant", "A credit score is a numerical representation of your creditworthiness...")
    memory.add_message(session_id, "user", "How can I improve it?")

    # Get context
    context = memory.get_conversation_context(session_id)
    print(f"📝 Context:\n{context}")

    # Get stats
    stats = memory.get_memory_stats()
    print(f"📊 Memory Stats: {stats}")

    # Test persistence
    print("\n🔄 Testing persistence...")
    memory2 = ConversationMemory(enable_persistence=True)
    loaded_context = memory2.get_conversation_context(session_id)
    print(f"📂 Loaded context: {'✅ Success' if loaded_context else '❌ Failed'}")

    print("✅ Conversation memory test completed!")
