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
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create from dictionary (JSON retrieval)."""
        timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )


@dataclass
class Conversation:
    """Represents a complete conversation session."""
    session_id: str
    user_id: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create from dictionary (JSON retrieval)."""
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        updated_at = datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]

        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            messages=[ChatMessage.from_dict(msg) for msg in data["messages"]],
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get("metadata", {})
        )


class ConversationMemory:
    """
    Manages conversation history and provides context for RAG chatbot using local file storage.

    Features:
    - Stores conversations in local JSON files
    - Maintains session-based chat history
    - Provides conversation context for AI responses
    - Handles memory cleanup and management
    - No external database dependencies
    """

    def __init__(
        self,
        storage_dir: str = None,
        max_context_messages: int = 10,
        enable_persistence: bool = True
    ):
        """
        Initialize conversation memory manager with local file storage.

        Args:
            storage_dir: Directory to store conversation files (default: ./conversation_storage)
            max_context_messages: Maximum messages to keep in context
            enable_persistence: Whether to persist conversations to files
        """
        # Set up storage directory
        if storage_dir is None:
            current_dir = Path(__file__).parent
            storage_dir = current_dir / "conversation_storage"

        self.storage_dir = Path(storage_dir)
        self.max_context_messages = max_context_messages
        self.enable_persistence = enable_persistence

        # Create storage directory if it doesn't exist
        if self.enable_persistence:
            self.storage_dir.mkdir(exist_ok=True)
            logger.info(f"Conversation storage directory: {self.storage_dir}")

        # In-memory conversation storage for current session
        self.current_conversations = {}  # session_id -> Conversation

        logger.info(f"Conversation memory initialized with local file storage: {self.enable_persistence}")

    def start_conversation(self, user_id: str = "default_user") -> str:
        """
        Start a new conversation session.

        Args:
            user_id: Identifier for the user

        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(UTC)

        conversation = Conversation(
            session_id=session_id,
            user_id=user_id,
            messages=[],
            created_at=now,
            updated_at=now,
            metadata={"source": "rag_chatbot"}
        )

        self.current_conversations[session_id] = conversation

        logger.info(f"Started new conversation session: {session_id}")
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to the conversation.

        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Additional message metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            if session_id not in self.current_conversations:
                # Try to load from file
                if not self._load_conversation(session_id):
                    # Create new conversation if not found
                    self.start_conversation()
                    session_id = list(self.current_conversations.keys())[-1]

            conversation = self.current_conversations[session_id]

            message = ChatMessage(
                role=role,
                content=content,
                timestamp=datetime.now(UTC),
                metadata=metadata
            )

            conversation.messages.append(message)
            conversation.updated_at = datetime.now(UTC)

            # Trim conversation if too long
            if len(conversation.messages) > self.max_context_messages * 2:
                # Keep recent messages
                conversation.messages = conversation.messages[-self.max_context_messages:]

            # Persist to file if enabled
            if self.enable_persistence:
                self._save_conversation(conversation)

            logger.debug(f"Added {role} message to session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            return False

    def get_conversation_context(self, session_id: str, max_messages: Optional[int] = None) -> str:
        """
        Get formatted conversation context for AI prompts.

        Args:
            session_id: Session identifier
            max_messages: Maximum messages to include (default: max_context_messages)

        Returns:
            Formatted conversation context string
        """
        try:
            if session_id not in self.current_conversations:
                if not self._load_conversation(session_id):
                    return ""

            conversation = self.current_conversations[session_id]
            max_msgs = max_messages or self.max_context_messages

            # Get recent messages
            recent_messages = conversation.messages[-max_msgs:] if conversation.messages else []

            if not recent_messages:
                return ""

            # Format for AI context
            context_parts = ["Previous conversation:"]
            for msg in recent_messages:
                role_label = "User" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role_label}: {msg.content}")

            context_parts.append("Current question:")

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return ""

    def get_conversation_history(self, session_id: str) -> List[ChatMessage]:
        """
        Get full conversation history.

        Args:
            session_id: Session identifier

        Returns:
            List of chat messages
        """
        try:
            if session_id not in self.current_conversations:
                if not self._load_conversation(session_id):
                    return []

            return self.current_conversations[session_id].messages.copy()

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    def clear_conversation(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        try:
            if session_id in self.current_conversations:
                del self.current_conversations[session_id]

            # Remove from file storage if persistence is enabled
            if self.enable_persistence:
                conversation_file = self.storage_dir / f"{session_id}.json"
                if conversation_file.exists():
                    conversation_file.unlink()

            logger.info(f"Cleared conversation session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear conversation: {e}")
            return False

    def _load_conversation(self, session_id: str) -> bool:
        """Load conversation from local file."""
        try:
            if not self.enable_persistence:
                return False

            conversation_file = self.storage_dir / f"{session_id}.json"
            if conversation_file.exists():
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                conversation = Conversation.from_dict(data)
                self.current_conversations[session_id] = conversation
                logger.debug(f"Loaded conversation from file: {session_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to load conversation from file: {e}")
            return False

    def _save_conversation(self, conversation: Conversation) -> bool:
        """Save conversation to local file."""
        try:
            if not self.enable_persistence:
                return False

            conversation_file = self.storage_dir / f"{conversation.session_id}.json"
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation.to_dict(), f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved conversation to file: {conversation.session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save conversation to file: {e}")
            return False

    def get_user_conversations(self, user_id: str, limit: int = 10) -> List[str]:
        """
        Get list of conversation session IDs for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return

        Returns:
            List of session IDs
        """
        try:
            session_ids = []

            # Check current conversations
            for sid, conv in self.current_conversations.items():
                if conv.user_id == user_id:
                    session_ids.append(sid)

            # Check stored conversations if persistence is enabled
            if self.enable_persistence and self.storage_dir.exists():
                for conversation_file in self.storage_dir.glob("*.json"):
                    session_id = conversation_file.stem
                    if session_id not in session_ids:
                        try:
                            with open(conversation_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            if data.get("user_id") == user_id:
                                session_ids.append(session_id)
                        except Exception:
                            continue

            return session_ids[:limit]

        except Exception as e:
            logger.error(f"Failed to get user conversations: {e}")
            return []

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation memory."""
        stats = {
            "current_sessions": len(self.current_conversations),
            "persistence_enabled": self.enable_persistence,
            "max_context_messages": self.max_context_messages,
            "storage_type": "local_files",
            "storage_dir": str(self.storage_dir) if self.enable_persistence else None
        }

        if self.enable_persistence and self.storage_dir.exists():
            try:
                conversation_files = list(self.storage_dir.glob("*.json"))
                stats["total_stored_conversations"] = len(conversation_files)
                stats["storage_size_mb"] = sum(f.stat().st_size for f in conversation_files) / (1024 * 1024)
            except Exception:
                stats["total_stored_conversations"] = "unknown"
                stats["storage_size_mb"] = "unknown"

        return stats

    def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """
        Clean up conversations older than specified days.

        Args:
            days_old: Remove conversations older than this many days

        Returns:
            Number of conversations removed
        """
        if not self.enable_persistence or not self.storage_dir.exists():
            return 0

        removed_count = 0
        cutoff_time = datetime.now(UTC).timestamp() - (days_old * 24 * 60 * 60)

        try:
            for conversation_file in self.storage_dir.glob("*.json"):
                try:
                    # Check file modification time
                    if conversation_file.stat().st_mtime < cutoff_time:
                        conversation_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old conversation: {conversation_file.stem}")
                except Exception:
                    continue

            logger.info(f"Cleaned up {removed_count} old conversations")
            return removed_count

        except Exception as e:
            logger.error(f"Failed to cleanup old conversations: {e}")
            return 0


# Global conversation memory instance
conversation_memory = None

def get_conversation_memory() -> ConversationMemory:
    """Get or create global conversation memory instance with local file storage."""
    global conversation_memory
    if conversation_memory is None:
        conversation_memory = ConversationMemory(enable_persistence=True)
    return conversation_memory


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
