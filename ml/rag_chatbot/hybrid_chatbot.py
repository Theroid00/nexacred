#!/usr/bin/env python3
"""
Hybrid NexaCred Chatbot
======================

A unified chatbot that combines:
1. Fast template responses for common queries (instant responses)
2. Real RAG with IBM Granite 3.3 2B for complex questions (AI responses)
3. Proper conversation memory with persistent storage
4. Fallback mechanisms for reliability

This solves all the identified issues while maintaining speed and accuracy.
"""

import sys
import time
import logging
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from conversation_memory import ConversationMemory, ChatMessage, get_conversation_memory
from fast_chatbot import FastFinancialChatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridNexaCredChatbot:
    """
    Hybrid chatbot that intelligently chooses between fast templates and real AI.

    Features:
    - Fast template responses for common financial terms and simple questions
    - Real RAG with IBM Granite 3.3 2B for complex questions and analysis
    - Conversation memory that persists across sessions
    - Intelligent routing between fast and AI responses
    - Fallback mechanisms for reliability
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the hybrid chatbot."""
        self.config = config or Config()

        # Initialize conversation memory
        self.conversation_memory = get_conversation_memory()
        self.current_session_id = None

        # Initialize fast chatbot (always available)
        self.fast_chatbot = FastFinancialChatbot()

        # Real RAG pipeline (loaded on demand)
        self._rag_pipeline = None
        self._rag_available = False

        # Track performance and usage
        self.stats = {
            "fast_responses": 0,
            "ai_responses": 0,
            "errors": 0,
            "session_count": 0
        }

        logger.info("HybridNexaCredChatbot initialized successfully")

    def start_new_conversation(self, user_id: str = "default_user") -> str:
        """Start a new conversation session."""
        self.current_session_id = self.conversation_memory.start_conversation(user_id)
        self.stats["session_count"] += 1

        # Also start conversation in fast chatbot for consistency
        self.fast_chatbot.current_session_id = self.current_session_id

        logger.info(f"Started new conversation: {self.current_session_id}")
        return self.current_session_id

    def generate_response(self, user_input: str, session_id: Optional[str] = None) -> str:
        """
        Generate response using intelligent routing between fast and AI systems.

        Routing Logic:
        1. Check for exact matches in fast template system
        2. For complex questions, use real RAG with IBM Granite
        3. Fallback to enhanced fast responses
        4. Always maintain conversation memory
        """
        try:
            # Use provided session or current one
            if session_id:
                self.current_session_id = session_id
            elif not self.current_session_id:
                self.current_session_id = self.start_new_conversation()

            # Add user message to memory
            self._add_user_message(user_input)

            # Determine response strategy
            response_strategy = self._determine_response_strategy(user_input)

            if response_strategy == "fast":
                response = self._get_fast_response(user_input)
                self.stats["fast_responses"] += 1

            elif response_strategy == "ai" and self._ensure_rag_available():
                response = self._get_ai_response(user_input)
                self.stats["ai_responses"] += 1

            else:
                # Fallback to enhanced fast response
                response = self._get_enhanced_fast_response(user_input)
                self.stats["fast_responses"] += 1

            # Add assistant response to memory
            self._add_assistant_message(response)

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self.stats["errors"] += 1
            error_response = "I apologize, but I encountered an error. Please try rephrasing your question."
            self._add_assistant_message(error_response)
            return error_response

    def _determine_response_strategy(self, user_input: str) -> str:
        """
        Determine whether to use fast templates or AI based on query complexity.

        Returns:
            "fast" for template responses, "ai" for complex queries
        """
        user_input_lower = user_input.lower()

        # Fast response indicators
        fast_indicators = [
            # Simple definitions
            "what is", "define", "meaning of", "explain",
            # Quick facts
            "emi", "roi", "cibil", "credit score", "kyc", "neft", "rtgs", "upi",
            # Simple how-to
            "how to check", "how to improve", "how to apply"
        ]

        # AI response indicators (complex queries)
        ai_indicators = [
            # Analysis and comparison
            "compare", "analyze", "which is better", "should i choose",
            "recommend", "suggest", "advice", "opinion",
            # Complex scenarios
            "my situation", "in my case", "i have", "i am", "i want to",
            # Multi-part questions
            "and also", "additionally", "furthermore", "what about",
            # Previous conversation references
            "you mentioned", "earlier you said", "from before"
        ]

        # Check for AI indicators first (more specific)
        for indicator in ai_indicators:
            if indicator in user_input_lower:
                return "ai"

        # Check for fast indicators
        for indicator in fast_indicators:
            if indicator in user_input_lower:
                return "fast"

        # Check conversation context for complexity
        if self._has_conversation_context():
            return "ai"

        # Default to fast for unknown queries
        return "fast"

    def _get_fast_response(self, user_input: str) -> str:
        """Get response from fast template system."""
        return self.fast_chatbot.generate_response(user_input, self.current_session_id)

    def _get_ai_response(self, user_input: str) -> str:
        """Get response from real RAG with IBM Granite."""
        try:
            return self._rag_pipeline.query(user_input, self.current_session_id, include_context=True)
        except Exception as e:
            logger.warning(f"AI response failed, falling back to fast: {e}")
            return self._get_enhanced_fast_response(user_input)

    def _get_enhanced_fast_response(self, user_input: str) -> str:
        """Get enhanced fast response with conversation context."""
        base_response = self.fast_chatbot.generate_response(user_input, self.current_session_id)

        # Add conversation context awareness
        if self._has_conversation_context():
            context_addition = " Based on our previous discussion, this information should help with your financial planning."
            return base_response + context_addition

        return base_response

    def _ensure_rag_available(self) -> bool:
        """Ensure RAG pipeline is loaded and available."""
        if self._rag_available and self._rag_pipeline:
            return True

        try:
            if not self._rag_pipeline:
                logger.info("Loading RAG pipeline with IBM Granite model...")
                from pipeline.rag import create_rag_pipeline
                self._rag_pipeline = create_rag_pipeline(self.config)

            self._rag_available = True
            logger.info("RAG pipeline is ready")
            return True

        except Exception as e:
            logger.warning(f"RAG pipeline unavailable: {e}")
            self._rag_available = False
            return False

    def _has_conversation_context(self) -> bool:
        """Check if there's meaningful conversation history."""
        if not self.current_session_id:
            return False

        conversation = self.conversation_memory.get_conversation(self.current_session_id)
        return conversation and len(conversation.messages) > 1

    def get_conversation_context(self, session_id: str, max_messages: int = 6) -> str:
        """Get conversation context for the fast chatbot compatibility."""
        return self.conversation_memory.get_recent_context(session_id, max_messages)

    def _add_user_message(self, content: str):
        """Add user message to conversation memory."""
        self.conversation_memory.add_message(
            self.current_session_id,
            ChatMessage(
                role="user",
                content=content,
                timestamp=time.time()
            )
        )

    def _add_assistant_message(self, content: str):
        """Add assistant message to conversation memory."""
        self.conversation_memory.add_message(
            self.current_session_id,
            ChatMessage(
                role="assistant",
                content=content,
                timestamp=time.time()
            )
        )

    def get_conversation_history(self, session_id: Optional[str] = None) -> List[Dict]:
        """Get conversation history for a session."""
        session_id = session_id or self.current_session_id
        if not session_id:
            return []

        conversation = self.conversation_memory.get_conversation(session_id)
        if not conversation:
            return []

        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in conversation.messages
        ]

    def clear_conversation(self, session_id: Optional[str] = None):
        """Clear conversation history."""
        session_id = session_id or self.current_session_id
        if session_id:
            self.conversation_memory.clear_conversation(session_id)
            if session_id == self.current_session_id:
                self.current_session_id = None
            logger.info(f"Cleared conversation: {session_id}")

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            **self.stats,
            "rag_available": self._rag_available,
            "total_responses": self.stats["fast_responses"] + self.stats["ai_responses"]
        }

    def cleanup(self):
        """Clean up resources."""
        try:
            if self._rag_pipeline:
                self._rag_pipeline.cleanup()
            logger.info("Hybrid chatbot cleanup completed")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def main():
    """Interactive CLI for the hybrid chatbot."""
    print("🤖 NexaCred Hybrid Financial Assistant")
    print("=" * 50)
    print("✨ Fast template responses + IBM Granite 3.3 2B AI")
    print("💾 Persistent conversation memory")
    print("🔄 Intelligent response routing")
    print("=" * 50)
    print("Commands: 'quit' to exit, 'clear' to start fresh, 'history' to see conversation, 'stats' for usage")
    print("=" * 50)

    chatbot = HybridNexaCredChatbot()
    session_id = chatbot.start_new_conversation("cli_user")

    try:
        while True:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_conversation()
                session_id = chatbot.start_new_conversation("cli_user")
                print("🔄 Conversation cleared!")
                continue
            elif user_input.lower() == 'history':
                history = chatbot.get_conversation_history()
                print("\n📝 Conversation History:")
                for msg in history:
                    role_emoji = "👤" if msg["role"] == "user" else "🤖"
                    print(f"{role_emoji} {msg['role'].title()}: {msg['content']}")
                continue
            elif user_input.lower() == 'stats':
                stats = chatbot.get_stats()
                print(f"\n📊 Usage Statistics:")
                print(f"Fast responses: {stats['fast_responses']}")
                print(f"AI responses: {stats['ai_responses']}")
                print(f"Total sessions: {stats['session_count']}")
                print(f"RAG available: {'✅ Yes' if stats['rag_available'] else '❌ No'}")
                continue
            elif not user_input:
                continue

            print("\n🤖 Assistant: ", end="", flush=True)
            response = chatbot.generate_response(user_input)
            print(response)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        chatbot.cleanup()


if __name__ == "__main__":
    main()
