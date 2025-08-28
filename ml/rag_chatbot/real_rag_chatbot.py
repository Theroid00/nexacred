#!/usr/bin/env python3
"""
Real RAG Chatbot with IBM Granite 3.3 2B Instruct
================================================

A proper RAG chatbot that uses the actual IBM Granite model for responses,
with conversation memory and document retrieval.
"""

import sys
import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from models.generator import load_generator_from_config, cleanup_model_memory
from models.embeddings import load_embedder
from retrieval.dummy import DummyRetriever
from pipeline.rag import RAGPipeline
from conversation_memory import ConversationMemory, ChatMessage, get_conversation_memory
from prompts import GRANITE_FINANCIAL_PROMPT

logger = logging.getLogger(__name__)


class RealRAGChatbot:
    """Real RAG chatbot using IBM Granite 3.3 2B Instruct model."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the real RAG chatbot."""
        self.config = config or Config()
        
        # Initialize components
        self.conversation_memory = get_conversation_memory()
        self.current_session_id = None
        
        # Will be loaded on first use
        self._rag_pipeline = None
        self._model = None
        self._tokenizer = None
        
        # Track initialization state
        self._initialized = False
        
        logger.info("RealRAGChatbot initialized (models will load on first use)")

    def _ensure_initialized(self):
        """Ensure all components are loaded and ready."""
        if self._initialized:
            return
            
        try:
            logger.info("Loading AI models... This may take a few minutes on first run.")
            
            # Load the IBM Granite model
            logger.info("Loading IBM Granite 3.3 2B Instruct model...")
            self._model, self._tokenizer = load_generator_from_config(self.config)
            
            # Initialize retriever
            retriever = DummyRetriever()
            
            # Initialize RAG pipeline
            self._rag_pipeline = RAGPipeline(retriever, self.config)
            
            self._initialized = True
            logger.info("✅ All components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def start_new_conversation(self, user_id: str = "default_user") -> str:
        """Start a new conversation session."""
        self.current_session_id = self.conversation_memory.start_conversation(user_id)
        logger.info(f"Started new conversation: {self.current_session_id}")
        return self.current_session_id

    def generate_response(self, user_input: str, session_id: Optional[str] = None) -> str:
        """Generate AI response using the real RAG pipeline."""
        try:
            # Ensure everything is initialized
            self._ensure_initialized()
            
            # Use provided session or current one
            if session_id:
                self.current_session_id = session_id
            elif not self.current_session_id:
                self.current_session_id = self.start_new_conversation()

            # Add user message to memory
            self.conversation_memory.add_message(
                self.current_session_id,
                ChatMessage(
                    role="user",
                    content=user_input,
                    timestamp=time.time()
                )
            )

            # Get conversation history for context
            conversation = self.conversation_memory.get_conversation(self.current_session_id)
            conversation_context = ""
            
            if conversation and len(conversation.messages) > 1:
                # Include last few exchanges for context
                recent_messages = conversation.messages[-6:]  # Last 3 exchanges
                for msg in recent_messages[:-1]:  # Exclude the current user message
                    conversation_context += f"{msg.role.title()}: {msg.content}\n"

            # Use RAG pipeline to generate response
            logger.info("Generating AI response...")
            response = self._rag_pipeline.query(
                user_input, 
                session_id=self.current_session_id,
                include_context=True
            )

            # Add assistant response to memory
            self.conversation_memory.add_message(
                self.current_session_id,
                ChatMessage(
                    role="assistant",
                    content=response,
                    timestamp=time.time()
                )
            )

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your request. Please try again. Error: {str(e)}"

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
            logger.info(f"Cleared conversation: {session_id}")

    def cleanup(self):
        """Clean up resources."""
        try:
            if self._model or self._tokenizer:
                cleanup_model_memory(self._model, self._tokenizer)
                self._model = None
                self._tokenizer = None
            logger.info("Cleaned up model resources")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def main():
    """Interactive CLI for the real RAG chatbot."""
    print("🤖 NexaCred Real RAG Chatbot with IBM Granite 3.3 2B Instruct")
    print("=" * 60)
    print("Initializing AI models... This may take a few minutes on first run.")
    print("Type 'quit' to exit, 'clear' to start fresh, 'history' to see conversation")
    print("=" * 60)

    chatbot = RealRAGChatbot()
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
