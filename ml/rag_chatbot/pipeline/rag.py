"""
RAG Pipeline
============

Orchestrates the complete RAG (Retrieval-Augmented Generation) workflow:
1. Document retrieval based on query
2. Context construction with retrieved documents
3. Prompt formatting for Granite model
4. Text generation with retrieved context
5. Conversation memory management

This is the main orchestration component that ties together:
- Embedding models for query encoding
- Retrievers for document search
- Text generators for response creation
- Conversation memory for context continuity
"""

import logging
import time
from typing import List, Dict, Any, Optional
import numpy as np

from ..config import Config
from ..models.generator import load_generator_from_config, generate, cleanup_model_memory
from ..models.embeddings import load_embedder, embed_texts
from ..retrieval.base import AbstractRetriever
from ..prompts import GRANITE_FINANCIAL_PROMPT
from .token_utils import count_tokens, truncate_context
from ..conversation_memory import ConversationMemory, ChatMessage, get_conversation_memory

logger = logging.getLogger(__name__)


class RAGPipelineError(Exception):
    """Custom exception for RAG pipeline errors."""
    pass


class RAGPipeline:
    """
    Complete RAG pipeline for Indian financial regulation chatbot with conversation memory.

    Coordinates embedding, retrieval, prompt construction, generation, and memory management
    to provide contextual responses about financial regulations with conversation continuity.
    """
    
    def __init__(self, retriever: AbstractRetriever, config: Optional[Config] = None):
        """
        Initialize RAG pipeline with retriever and configuration.
        
        Args:
            retriever: Document retriever implementation
            config: Configuration object (creates default if None)
        """
        self.config = config or Config()
        self.retriever = retriever
        
        # Models will be loaded on first use (lazy loading)
        self._generator = None
        self._embedder = None
        self._tokenizer = None
        self._model_load_attempts = 0
        self._max_retries = 3

        # Initialize conversation memory
        self.conversation_memory = get_conversation_memory()
        self.current_session_id = None

    def start_conversation(self, user_id: str = "default_user") -> str:
        """
        Start a new conversation session.

        Args:
            user_id: User identifier for the conversation

        Returns:
            session_id: Unique session identifier
        """
        self.current_session_id = self.conversation_memory.start_conversation(user_id)
        logger.info(f"Started new conversation session: {self.current_session_id}")
        return self.current_session_id

    def query(self, user_input: str, session_id: Optional[str] = None, include_context: bool = True) -> str:
        """
        Process user query with conversation memory and generate response.

        Args:
            user_input: User's question or query
            session_id: Optional session ID (uses current if not provided)
            include_context: Whether to include conversation context

        Returns:
            Generated response from the AI system
        """
        try:
            # Use provided session_id or current one, or start new if none exists
            if session_id:
                self.current_session_id = session_id
            elif not self.current_session_id:
                self.current_session_id = self.start_conversation()

            # Add user message to conversation memory
            import time
            self.conversation_memory.add_message(
                self.current_session_id,
                ChatMessage(
                    role="user",
                    content=user_input,
                    timestamp=time.time(),
                    metadata={"source": "rag_pipeline"}
                )
            )

            # Get conversation context if enabled
            conversation_context = ""
            if include_context:
                conversation_context = self.conversation_memory.get_recent_context(
                    self.current_session_id,
                    max_messages=6  # Last 3 exchanges
                )

            # Process the query with context
            response = self._process_query_with_context(user_input, conversation_context)

            # Add assistant response to conversation memory
            self.conversation_memory.add_message(
                self.current_session_id,
                ChatMessage(
                    role="assistant",
                    content=response,
                    timestamp=time.time(),
                    metadata={"source": "rag_pipeline"}
                )
            )

            return response

        except Exception as e:
            logger.error(f"Error in RAG query processing: {e}")
            error_response = "I apologize, but I encountered an error processing your question. Could you please try rephrasing it?"

            # Still add to memory for debugging
            if self.current_session_id:
                self.conversation_memory.add_message(
                    self.current_session_id,
                    "assistant",
                    error_response,
                    metadata={"error": str(e), "timestamp": time.time()}
                )

            return error_response

    def _process_query_with_context(self, user_input: str, conversation_context: str) -> str:
        """Process query with conversation context and document retrieval."""
        try:
            self._ensure_models_loaded()

            # Step 1: Retrieve relevant documents
            logger.debug("Retrieving relevant documents...")
            retrieved_docs = self.retriever.retrieve(user_input, top_k=self.config.top_k_retrieve)

            # Step 2: Build context from retrieved documents
            context_parts = []
            for doc in retrieved_docs:
                context_parts.append(doc.get("content", str(doc)))

            document_context = "\n\n".join(context_parts) if context_parts else ""

            # Step 3: Construct the prompt with conversation and document context
            prompt = self._build_prompt(user_input, conversation_context, document_context)

            # Step 4: Generate response using IBM Granite model
            logger.debug("Generating response with IBM Granite model...")
            response = generate(
                self._generator,
                self._tokenizer,
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"

    def _build_prompt(self, user_input: str, conversation_context: str, document_context: str) -> str:
        """Build the final prompt for the IBM Granite model."""

        # Use the financial prompt template
        prompt_parts = [GRANITE_FINANCIAL_PROMPT]

        # Add document context if available
        if document_context.strip():
            prompt_parts.append("\nRelevant Financial Information:")
            prompt_parts.append(document_context)

        # Add conversation context if available
        if conversation_context.strip():
            prompt_parts.append("\nPrevious Conversation:")
            prompt_parts.append(conversation_context)

        # Add the current user question
        prompt_parts.append(f"\nUser Question: {user_input}")
        prompt_parts.append("\nAssistant Response:")

        full_prompt = "\n".join(prompt_parts)

        # Truncate if too long
        return truncate_context(full_prompt, self.config.max_context_chars)

    def get_conversation_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for the current or specified session.

        Args:
            session_id: Optional session ID (uses current if not provided)

        Returns:
            List of conversation messages
        """
        session = session_id or self.current_session_id
        if not session:
            return []

        messages = self.conversation_memory.get_conversation_history(session)
        return [{"role": msg.role, "content": msg.content, "timestamp": msg.timestamp} for msg in messages]

    def clear_conversation(self, session_id: Optional[str] = None) -> bool:
        """
        Clear conversation history.

        Args:
            session_id: Optional session ID (uses current if not provided)

        Returns:
            True if successful
        """
        session = session_id or self.current_session_id
        if not session:
            return False

        success = self.conversation_memory.clear_conversation(session)
        if success and session == self.current_session_id:
            self.current_session_id = None

        return success

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get conversation memory statistics."""
        return self.conversation_memory.get_memory_stats()

    def _ensure_models_loaded(self):
        """Ensure all models are loaded."""
        if self._generator is None or self._tokenizer is None:
            logger.info("Loading IBM Granite model...")
            self._generator, self._tokenizer = load_generator_from_config(self.config)

        if self._embedder is None:
            logger.info("Loading embedding model...")
            self._embedder = load_embedder(self.config.embedding_model_id)

    def cleanup(self):
        """Clean up resources."""
        try:
            cleanup_model_memory(self._generator, self._tokenizer)
            self._generator = None
            self._tokenizer = None
            self._embedder = None
            logger.info("RAG pipeline cleanup completed")
        except Exception as e:
            logger.warning(f"Error during RAG pipeline cleanup: {e}")


def create_rag_pipeline(config: Optional[Config] = None) -> RAGPipeline:
    """
    Factory function to create a configured RAG pipeline.

    Args:
        config: Configuration object

    Returns:
        Initialized RAG pipeline
    """
    from ..retrieval.dummy import DummyRetriever

    config = config or Config()
    retriever = DummyRetriever()

    return RAGPipeline(retriever, config)
