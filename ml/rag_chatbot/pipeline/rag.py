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
from ..conversation_memory import ConversationMemory, get_conversation_memory

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
            self.conversation_memory.add_message(
                self.current_session_id,
                "user",
                user_input,
                metadata={"timestamp": time.time(), "source": "rag_pipeline"}
            )

            # Get conversation context if enabled
            conversation_context = ""
            if include_context:
                conversation_context = self.conversation_memory.get_conversation_context(
                    self.current_session_id,
                    max_messages=6  # Last 3 exchanges
                )

            # Process the query with context
            response = self._process_query_with_context(user_input, conversation_context)

            # Add assistant response to conversation memory
            self.conversation_memory.add_message(
                self.current_session_id,
                "assistant",
                response,
                metadata={"timestamp": time.time(), "source": "rag_pipeline"}
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

            # Combine user input with conversation context for better retrieval
            search_query = user_input
            if conversation_context:
                # Extract recent context for search enhancement
                recent_context = conversation_context.split('\n')[-3:]  # Last few lines
                search_query = f"{user_input} {' '.join(recent_context)}"

            # Generate query embedding
            query_embedding = embed_texts([search_query], self._embedder)[0]

            # Retrieve relevant documents
            start_time = time.time()
            documents = self.retriever.retrieve(query_embedding, top_k=self.config.top_k_retrieve)
            retrieval_time = time.time() - start_time

            if not documents:
                logger.warning("No documents retrieved, generating response without context")
                return self._generate_fallback_response(user_input, conversation_context)

            # Construct context from retrieved documents
            document_context = self._construct_context(documents)

            # Build complete prompt with conversation history
            prompt = self._build_prompt_with_memory(
                user_input,
                document_context,
                conversation_context
            )

            # Generate response
            start_time = time.time()
            response = generate(
                model=self._generator,
                tokenizer=self._tokenizer,
                prompt=prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample
            )
            generation_time = time.time() - start_time

            logger.info(f"Query processed - Retrieval: {retrieval_time:.2f}s, Generation: {generation_time:.2f}s")

            return response.strip()

        except Exception as e:
            logger.error(f"Error in context processing: {e}")
            return self._generate_fallback_response(user_input, conversation_context)

    def _build_prompt_with_memory(self, user_input: str, document_context: str, conversation_context: str) -> str:
        """Build prompt that includes conversation memory and document context."""

        # Enhanced prompt template with conversation awareness
        prompt_template = """You are NexaCred AI, a helpful financial assistant specializing in Indian banking and finance regulations.

{conversation_context}

Context from knowledge base:
{document_context}

Current user question: {user_input}

Instructions:
- Provide accurate, helpful responses about Indian financial regulations, banking, loans, and credit systems
- If this is a follow-up question, acknowledge the previous conversation context
- Be conversational and refer to previous topics when relevant
- If you don't know something, say so clearly
- Keep responses focused and practical

Response:"""

        return prompt_template.format(
            conversation_context=conversation_context if conversation_context else "This is the start of our conversation.",
            document_context=document_context,
            user_input=user_input
        )

    def _generate_fallback_response(self, user_input: str, conversation_context: str) -> str:
        """Generate response when document retrieval fails, using conversation context."""
        try:
            self._ensure_models_loaded()

            # Simple prompt with just conversation context
            prompt = f"""You are NexaCred AI, a helpful financial assistant.

{conversation_context}

Current question: {user_input}

Provide a helpful response based on your knowledge of Indian finance and banking. If this relates to our previous conversation, acknowledge that context.

Response:"""

            response = generate(
                model=self._generator,
                tokenizer=self._tokenizer,
                prompt=prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample
            )

            return response.strip()

        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return "I apologize, but I'm having technical difficulties. Could you please try your question again?"

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
        """Lazy load models when first needed with retry logic."""
        if self._generator is None or self._embedder is None:
            try:
                if self._generator is None:
                    logger.info("Loading Granite generator model...")
                    self._generator, self._tokenizer = load_generator_from_config(self.config)

                if self._embedder is None:
                    logger.info("Loading embedding model...")
                    self._embedder = load_embedder(self.config.embedding_model_id)

                self._model_load_attempts = 0  # Reset on success

            except Exception as e:
                self._model_load_attempts += 1
                error_msg = f"Failed to load models (attempt {self._model_load_attempts}/{self._max_retries}): {e}"

                if self._model_load_attempts >= self._max_retries:
                    logger.error(f"Max retries exceeded for model loading: {e}")
                    raise RAGPipelineError(f"Could not load required models after {self._max_retries} attempts") from e
                else:
                    logger.warning(error_msg)
                    # Clean up partial state
                    cleanup_model_memory(self._generator, self._embedder)
                    self._generator = None
                    self._embedder = None
                    self._tokenizer = None
                    raise RAGPipelineError(error_msg) from e

    def cleanup_resources(self):
        """Clean up model resources to free memory."""
        try:
            cleanup_model_memory(self._generator, self._embedder)
            self._generator = None
            self._embedder = None
            self._tokenizer = None
            logger.info("Pipeline resources cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")

    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        try:
            self.cleanup_resources()
        except Exception:
            pass  # Ignore errors during destruction
