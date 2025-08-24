"""
RAG Pipeline
============

Orchestrates the complete RAG (Retrieval-Augmented Generation) workflow:
1. Document retrieval based on query
2. Context construction with retrieved documents
3. Prompt formatting for Granite model
4. Text generation with retrieved context

This is the main orchestration component that ties together:
- Embedding models for query encoding
- Retrievers for document search
- Text generators for response creation
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from ..config import Config
from ..models.generator import load_generator_from_config, generate
from ..models.embeddings import load_embedder, embed_texts
from ..retrieval.base import AbstractRetriever
from ..prompts import GRANITE_FINANCIAL_PROMPT
from .token_utils import count_tokens, truncate_context

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for Indian financial regulation chatbot.
    
    Coordinates embedding, retrieval, prompt construction, and generation
    to provide contextual responses about financial regulations.
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
    
    def _ensure_models_loaded(self):
        """Lazy load models when first needed."""
        if self._generator is None:
            logger.info("Loading Granite generator model...")
            self._generator, self._tokenizer = load_generator_from_config(self.config)
            
        if self._embedder is None:
            logger.info("Loading embedding model...")
            self._embedder = load_embedder(self.config)
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate response for user query using RAG approach.
        
        Args:
            query: User's question about financial regulations
            
        Returns:
            Dictionary containing:
            - response: Generated answer
            - retrieved_docs: List of relevant documents used
            - metadata: Additional information about the process
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Ensure models are loaded
            self._ensure_models_loaded()
            
            # Step 1: Retrieve relevant documents
            logger.debug("Retrieving relevant documents...")
            retrieved_docs = self.retriever.retrieve(query, top_k=self.config.retrieval_top_k)
            
            if not retrieved_docs:
                logger.warning("No relevant documents found for query")
                return {
                    "response": "I couldn't find relevant information in the database. Please try rephrasing your question about Indian financial regulations.",
                    "retrieved_docs": [],
                    "metadata": {"status": "no_documents_found"}
                }
            
            # Step 2: Construct context from retrieved documents
            context = self._build_context(retrieved_docs)
            
            # Step 3: Create prompt with context and query
            prompt = self._build_prompt(query, context)
            
            # Step 4: Generate response
            logger.debug("Generating response...")
            response = generate(
                model=self._generator,
                tokenizer=self._tokenizer,
                prompt=prompt,
                max_length=self.config.max_output_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample
            )
            
            # Clean up the response (remove prompt echo if present)
            clean_response = self._clean_response(response, prompt)
            
            logger.info("Response generated successfully")
            
            return {
                "response": clean_response,
                "retrieved_docs": [{"content": doc.content, "metadata": doc.metadata} for doc in retrieved_docs],
                "metadata": {
                    "status": "success",
                    "num_retrieved_docs": len(retrieved_docs),
                    "context_length": len(context),
                    "prompt_length": len(prompt)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}", exc_info=True)
            return {
                "response": "I encountered an error while processing your question. Please try again.",
                "retrieved_docs": [],
                "metadata": {"status": "error", "error": str(e)}
            }
    
    def _build_context(self, documents: List[Any]) -> str:
        """
        Build context string from retrieved documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            doc_context = f"Document {i}:\n{doc.content.strip()}"
            if hasattr(doc, 'metadata') and doc.metadata:
                source_info = doc.metadata.get('source', 'Unknown')
                doc_context += f"\n(Source: {source_info})"
            context_parts.append(doc_context)
        
        full_context = "\n\n".join(context_parts)
        
        # Truncate context if too long
        max_context_tokens = self.config.max_input_length - 500  # Reserve space for query and prompt template
        truncated_context = truncate_context(full_context, max_context_tokens, self._tokenizer)
        
        return truncated_context
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build final prompt using template, context, and query.
        
        Args:
            query: User's question
            context: Retrieved context documents
            
        Returns:
            Formatted prompt string
        """
        return GRANITE_FINANCIAL_PROMPT.format(context=context, question=query)
    
    def _clean_response(self, response: str, prompt: str) -> str:
        """
        Clean generated response by removing prompt echo and formatting.
        
        Args:
            response: Raw generated text
            prompt: Original prompt
            
        Returns:
            Clean response text
        """
        # Remove prompt if the model echoed it
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Remove common generation artifacts
        response = response.strip()
        
        # Stop at common end-of-response markers
        stop_markers = ["\n\nQuestion:", "\n\nContext:", "<|endoftext|>", "</s>"]
        for marker in stop_markers:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        return response
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on pipeline components.
        
        Returns:
            Health status dictionary
        """
        status = {
            "pipeline": "healthy",
            "retriever": "unknown",
            "generator": "not_loaded",
            "embedder": "not_loaded"
        }
        
        try:
            # Check retriever
            if hasattr(self.retriever, 'health_check'):
                status["retriever"] = "healthy" if self.retriever.health_check() else "unhealthy"
            else:
                status["retriever"] = "no_health_check_method"
            
            # Check if models are loaded
            if self._generator is not None:
                status["generator"] = "loaded"
            if self._embedder is not None:
                status["embedder"] = "loaded"
                
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            status["pipeline"] = "unhealthy"
            status["error"] = str(e)
        
        return status
