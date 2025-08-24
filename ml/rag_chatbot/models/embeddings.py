"""
Embedding Models
================

Sentence transformer models for semantic embeddings.
"""

import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


def load_embedder(model_id: str) -> Any:
    """
    Load sentence transformer model for embeddings.
    
    Args:
        model_id: Sentence transformer model identifier
        
    Returns:
        Loaded SentenceTransformer model
        
    Raises:
        RuntimeError: If model loading fails
    """
    
    try:
        logger.info(f"Loading embedding model: {model_id}")
        embedder = SentenceTransformer(model_id)
        logger.info("Embedding model loaded successfully")
        return embedder
        
    except Exception as e:
        error_msg = f"Failed to load embedding model '{model_id}': {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def embed_texts(embedder: Any, texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        embedder: Loaded SentenceTransformer model
        texts: List of texts to embed
        
    Returns:
        Numpy array of embeddings with shape (n, d) and dtype float32
    """
    
    try:
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return np.array([], dtype=np.float32).reshape(0, embedder.get_sentence_embedding_dimension())
        
        logger.debug(f"Embedding {len(texts)} texts")
        
        # Generate embeddings
        embeddings = embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False
        )
        
        # Ensure float32 dtype
        embeddings = embeddings.astype(np.float32)
        
        logger.debug(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        # Return zero embeddings as fallback
        dim = getattr(embedder, 'get_sentence_embedding_dimension', lambda: 384)()
        return np.zeros((len(texts), dim), dtype=np.float32)
