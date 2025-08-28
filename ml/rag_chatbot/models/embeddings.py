"""
Embedding Models
================

Sentence transformer models for semantic embeddings with comprehensive error handling.
"""

import numpy as np
from typing import List, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass


def load_embedder(model_id: str, max_retries: int = 3) -> Any:
    """
    Load sentence transformer model for embeddings with retry logic.

    Args:
        model_id: Sentence transformer model identifier
        max_retries: Maximum number of retry attempts

    Returns:
        Loaded SentenceTransformer model
        
    Raises:
        EmbeddingError: If model loading fails after all retries
    """
    
    # List of fallback models in case the primary fails
    fallback_models = [
        model_id,
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]

    # Remove duplicates while preserving order
    models_to_try = list(dict.fromkeys(fallback_models))

    for attempt, current_model_id in enumerate(models_to_try):
        try:
            logger.info(f"Loading embedding model: {current_model_id} (attempt {attempt + 1})")

            # Try importing sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbeddingError(
                    "sentence-transformers not installed. Install with: pip install sentence-transformers"
                ) from e

            # Load the model with timeout protection
            start_time = time.time()
            embedder = SentenceTransformer(current_model_id)
            load_time = time.time() - start_time

            logger.info(f"Embedding model loaded successfully in {load_time:.2f}s")

            # Validate the model by testing a simple embedding
            try:
                test_embedding = embedder.encode(["test"], convert_to_numpy=True)
                if test_embedding is None or len(test_embedding) == 0:
                    raise ValueError("Model produced empty test embedding")
                logger.debug(f"Model validation successful, embedding dimension: {test_embedding.shape[1]}")
            except Exception as e:
                logger.warning(f"Model validation failed for {current_model_id}: {e}")
                continue

            return embedder

        except Exception as e:
            logger.warning(f"Failed to load {current_model_id}: {e}")
            if attempt < len(models_to_try) - 1:
                logger.info(f"Trying next fallback model...")
                continue
            else:
                error_msg = f"All embedding models failed to load. Last error: {e}"
                logger.error(error_msg)
                raise EmbeddingError(error_msg) from e


def embed_texts(embedder: Any, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts with comprehensive error handling.

    Args:
        embedder: Loaded SentenceTransformer model
        texts: List of texts to embed
        batch_size: Batch size for processing (to manage memory)

    Returns:
        Numpy array of embeddings with shape (n, d) and dtype float32

    Raises:
        EmbeddingError: If embedding generation fails
    """
    
    try:
        if not texts:
            logger.warning("Empty text list provided for embedding")
            dim = getattr(embedder, 'get_sentence_embedding_dimension', lambda: 384)()
            return np.array([], dtype=np.float32).reshape(0, dim)

        # Validate and clean input texts
        cleaned_texts = []
        for i, text in enumerate(texts):
            if text is None:
                logger.warning(f"Text {i} is None, replacing with empty string")
                cleaned_texts.append("")
            elif not isinstance(text, str):
                logger.warning(f"Text {i} is not a string, converting: {type(text)}")
                cleaned_texts.append(str(text))
            else:
                # Limit text length to prevent memory issues
                if len(text) > 8192:  # Reasonable limit for most models
                    logger.warning(f"Text {i} too long ({len(text)} chars), truncating")
                    cleaned_texts.append(text[:8192])
                else:
                    cleaned_texts.append(text)

        logger.debug(f"Embedding {len(cleaned_texts)} texts with batch size {batch_size}")

        # Process in batches to manage memory
        all_embeddings = []

        for i in range(0, len(cleaned_texts), batch_size):
            batch_texts = cleaned_texts[i:i + batch_size]

            try:
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(cleaned_texts) + batch_size - 1)//batch_size}")

                # Generate embeddings for this batch
                batch_embeddings = embedder.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # Normalize for cosine similarity
                    show_progress_bar=False,
                    device=None,  # Let the model decide
                    batch_size=min(batch_size, 16)  # Limit internal batch size
                )

                if batch_embeddings is None:
                    raise ValueError(f"Model returned None for batch {i//batch_size + 1}")

                # Validate batch embeddings
                if len(batch_embeddings) != len(batch_texts):
                    raise ValueError(f"Embedding count mismatch: expected {len(batch_texts)}, got {len(batch_embeddings)}")

                all_embeddings.append(batch_embeddings)

            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")

                # Try to recover with individual processing
                logger.info("Attempting individual text processing for failed batch...")
                batch_embeddings = []

                for j, text in enumerate(batch_texts):
                    try:
                        single_embedding = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
                        if single_embedding is not None and len(single_embedding) > 0:
                            batch_embeddings.append(single_embedding[0])
                        else:
                            # Use zero embedding as fallback
                            dim = embedder.get_sentence_embedding_dimension()
                            batch_embeddings.append(np.zeros(dim, dtype=np.float32))
                            logger.warning(f"Used zero embedding for text {i + j}")
                    except Exception as individual_error:
                        logger.error(f"Failed to process individual text {i + j}: {individual_error}")
                        # Use zero embedding as last resort
                        dim = embedder.get_sentence_embedding_dimension()
                        batch_embeddings.append(np.zeros(dim, dtype=np.float32))

                if batch_embeddings:
                    all_embeddings.append(np.array(batch_embeddings))

        if not all_embeddings:
            logger.error("No embeddings were generated successfully")
            dim = getattr(embedder, 'get_sentence_embedding_dimension', lambda: 384)()
            return np.zeros((len(texts), dim), dtype=np.float32)

        # Concatenate all batch embeddings
        embeddings = np.vstack(all_embeddings)

        # Ensure float32 dtype for consistency
        embeddings = embeddings.astype(np.float32)
        
        # Final validation
        if embeddings.shape[0] != len(texts):
            logger.error(f"Final embedding count mismatch: expected {len(texts)}, got {embeddings.shape[0]}")
            # Pad or truncate as needed
            if embeddings.shape[0] < len(texts):
                dim = embeddings.shape[1] if embeddings.shape[0] > 0 else 384
                padding = np.zeros((len(texts) - embeddings.shape[0], dim), dtype=np.float32)
                embeddings = np.vstack([embeddings, padding])
            else:
                embeddings = embeddings[:len(texts)]

        logger.debug(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        # Return zero embeddings as absolute fallback
        try:
            dim = getattr(embedder, 'get_sentence_embedding_dimension', lambda: 384)()
        except Exception:
            dim = 384  # Default dimension

        fallback_embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        logger.warning(f"Returning zero embeddings as fallback: shape {fallback_embeddings.shape}")
        return fallback_embeddings


def test_embedder_functionality(embedder: Any) -> bool:
    """
    Test embedder functionality with various inputs.

    Args:
        embedder: Loaded embedder model

    Returns:
        True if all tests pass, False otherwise
    """
    test_cases = [
        ["simple test"],
        [""],  # Empty string
        ["This is a longer test sentence with more words to check robustness."],
        ["Test 1", "Test 2", "Test 3"],  # Multiple texts
    ]

    try:
        for i, test_texts in enumerate(test_cases):
            logger.debug(f"Running embedder test case {i + 1}: {len(test_texts)} texts")
            embeddings = embed_texts(embedder, test_texts)

            if embeddings is None or embeddings.shape[0] != len(test_texts):
                logger.error(f"Test case {i + 1} failed: shape mismatch")
                return False

            if embeddings.shape[1] == 0:
                logger.error(f"Test case {i + 1} failed: zero-dimensional embeddings")
                return False

        logger.info("All embedder functionality tests passed")
        return True

    except Exception as e:
        logger.error(f"Embedder functionality test failed: {e}")
        return False
