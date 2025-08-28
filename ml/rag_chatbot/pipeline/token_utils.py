"""
Token Utilities
===============

Token counting and context truncation utilities for text processing.
Handles tokenization for context window management and prompt optimization.
"""

import logging
from typing import Optional, Any
import re

logger = logging.getLogger(__name__)


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """
    Count tokens in text using tokenizer or approximation.
    
    Args:
        text: Input text to count tokens for
        tokenizer: HuggingFace tokenizer (optional)
        
    Returns:
        Estimated token count
    """
    if tokenizer is not None:
        try:
            # Use actual tokenizer if available
            tokens = tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Tokenizer encoding failed: {e}, falling back to approximation")
    
    # Fallback: rough approximation (1 token ≈ 4 characters for most models)
    # This is a conservative estimate for English text
    return max(1, len(text) // 4)


def truncate_context(context: str, max_tokens: int, tokenizer: Optional[Any] = None) -> str:
    """
    Truncate context to fit within token limit.
    
    Tries to preserve complete sentences and document boundaries.
    
    Args:
        context: Context text to truncate
        max_tokens: Maximum allowed tokens
        tokenizer: HuggingFace tokenizer (optional)
        
    Returns:
        Truncated context string
    """
    if not context:
        return ""
    
    current_tokens = count_tokens(context, tokenizer)
    
    if current_tokens <= max_tokens:
        return context
    
    logger.info(f"Truncating context from {current_tokens} to {max_tokens} tokens")
    
    # Strategy 1: Try sentence-level truncation
    sentences = _split_into_sentences(context)
    truncated_sentences = []
    current_token_count = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, tokenizer)
        if current_token_count + sentence_tokens <= max_tokens:
            truncated_sentences.append(sentence)
            current_token_count += sentence_tokens
        else:
            break
    
    if truncated_sentences:
        return " ".join(truncated_sentences)
    
    # Strategy 2: Character-based truncation with safety margin
    # Use 80% of character limit to account for tokenization differences
    estimated_char_limit = int(max_tokens * 4 * 0.8)
    
    if len(context) > estimated_char_limit:
        truncated = context[:estimated_char_limit]
        # Try to end at a word boundary
        last_space = truncated.rfind(' ')
        if last_space > estimated_char_limit * 0.8:  # Don't cut too much
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    return context


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using simple heuristics.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting on common sentence endings
    # This is basic but should work for most financial regulation text
    sentence_endings = r'[.!?]+\s+'
    sentences = re.split(sentence_endings, text)
    
    # Clean up empty sentences and add back periods where needed
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Add period if sentence doesn't end with punctuation
            if not sentence[-1] in '.!?':
                sentence += '.'
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def estimate_tokens_from_chars(char_count: int) -> int:
    """
    Estimate token count from character count.
    
    Args:
        char_count: Number of characters
        
    Returns:
        Estimated token count
    """
    # Conservative estimate: 1 token ≈ 4 characters
    return max(1, char_count // 4)


def estimate_chars_from_tokens(token_count: int) -> int:
    """
    Estimate character count from token count.
    
    Args:
        token_count: Number of tokens
        
    Returns:
        Estimated character count
    """
    # Conservative estimate: 1 token ≈ 4 characters
    return token_count * 4


def validate_context_length(context: str, max_tokens: int, tokenizer: Optional[Any] = None) -> bool:
    """
    Check if context fits within token limit.
    
    Args:
        context: Context text
        max_tokens: Maximum allowed tokens
        tokenizer: HuggingFace tokenizer (optional)
        
    Returns:
        True if context fits, False otherwise
    """
    return count_tokens(context, tokenizer) <= max_tokens
