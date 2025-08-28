"""
Text Chunking Utilities
========================

Document chunking and text segmentation for efficient processing.
Handles splitting large documents into manageable chunks for embedding and retrieval.
"""

import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    preserve_sentences: bool = True
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        preserve_sentences: Try to preserve sentence boundaries
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    if preserve_sentences:
        chunks = _chunk_by_sentences(text, chunk_size, chunk_overlap)
    else:
        chunks = _chunk_by_characters(text, chunk_size, chunk_overlap)
    
    # Filter out very small chunks (less than 10% of target size)
    min_chunk_size = max(10, chunk_size // 10)
    filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= min_chunk_size]
    
    logger.debug(f"Split text into {len(filtered_chunks)} chunks (target size: {chunk_size})")
    
    return filtered_chunks


def _chunk_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Chunk text while preserving sentence boundaries.
    
    Args:
        text: Input text
        chunk_size: Target chunk size
        chunk_overlap: Overlap size
        
    Returns:
        List of text chunks
    """
    # Split into sentences
    sentences = _split_into_sentences(text)
    
    if not sentences:
        return _chunk_by_characters(text, chunk_size, chunk_overlap)
    
    chunks = []
    current_chunk = ""
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        
        # If adding this sentence would exceed chunk size, finalize current chunk
        if current_chunk and len(current_chunk + " " + sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            current_chunk = _get_overlap_text(current_chunk, chunk_overlap)
            
        # Add sentence to current chunk
        if current_chunk:
            current_chunk += " " + sentence
        else:
            current_chunk = sentence
        
        i += 1
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def _chunk_by_characters(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Chunk text by character count with overlap.
    
    Args:
        text: Input text
        chunk_size: Target chunk size
        chunk_overlap: Overlap size
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            # Final chunk
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        
        # Try to end at word boundary
        chunk_text = text[start:end]
        last_space = chunk_text.rfind(' ')
        
        if last_space > chunk_size * 0.7:  # Only use word boundary if not too far back
            end = start + last_space
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position (with overlap)
        start = end - chunk_overlap
        
        # Ensure we make progress
        if start <= 0:
            start = end
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Enhanced sentence splitting for financial/legal text
    # Handle common abbreviations and numbering
    
    # Common abbreviations in financial text
    abbreviations = [
        'vs.', 'Inc.', 'Ltd.', 'Corp.', 'Co.', 'LLC', 'LLP',
        'Mr.', 'Mrs.', 'Dr.', 'Prof.', 'Sr.', 'Jr.',
        'U.S.', 'U.K.', 'E.U.', 'etc.', 'i.e.', 'e.g.',
        'RBI', 'SEBI', 'IRDAI', 'NPCI'
    ]
    
    # Protect abbreviations
    protected_text = text
    placeholder_map = {}
    
    for i, abbrev in enumerate(abbreviations):
        placeholder = f"__ABBREV_{i}__"
        protected_text = protected_text.replace(abbrev, placeholder)
        placeholder_map[placeholder] = abbrev
    
    # Split on sentence endings
    sentence_pattern = r'[.!?]+(?:\s+|$)'
    sentences = re.split(sentence_pattern, protected_text)
    
    # Restore abbreviations and clean up
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Restore abbreviations
            for placeholder, abbrev in placeholder_map.items():
                sentence = sentence.replace(placeholder, abbrev)
            
            # Ensure sentence ends with punctuation
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def _get_overlap_text(chunk: str, overlap_size: int) -> str:
    """
    Get overlap text from the end of a chunk.
    
    Args:
        chunk: Source chunk
        overlap_size: Size of overlap
        
    Returns:
        Overlap text
    """
    if len(chunk) <= overlap_size:
        return chunk
    
    overlap_text = chunk[-overlap_size:]
    
    # Try to start at word boundary
    first_space = overlap_text.find(' ')
    if first_space > 0:
        overlap_text = overlap_text[first_space:].strip()
    
    return overlap_text


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    content_key: str = "content"
) -> List[Dict[str, Any]]:
    """
    Chunk a list of documents.
    
    Args:
        documents: List of document dictionaries
        chunk_size: Target chunk size
        chunk_overlap: Overlap size
        content_key: Key containing document content
        
    Returns:
        List of chunked documents with metadata
    """
    chunked_docs = []
    
    for doc_idx, doc in enumerate(documents):
        content = doc.get(content_key, "")
        
        if not content:
            logger.warning(f"Document {doc_idx} has no content")
            continue
        
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        
        for chunk_idx, chunk in enumerate(chunks):
            chunked_doc = doc.copy()  # Preserve original metadata
            chunked_doc[content_key] = chunk
            
            # Add chunk metadata
            chunked_doc.update({
                "original_doc_id": doc.get("id", doc_idx),
                "chunk_id": f"{doc_idx}_{chunk_idx}",
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            })
            
            chunked_docs.append(chunked_doc)
    
    logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
    
    return chunked_docs


def estimate_chunks(text: str, chunk_size: int) -> int:
    """
    Estimate number of chunks for given text and chunk size.
    
    Args:
        text: Input text
        chunk_size: Target chunk size
        
    Returns:
        Estimated number of chunks
    """
    if not text:
        return 0
    
    return max(1, (len(text) + chunk_size - 1) // chunk_size)  # Ceiling division
