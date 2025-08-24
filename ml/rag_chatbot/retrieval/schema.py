"""
Document Schema
===============

Dataclass definition for regulation documents.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Document:
    """
    Represents a regulation document with metadata and similarity score.
    
    Attributes:
        id: Unique document identifier
        title: Document title/heading
        content: Full document content
        metadata: Optional metadata dictionary
        score: Similarity score (populated during retrieval)
    """
    
    id: str
    title: str
    content: str
    metadata: Optional[Dict[str, str]] = None
    score: Optional[float] = None
    
    def __post_init__(self):
        """Validate document after initialization."""
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.title:
            raise ValueError("Document title cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")
    
    def to_dict(self) -> Dict:
        """Convert document to dictionary format."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata or {},
            "score": self.score
        }
