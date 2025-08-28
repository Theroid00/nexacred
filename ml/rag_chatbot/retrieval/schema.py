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

    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        """Create document from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            content=data["content"],
            metadata=data.get("metadata"),
            score=data.get("score")
        )

    def __str__(self) -> str:
        """String representation of document."""
        score_str = f" (score: {self.score:.4f})" if self.score is not None else ""
        return f"Document[{self.id}]: {self.title}{score_str}"

    def get_snippet(self, max_length: int = 200) -> str:
        """Get a snippet of the document content."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length].rsplit(' ', 1)[0] + "..."
