"""
Abstract Retriever Interface
=============================

Protocol definition for document retrievers.
"""

import numpy as np
from typing import List, Protocol
from .schema import Document


class AbstractRetriever(Protocol):
    """
    Abstract interface for document retrievers.
    
    All retriever implementations must implement the retrieve method.
    """
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[Document]:
        """
        Retrieve relevant documents based on query embedding.
        
        Args:
            query_embedding: Query embedding vector (1D numpy array)
            top_k: Number of top documents to retrieve
            
        Returns:
            List of Document objects with similarity scores populated
        """
        ...
