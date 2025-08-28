"""
Dummy Retriever Implementation
==============================

In-memory retriever with sample financial regulation documents.
"""

import numpy as np
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .schema import Document
from .base import AbstractRetriever

logger = logging.getLogger(__name__)


class DummyRetriever:
    """
    In-memory retriever with dummy financial regulation documents.
    
    Uses pre-computed embeddings and cosine similarity for retrieval.
    """
    
    def __init__(self, embedder: Any):
        """
        Initialize with embedding model and sample documents.
        
        Args:
            embedder: Loaded sentence transformer model
        """
        self.embedder = embedder
        self.documents = self._create_sample_documents()
        self.document_embeddings = self._precompute_embeddings()
        logger.info(f"DummyRetriever initialized with {len(self.documents)} documents")
    
    def _create_sample_documents(self) -> List[Document]:
        """Create sample financial regulation documents."""
        
        return [
            Document(
                id="rbi_car_2023",
                title="RBI Capital Adequacy Ratio Guidelines 2023",
                content="Banks must maintain a minimum Capital Adequacy Ratio (CAR) of 9% as per Basel III norms. This includes Tier 1 capital of at least 6% and Common Equity Tier 1 (CET1) capital of at least 4.5%. The capital adequacy framework ensures banks have sufficient capital to absorb unexpected losses and maintain financial stability.",
                metadata={"category": "banking", "regulator": "RBI", "year": "2023"}
            ),
            Document(
                id="sebi_mf_2024",
                title="SEBI Mutual Fund Investment Guidelines",
                content="Mutual funds must follow strict asset allocation norms with equity exposure limits based on fund category. Large cap funds must invest at least 80% in large cap stocks. Mid cap funds require 65% investment in mid cap stocks. Risk management includes daily NAV calculation, independent valuation, and regular stress testing.",
                metadata={"category": "securities", "regulator": "SEBI", "year": "2024"}
            ),
            Document(
                id="rbi_kyc_2024",
                title="RBI Know Your Customer (KYC) Compliance Framework",
                content="All banks must implement comprehensive KYC procedures including customer identification, verification, and ongoing monitoring. Enhanced due diligence is required for high-risk customers. Digital KYC using Aadhaar authentication is permitted with customer consent. Banks must report suspicious transactions to FIU-IND within prescribed timelines.",
                metadata={"category": "compliance", "regulator": "RBI", "year": "2024"}
            ),
            Document(
                id="rbi_digital_2024",
                title="RBI Digital Payment Security Guidelines",
                content="Payment service providers must implement multi-factor authentication for transactions above ₹5,000. Real-time fraud monitoring systems are mandatory with transaction velocity checks and behavioral analytics. Customer data must be stored only in India with end-to-end encryption. Regular security audits and penetration testing are required.",
                metadata={"category": "digital_payments", "regulator": "RBI", "year": "2024"}
            ),
            Document(
                id="sebi_p2p_2023",
                title="SEBI Peer-to-Peer Lending Platform Regulations",
                content="P2P lending platforms must maintain escrow accounts with scheduled commercial banks. Maximum exposure per borrower across all platforms is ₹50,000. Platform operators require minimum net worth of ₹2 crore and must conduct due diligence on borrowers. Interest rates and fees must be transparently disclosed to investors.",
                metadata={"category": "p2p_lending", "regulator": "SEBI", "year": "2023"}
            )
        ]
    
    def _precompute_embeddings(self) -> np.ndarray:
        """Precompute embeddings for all documents."""
        
        try:
            # Combine title and content for better retrieval
            texts = [f"{doc.title}: {doc.content}" for doc in self.documents]
            
            from ..models.embeddings import embed_texts
            embeddings = embed_texts(self.embedder, texts)
            
            logger.debug(f"Precomputed embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to precompute embeddings: {e}")
            # Return zero embeddings as fallback
            dim = getattr(self.embedder, 'get_sentence_embedding_dimension', lambda: 384)()
            return np.zeros((len(self.documents), dim), dtype=np.float32)
    
    def retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[Document]:
        """
        Retrieve documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents with similarity scores
        """
        
        try:
            if len(self.document_embeddings) == 0:
                logger.warning("No document embeddings available")
                return []
            
            # Ensure query embedding is 2D for cosine_similarity
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Compute cosine similarities
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Create result documents with scores
            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                # Create a copy with the similarity score
                result_doc = Document(
                    id=doc.id,
                    title=doc.title,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=float(similarities[idx])
                )
                results.append(result_doc)
            
            logger.debug(f"Retrieved {len(results)} documents, top score: {results[0].score:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
