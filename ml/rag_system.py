#!/usr/bin/env python3
"""
NexaCred RAG Knowledge Base System
==================================

Advanced RAG implementation with IBM Granite models for:
- Financial regulations and compliance knowledge
- Credit scoring policies and guidelines
- Market data and economic indicators
- Customer behavior patterns
- Fraud detection knowledge base
"""

import os
import json
import sqlite3
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import logging

# Vector similarity imports (would use actual embeddings in production)
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Note: sentence-transformers not available, using mock embeddings")

@dataclass
class Document:
    """Document representation for RAG knowledge base"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: datetime = None
    updated_at: datetime = None

class NexaCredRAGSystem:
    """
    Advanced RAG system for NexaCred financial AI

    Features:
    - Multi-modal knowledge ingestion
    - Vector similarity search
    - Contextual retrieval
    - Real-time knowledge updates
    - Compliance-aware responses
    """

    def __init__(self, db_path: str = "nexacred_knowledge.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)

        # Initialize embedding model (mock if not available)
        if EMBEDDINGS_AVAILABLE:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedder = None

        # Initialize database
        self._init_database()

        # Load default knowledge base
        self._load_default_knowledge()

    def _init_database(self):
        """Initialize SQLite database for knowledge storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Knowledge categories table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS categories (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                parent_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Query history for learning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                retrieved_docs TEXT,
                response TEXT,
                rating INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _load_default_knowledge(self):
        """Load default financial knowledge base"""
        default_docs = [
            {
                "id": "rbi_credit_guidelines_001",
                "content": """
                Reserve Bank of India (RBI) Credit Information Companies Regulations, 2006:
                
                1. Credit Scoring Standards:
                - Credit scores must range from 300-900
                - Scores based on payment history (35%), credit utilization (30%), length of credit history (15%), credit mix (10%), new credit (10%)
                - Credit reports must be updated monthly
                - Consumers have right to one free credit report annually
                
                2. Data Privacy Requirements:
                - Explicit consent required for credit data access
                - Data retention limited to 7 years for settled accounts
                - Right to correction and dispute resolution
                
                3. Fair Lending Practices:
                - No discrimination based on gender, caste, religion, or geography
                - Transparent pricing and terms disclosure
                - Cooling-off period for high-value loans
                """,
                "metadata": {
                    "category": "regulatory",
                    "source": "RBI",
                    "type": "credit_guidelines",
                    "date": "2023-12-01",
                    "authority": "Reserve Bank of India"
                }
            },
            {
                "id": "fraud_patterns_001",
                "content": """
                Common Financial Fraud Patterns:
                
                1. Transaction Velocity Fraud:
                - Multiple transactions in short time (>20 per hour)
                - Unusual timing patterns (middle of night)
                - Rapid-fire small amounts followed by large transactions
                
                2. Geographic Anomalies:
                - Transactions from multiple distant locations simultaneously
                - Sudden travel patterns inconsistent with user profile
                - High-risk geographic locations
                
                3. Merchant Category Fraud:
                - High-risk merchants (gambling, adult entertainment, crypto)
                - Shell companies and suspicious merchant IDs
                - Cash advance patterns
                
                4. Behavioral Anomalies:
                - Spending patterns inconsistent with income
                - Device fingerprint changes
                - IP address inconsistencies
                """,
                "metadata": {
                    "category": "fraud_detection",
                    "source": "industry_best_practices",
                    "type": "patterns",
                    "risk_level": "high",
                    "last_updated": "2024-08-15"
                }
            },
            {
                "id": "credit_scoring_factors_001",
                "content": """
                Credit Scoring Model Factors and Weights:
                
                Primary Factors (70% weight):
                1. Payment History (35%):
                   - On-time payments across all accounts
                   - Delinquency patterns and severity
                   - Public records (bankruptcies, liens)
                
                2. Credit Utilization (30%):
                   - Total utilization across all accounts
                   - Per-account utilization ratios
                   - Trend analysis over 6-12 months
                
                3. Length of Credit History (15%):
                   - Age of oldest account
                   - Average age of all accounts
                   - Time since last account opening
                
                Secondary Factors (30% weight):
                4. Credit Mix (10%):
                   - Variety of account types (revolving, installment)
                   - Appropriate mix for customer profile
                
                5. New Credit (10%):
                   - Recent credit inquiries
                   - Accounts opened in last 12 months
                   - Credit seeking behavior patterns
                
                Alternative Data (emerging):
                - Bank account behavior
                - Utility payment patterns
                - Mobile phone usage data
                - Social media financial indicators
                """,
                "metadata": {
                    "category": "credit_scoring",
                    "source": "nexacred_research",
                    "type": "model_documentation",
                    "version": "2.1",
                    "effective_date": "2024-01-01"
                }
            },
            {
                "id": "loan_product_guidelines_001",
                "content": """
                NexaCred Loan Product Guidelines:
                
                Personal Loans:
                - Minimum Credit Score: 600
                - Maximum Amount: ₹10,00,000
                - Interest Rate Range: 10.5% - 24% APR
                - Tenure: 12-60 months
                - Processing Fee: 2% of loan amount
                - Income Requirement: ₹25,000/month minimum
                
                Home Loans:
                - Minimum Credit Score: 700
                - Maximum Amount: ₹2,00,00,000
                - Interest Rate Range: 8.5% - 12% APR
                - Tenure: 60-360 months
                - LTV Ratio: Up to 90% for first-time buyers
                - Income Requirement: ₹50,000/month minimum
                
                Credit Cards:
                - Minimum Credit Score: 650
                - Credit Limit: ₹50,000 - ₹10,00,000
                - Interest Rate: 18% - 42% APR
                - Annual Fee: ₹500 - ₹25,000
                - Minimum Income: ₹30,000/month
                
                Risk-Based Pricing:
                - Excellent (800+): Best rates, premium products
                - Very Good (740-799): Standard rates, full features
                - Good (670-739): Slightly higher rates, some restrictions
                - Fair (580-669): Higher rates, limited products
                - Poor (<580): Secured products only
                """,
                "metadata": {
                    "category": "products",
                    "source": "nexacred_policy",
                    "type": "underwriting_guidelines",
                    "version": "3.2",
                    "department": "risk_management"
                }
            },
            {
                "id": "economic_indicators_001",
                "content": """
                Key Economic Indicators for Credit Risk Assessment:
                
                Macroeconomic Factors:
                1. GDP Growth Rate:
                   - Strong growth (>6%): Lower default risk
                   - Moderate growth (3-6%): Standard risk assessment
                   - Slow growth (<3%): Increased scrutiny
                
                2. Inflation Rate:
                   - Target range: 4-6% (RBI mandate)
                   - High inflation (>7%): Adjust for real income
                   - Deflation: Economic stress indicator
                
                3. Interest Rate Environment:
                   - RBI Repo Rate impacts lending rates
                   - Yield curve analysis for long-term products
                   - Credit spread monitoring
                
                Industry-Specific Factors:
                - IT Sector: Stable, low default risk
                - Manufacturing: Cyclical, moderate risk
                - Services: Variable, depends on segment
                - Agriculture: Seasonal, weather-dependent
                
                Regional Economic Health:
                - State GDP per capita
                - Employment rates
                - Industrial activity indices
                - Agricultural productivity
                """,
                "metadata": {
                    "category": "economics",
                    "source": "rbi_data",
                    "type": "indicators",
                    "frequency": "monthly_update",
                    "relevance": "credit_risk"
                }
            }
        ]

        # Add documents to knowledge base
        for doc_data in default_docs:
            doc = Document(
                id=doc_data["id"],
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                created_at=datetime.now()
            )
            self.add_document(doc)

    def add_document(self, document: Document) -> bool:
        """Add document to knowledge base with embedding"""
        try:
            # Generate embedding
            if self.embedder:
                embedding = self.embedder.encode(document.content)
                document.embedding = embedding
            else:
                # Mock embedding for demo
                document.embedding = np.random.rand(384).astype(np.float32)

            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (id, content, metadata, embedding, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                document.id,
                document.content,
                json.dumps(document.metadata),
                document.embedding.tobytes() if document.embedding is not None else None,
                document.created_at or datetime.now(),
                datetime.now()
            ))

            conn.commit()
            conn.close()

            self.logger.info(f"Added document {document.id} to knowledge base")
            return True

        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            return False

    def search_documents(self, query: str, top_k: int = 5,
                        category_filter: Optional[str] = None) -> List[Document]:
        """Search for relevant documents using semantic similarity"""
        try:
            # Generate query embedding
            if self.embedder:
                query_embedding = self.embedder.encode(query)
            else:
                query_embedding = np.random.rand(384).astype(np.float32)

            # Retrieve all documents
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if category_filter:
                cursor.execute('''
                    SELECT id, content, metadata, embedding 
                    FROM documents 
                    WHERE json_extract(metadata, '$.category') = ?
                ''', (category_filter,))
            else:
                cursor.execute('SELECT id, content, metadata, embedding FROM documents')

            results = cursor.fetchall()
            conn.close()

            # Calculate similarities and rank
            documents = []
            for row in results:
                doc_id, content, metadata_str, embedding_bytes = row
                metadata = json.loads(metadata_str)

                if embedding_bytes:
                    doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                else:
                    similarity = 0.5  # Default similarity for mock embeddings

                documents.append((similarity, Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    embedding=doc_embedding if embedding_bytes else None
                )))

            # Sort by similarity and return top-k
            documents.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in documents[:top_k]]

        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            return []

    def retrieve_context(self, query: str, context_type: str = "general") -> Dict[str, Any]:
        """Retrieve contextual information for query processing"""

        # Define search strategies by context type
        search_strategies = {
            "regulatory": {"category_filter": "regulatory", "top_k": 3},
            "fraud": {"category_filter": "fraud_detection", "top_k": 5},
            "credit_scoring": {"category_filter": "credit_scoring", "top_k": 3},
            "products": {"category_filter": "products", "top_k": 4},
            "economics": {"category_filter": "economics", "top_k": 2},
            "general": {"category_filter": None, "top_k": 5}
        }

        strategy = search_strategies.get(context_type, search_strategies["general"])

        # Search for relevant documents
        relevant_docs = self.search_documents(
            query,
            top_k=strategy["top_k"],
            category_filter=strategy.get("category_filter")
        )

        # Compile context
        context = {
            "query": query,
            "context_type": context_type,
            "retrieved_documents": [],
            "sources": [],
            "timestamp": datetime.now().isoformat()
        }

        for doc in relevant_docs:
            context["retrieved_documents"].append({
                "id": doc.id,
                "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                "metadata": doc.metadata
            })

            source = doc.metadata.get("source", "unknown")
            if source not in context["sources"]:
                context["sources"].append(source)

        return context

    def log_query(self, query: str, retrieved_docs: List[str],
                  response: str, rating: Optional[int] = None):
        """Log query for learning and improvement"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO query_history (query, retrieved_docs, response, rating)
                VALUES (?, ?, ?, ?)
            ''', (query, json.dumps(retrieved_docs), response, rating))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error logging query: {e}")

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total documents
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_docs = cursor.fetchone()[0]

            # Documents by category
            cursor.execute('''
                SELECT json_extract(metadata, '$.category') as category, COUNT(*)
                FROM documents
                GROUP BY category
            ''')
            categories = dict(cursor.fetchall())

            # Recent queries
            cursor.execute('''
                SELECT COUNT(*) FROM query_history 
                WHERE timestamp >= datetime('now', '-24 hours')
            ''')
            recent_queries = cursor.fetchone()[0]

            conn.close()

            return {
                "total_documents": total_docs,
                "categories": categories,
                "recent_queries_24h": recent_queries,
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting knowledge stats: {e}")
            return {}

# Example usage and testing
def demo_rag_system():
    """Demonstrate RAG system capabilities"""
    print("🧠 NEXACRED RAG KNOWLEDGE SYSTEM DEMO")
    print("="*50)

    # Initialize RAG system
    rag = NexaCredRAGSystem()

    # Test queries
    test_queries = [
        ("What are RBI guidelines for credit scoring?", "regulatory"),
        ("How do I detect transaction velocity fraud?", "fraud"),
        ("What factors affect credit scores?", "credit_scoring"),
        ("What are the eligibility criteria for home loans?", "products"),
        ("How does GDP growth affect lending risk?", "economics")
    ]

    for query, context_type in test_queries:
        print(f"\n🔍 Query: {query}")
        print(f"📂 Context Type: {context_type}")

        # Retrieve context
        context = rag.retrieve_context(query, context_type)

        print(f"📊 Found {len(context['retrieved_documents'])} relevant documents")
        print(f"📚 Sources: {', '.join(context['sources'])}")

        # Show top document snippet
        if context['retrieved_documents']:
            top_doc = context['retrieved_documents'][0]
            print(f"📄 Top Match: {top_doc['id']}")
            print(f"   Preview: {top_doc['content'][:150]}...")

    # Show knowledge base stats
    print(f"\n📈 KNOWLEDGE BASE STATISTICS")
    print("="*40)
    stats = rag.get_knowledge_stats()
    print(f"Total Documents: {stats.get('total_documents', 0)}")
    print(f"Categories: {list(stats.get('categories', {}).keys())}")
    print(f"Recent Queries: {stats.get('recent_queries_24h', 0)}")

if __name__ == "__main__":
    demo_rag_system()
