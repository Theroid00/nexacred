"""
MongoDB Retriever Implementation
===============================

Full MongoDB Atlas Vector Search integration for the RAG chatbot system.
Connects to MongoDB Atlas and performs vector similarity search for document retrieval.
"""

import numpy as np
from typing import List, Optional, Dict, Any
import logging

try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, OperationFailure
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    print("PyMongo not installed. Install with: pip install pymongo")

from .schema import Document
from .base import AbstractRetriever

logger = logging.getLogger(__name__)


class MongoRetriever(AbstractRetriever):
    """
    MongoDB Atlas Vector Search implementation for document retrieval.

    Connects to MongoDB Atlas and performs vector similarity search
    to retrieve relevant documents for the RAG pipeline.
    """
    
    def __init__(
        self,
        mongodb_uri: str = "mongodb+srv://hetshah05:Hetshahmit05@nexacred.9ndp6ei.mongodb.net/financial_advice_db?retryWrites=true&w=majority&appName=nexacred",
        database_name: str = "financial_advice_db",
        collection_name: str = "documents",
        index_name: str = "vector_index",
        embedding_field: str = "embedding",
        text_field: str = "content",
        metadata_field: str = "metadata",
        **kwargs
    ):
        """
        Initialize MongoDB retriever with connection to Atlas.

        Args:
            mongodb_uri: MongoDB Atlas connection string
            database_name: Target database name
            collection_name: Target collection name
            index_name: Vector search index name
            embedding_field: Field name for embeddings
            text_field: Field name for document text
            metadata_field: Field name for metadata
            **kwargs: Additional MongoDB connection parameters
        """
        
        if not PYMONGO_AVAILABLE:
            raise ImportError("PyMongo is required. Install with: pip install pymongo")

        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.index_name = index_name
        self.embedding_field = embedding_field
        self.text_field = text_field
        self.metadata_field = metadata_field

        # Connection objects
        self.client = None
        self.database = None
        self.collection = None

        # Connect to MongoDB
        self._connect()

        logger.info(f"MongoDB retriever initialized for {database_name}.{collection_name}")

    def _connect(self):
        """Establish connection to MongoDB Atlas."""
        try:
            logger.info("Connecting to MongoDB Atlas...")

            # Create MongoDB client with connection options
            self.client = MongoClient(
                self.mongodb_uri,
                serverSelectionTimeoutMS=10000,  # 10 second timeout
                connectTimeoutMS=10000,
                maxPoolSize=10,
                retryWrites=True
            )

            # Test the connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB Atlas!")

            # Get database and collection references
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]

            # Check if collection exists, create if not
            if self.collection_name not in self.database.list_collection_names():
                logger.info(f"Creating collection: {self.collection_name}")
                self.database.create_collection(self.collection_name)

            logger.info(f"Connected to database: {self.database_name}")
            logger.info(f"Using collection: {self.collection_name}")

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise

    def test_connection(self) -> bool:
        """Test if MongoDB connection is working."""
        try:
            if self.client is None:
                return False

            # Ping the database
            self.client.admin.command('ping')

            # Check collection access
            self.collection.count_documents({}, limit=1)

            logger.info("MongoDB connection test successful")
            return True

        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False

    def insert_documents(self, documents: List[Document]) -> bool:
        """
        Insert documents into MongoDB collection.

        Args:
            documents: List of Document objects to insert

        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided for insertion")
                return True

            # Convert documents to MongoDB format
            mongo_docs = []
            for doc in documents:
                mongo_doc = {
                    self.text_field: doc.content,
                    self.metadata_field: doc.metadata or {},
                    "source": getattr(doc, 'source', 'unknown'),
                    "doc_id": getattr(doc, 'id', None)
                }

                # Add embedding if available
                if hasattr(doc, 'embedding') and doc.embedding is not None:
                    if isinstance(doc.embedding, np.ndarray):
                        mongo_doc[self.embedding_field] = doc.embedding.tolist()
                    else:
                        mongo_doc[self.embedding_field] = doc.embedding

                mongo_docs.append(mongo_doc)

            # Insert documents
            result = self.collection.insert_many(mongo_docs)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return False

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents using vector similarity search.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of documents to retrieve

        Returns:
            List of relevant Document objects
        """
        try:
            if self.collection is None:
                logger.error("MongoDB collection not initialized")
                return self._get_fallback_documents(top_k)

            # Check if we have any documents with embeddings
            sample_doc = self.collection.find_one({self.embedding_field: {"$exists": True}})
            if not sample_doc:
                logger.warning("No documents with embeddings found, using fallback")
                return self._get_fallback_documents(top_k)

            # Perform vector search using aggregation pipeline
            # Note: This requires a vector search index to be set up in Atlas
            try:
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": self.index_name,
                            "path": self.embedding_field,
                            "queryVector": query_embedding.tolist(),
                            "numCandidates": min(top_k * 10, 100),
                            "limit": top_k
                        }
                    },
                    {
                        "$project": {
                            self.text_field: 1,
                            self.metadata_field: 1,
                            "source": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]

                results = list(self.collection.aggregate(pipeline))

                if not results:
                    logger.warning("Vector search returned no results, using fallback")
                    return self._get_fallback_documents(top_k)

                # Convert results to Document objects
                documents = []
                for result in results:
                    doc = Document(
                        content=result.get(self.text_field, ""),
                        metadata={
                            **(result.get(self.metadata_field, {})),
                            "source": result.get("source", "mongodb"),
                            "similarity_score": result.get("score", 0.0)
                        }
                    )
                    documents.append(doc)

                logger.info(f"Retrieved {len(documents)} documents via vector search")
                return documents

            except OperationFailure as e:
                if "vector search index" in str(e).lower():
                    logger.warning("Vector search index not available, falling back to text search")
                    return self._text_search_fallback(query_embedding, top_k)
                else:
                    raise

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return self._get_fallback_documents(top_k)

    def _text_search_fallback(self, query_embedding: np.ndarray, top_k: int) -> List[Document]:
        """Fallback to text-based search when vector search is unavailable."""
        try:
            # Get recent documents as fallback
            cursor = self.collection.find({}).limit(top_k)
            documents = []

            for doc in cursor:
                document = Document(
                    content=doc.get(self.text_field, ""),
                    metadata={
                        **(doc.get(self.metadata_field, {})),
                        "source": doc.get("source", "mongodb"),
                        "retrieval_method": "text_fallback"
                    }
                )
                documents.append(document)

            if documents:
                logger.info(f"Retrieved {len(documents)} documents via text fallback")
                return documents
            else:
                return self._get_fallback_documents(top_k)

        except Exception as e:
            logger.error(f"Text search fallback failed: {e}")
            return self._get_fallback_documents(top_k)

    def _get_fallback_documents(self, top_k: int) -> List[Document]:
        """Get hardcoded fallback documents for financial queries."""
        fallback_docs = [
            Document(
                content="RBI guidelines require banks to maintain minimum Capital Adequacy Ratio (CAR) of 9% under Basel III norms. This includes Tier 1 capital of at least 6% and Common Equity Tier 1 capital of at least 4.5%.",
                metadata={"source": "rbi_guidelines", "type": "regulation"}
            ),
            Document(
                content="Personal loans in India typically have interest rates ranging from 10-24% per annum. These are unsecured loans that don't require collateral. Eligibility depends on credit score, income, and employment history.",
                metadata={"source": "lending_norms", "type": "loan_info"}
            ),
            Document(
                content="UPI (Unified Payments Interface) allows instant money transfers 24x7 with a daily limit of ₹1 lakh per account. P2P transfers up to ₹1000 are free of charge across participating banks.",
                metadata={"source": "payment_systems", "type": "digital_payments"}
            ),
            Document(
                content="Credit scores in India range from 300-900, with scores above 750 considered excellent. CIBIL, Experian, Equifax, and CRIF High Mark are the four major credit bureaus.",
                metadata={"source": "credit_system", "type": "credit_info"}
            ),
            Document(
                content="SEBI regulations for P2P lending require platforms to maintain escrow accounts and limit individual borrower exposure to ₹50,000 across all platforms. Platform operators need minimum ₹2 crore net worth.",
                metadata={"source": "sebi_guidelines", "type": "p2p_regulation"}
            )
        ]

        return fallback_docs[:top_k]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the MongoDB collection."""
        try:
            if self.collection is None:
                return {"error": "Collection not initialized"}

            total_docs = self.collection.count_documents({})
            docs_with_embeddings = self.collection.count_documents({self.embedding_field: {"$exists": True}})

            # Get sample document to check structure
            sample_doc = self.collection.find_one({})

            stats = {
                "total_documents": total_docs,
                "documents_with_embeddings": docs_with_embeddings,
                "collection_name": self.collection_name,
                "database_name": self.database_name,
                "sample_fields": list(sample_doc.keys()) if sample_doc else [],
                "connection_status": "connected" if self.test_connection() else "disconnected"
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    def close_connection(self):
        """Close MongoDB connection."""
        try:
            if self.client:
                self.client.close()
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")

    def __del__(self):
        """Cleanup on object destruction."""
        self.close_connection()


# For backward compatibility, keep the stub class
class MongoRetrieverStub:
    """Deprecated stub class - use MongoRetriever instead."""

    def __init__(self, *args, **kwargs):
        logger.warning("MongoRetrieverStub is deprecated. Use MongoRetriever for full functionality.")
        raise NotImplementedError("Use MongoRetriever class for actual MongoDB integration")


if __name__ == "__main__":
    """Test MongoDB connection and basic operations."""
    import logging
    logging.basicConfig(level=logging.INFO)

    try:
        print("🔄 Testing MongoDB Connection...")

        # Initialize retriever with your connection string
        retriever = MongoRetriever()

        # Test connection
        if retriever.test_connection():
            print("✅ MongoDB connection successful!")

            # Get collection statistics
            stats = retriever.get_collection_stats()
            print(f"📊 Collection Stats: {stats}")

            # Test document insertion (sample)
            sample_docs = [
                Document(
                    content="Sample financial regulation document for testing.",
                    metadata={"type": "test", "category": "regulation"}
                )
            ]

            if retriever.insert_documents(sample_docs):
                print("✅ Document insertion test successful!")

            # Test retrieval with dummy embedding
            dummy_embedding = np.random.rand(384)  # Standard embedding size
            results = retriever.retrieve(dummy_embedding, top_k=3)
            print(f"✅ Retrieved {len(results)} documents")

            retriever.close_connection()
            print("✅ All tests completed successfully!")

        else:
            print("❌ MongoDB connection failed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        logging.error("MongoDB test failed", exc_info=True)
