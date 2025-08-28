"""
FastAPI Application
===================

RESTful API service for the Indian Financial Regulation RAG Chatbot.

Endpoints:
- GET /health: Health check
- POST /infer: Generate response to user query
- GET /docs: Swagger UI documentation

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# FastAPI imports with fallback
try:
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    logging.warning("FastAPI not available. Install with: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False
    
    # Create dummy classes for type hints
    class FastAPI:
        pass
    class BaseModel:
        pass
    def Field(**kwargs):
        return None

from ..config import Config
from ..pipeline.rag import RAGPipeline
from ..retrieval.dummy import DummyRetriever
from ..retrieval.mongo_stub import MongoRetrieverStub

logger = logging.getLogger(__name__)

# Global pipeline instance
_pipeline: Optional[RAGPipeline] = None
_config: Optional[Config] = None


class QueryRequest(BaseModel):
    """Request model for query inference."""
    query: str = Field(..., description="User question about financial regulations", min_length=1, max_length=1000)
    use_dummy_retriever: bool = Field(default=True, description="Use dummy retriever instead of MongoDB")


class QueryResponse(BaseModel):
    """Response model for query inference."""
    response: str = Field(..., description="Generated answer")
    retrieved_docs: list = Field(default_factory=list, description="Retrieved documents used for context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Overall service status")
    components: Dict[str, Any] = Field(default_factory=dict, description="Component health status")


if FASTAPI_AVAILABLE:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        # Startup
        logger.info("Starting Indian Financial Regulation RAG Chatbot API...")
        await startup_event()
        yield
        # Shutdown
        logger.info("Shutting down RAG Chatbot API...")
        await shutdown_event()

    app = FastAPI(
        title="Indian Financial Regulation RAG Chatbot",
        description="AI-powered chatbot for Indian financial regulations using IBM Granite 3.0 8B with RAG",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

else:
    # Dummy app for when FastAPI is not available
    class DummyApp:
        def __init__(self):
            logger.error("FastAPI not available. Cannot create web service.")
    
    app = DummyApp()


async def startup_event():
    """Initialize application on startup."""
    global _pipeline, _config
    
    try:
        logger.info("Initializing RAG pipeline...")
        _config = Config()
        
        # Load embedder first for DummyRetriever
        from ..models.embeddings import load_embedder
        from ..retrieval.dummy import DummyRetriever

        embedder = load_embedder(_config.embedding_model_id)
        retriever = DummyRetriever(embedder)
        _pipeline = RAGPipeline(retriever, _config)
        
        # Perform health check
        health_status = _pipeline.health_check()
        logger.info(f"Pipeline initialized. Health status: {health_status}")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}", exc_info=True)
        raise


async def shutdown_event():
    """Cleanup on application shutdown."""
    global _pipeline
    logger.info("Cleaning up resources...")
    _pipeline = None


def get_pipeline() -> RAGPipeline:
    """Dependency to get pipeline instance."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


if FASTAPI_AVAILABLE:
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Health check endpoint.
        
        Returns:
            Health status of all components
        """
        try:
            if _pipeline is None:
                return HealthResponse(
                    status="unhealthy",
                    components={"pipeline": "not_initialized"}
                )
            
            health_status = _pipeline.health_check()
            overall_status = "healthy" if health_status.get("pipeline") == "healthy" else "unhealthy"
            
            return HealthResponse(
                status=overall_status,
                components=health_status
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return HealthResponse(
                status="unhealthy",
                components={"error": str(e)}
            )

    @app.post("/infer", response_model=QueryResponse)
    async def generate_response(
        request: QueryRequest,
        pipeline: RAGPipeline = Depends(get_pipeline)
    ):
        """
        Generate response to user query.
        
        Args:
            request: Query request with user question
            pipeline: RAG pipeline instance
            
        Returns:
            Generated response with context
        """
        try:
            logger.info(f"Processing query: {request.query[:100]}...")
            
            # Switch retriever if requested
            if not request.use_dummy_retriever:
                # Note: MongoDB stub will raise NotImplementedError
                mongo_retriever = MongoRetrieverStub(_config)
                pipeline.retriever = mongo_retriever
            
            result = pipeline.generate_response(request.query)
            
            return QueryResponse(
                response=result["response"],
                retrieved_docs=result.get("retrieved_docs", []),
                metadata=result.get("metadata", {})
            )
            
        except NotImplementedError:
            logger.warning("MongoDB retriever not implemented, falling back to dummy")
            raise HTTPException(
                status_code=501,
                detail="MongoDB retriever not yet implemented. Use dummy retriever instead."
            )
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    @app.get("/")
    async def root():
        """Root endpoint with basic information."""
        return {
            "message": "Indian Financial Regulation RAG Chatbot API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "infer": "/infer"
        }

    @app.get("/info")
    async def get_info():
        """Get system information."""
        return {
            "model": _config.granite_model_id if _config else "Not loaded",
            "embedding_model": _config.embedding_model_id if _config else "Not loaded",
            "retriever": "DummyRetriever" if _pipeline else "Not initialized",
            "status": "ready" if _pipeline else "initializing"
        }


def setup_routes():
    """Setup additional routes if needed."""
    # Additional route configuration can be added here
    pass


if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI not available. Install with: pip install fastapi uvicorn")
        exit(1)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
