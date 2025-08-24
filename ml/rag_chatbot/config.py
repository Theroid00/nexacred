"""
Configuration Management
========================

Pydantic-based configuration with environment variable support.
"""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    Configuration for IBM Granite 8B Instruct RAG Chatbot.
    
    All settings can be overridden via environment variables.
    Example: GRANITE_MODEL_ID=custom-model python -m rag_chatbot
    """
    
    # Model Configuration
    granite_model_id: str = Field(
        default="ibm-granite/granite-3.1-8b-instruct",
        description="Hugging Face model ID for IBM Granite"
    )
    
    embedding_model_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    # Generation Parameters
    max_new_tokens: int = Field(
        default=300,
        description="Maximum tokens to generate"
    )
    
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation"
    )
    
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling parameter"
    )
    
    # Retrieval Parameters
    top_k_retrieve: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of documents to retrieve"
    )
    
    # Hardware Configuration
    use_4bit: bool = Field(
        default=False,
        description="Enable 4-bit quantization (requires bitsandbytes)"
    )
    
    device_map: str = Field(
        default="auto",
        description="Device mapping strategy for model loading"
    )
    
    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI host"
    )
    
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="FastAPI port"
    )
    
    # Retriever Backend
    retriever_backend: Literal["dummy", "mongo_stub"] = Field(
        default="dummy",
        description="Retriever implementation to use"
    )
    
    # Context Management
    max_context_chars: int = Field(
        default=5000,
        description="Maximum context length in characters"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global config instance
config = Config()
