"""
Configuration Management
========================

Streamlined configuration for the NexaCred RAG Chatbot.
"""

from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    Configuration for NexaCred RAG Chatbot with IBM Granite 3.3 2B Instruct.
    All settings can be overridden via environment variables.
    """
    
    # Model Configuration
    granite_model_id: str = Field(
        default="ibm-granite/granite-3.3-2b-instruct",
        description="IBM Granite 3.3 2B Instruct model from Hugging Face"
    )
    
    embedding_model_id: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    # Generation Parameters
    max_new_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")

    # Retrieval Parameters
    top_k_retrieve: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")

    # Hardware Configuration
    use_4bit: bool = Field(default=True, description="Enable 4-bit quantization")
    device_map: str = Field(default="auto", description="Device mapping strategy")
    max_memory_gb: float = Field(default=6.0, description="Maximum memory usage in GB")
    enable_gpu: bool = Field(default=True, description="Enable GPU usage if available")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="FastAPI port")

    # Context Management
    max_context_chars: int = Field(default=6000, description="Maximum context length in characters")
    max_input_length: int = Field(default=2048, description="Maximum input length in tokens")
    do_sample: bool = Field(default=True, description="Whether to use sampling during generation")

    # MongoDB Configuration (Optional)
    mongodb_uri: str = Field(
        default="mongodb+srv://hetshah05:Hetshahmit05@nexacred.9ndp6ei.mongodb.net/financial_advice_db?retryWrites=true&w=majority&appName=nexacred",
        description="MongoDB Atlas connection string"
    )
    mongodb_database: str = Field(default="financial_advice_db", description="MongoDB database name")
    mongodb_collection: str = Field(default="documents", description="MongoDB collection name")
    mongodb_index_name: str = Field(default="vector_index", description="MongoDB vector search index name")
    use_mongodb: bool = Field(default=False, description="Enable MongoDB Atlas for document retrieval")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global config instance
config = Config()
