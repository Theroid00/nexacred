"""
Configuration Management
========================

Streamlined configuration for the NexaCred RAG Chatbot.
"""

from typing import List
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

    granite_model_fallbacks: List[str] = Field(
        default=["microsoft/DialoGPT-medium", "distilgpt2"],
        description="Fallback models if primary model fails"
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

    class Config:
        """Pydantic configuration."""
        env_prefix = "NEXACRED_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global config instance
config = Config()
