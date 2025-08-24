"""
IBM Granite 8B Instruct RAG Chatbot
===================================

Production-ready RAG implementation for financial compliance and credit regulations.
"""

__version__ = "1.0.0"
__author__ = "NexaCred Team"

from .config import Config
from .prompts import build_prompt

__all__ = ["Config", "build_prompt"]
