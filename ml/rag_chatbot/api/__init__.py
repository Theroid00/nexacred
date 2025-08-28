"""
API Components
==============

FastAPI web service for RAG chatbot.
"""

from .app import app, setup_routes

__all__ = ["app", "setup_routes"]
