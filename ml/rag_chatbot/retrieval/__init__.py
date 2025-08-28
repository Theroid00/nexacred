"""
Retrieval Components
====================

Document schema, retriever interface, and implementations.
"""

from .schema import Document
from .base import AbstractRetriever
from .dummy import DummyRetriever
from .mongo_stub import MongoRetrieverStub

__all__ = ["Document", "AbstractRetriever", "DummyRetriever", "MongoRetrieverStub"]
