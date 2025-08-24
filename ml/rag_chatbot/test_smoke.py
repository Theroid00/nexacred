"""
Smoke Tests
===========

Basic functionality tests for the RAG chatbot system.
Validates core components and pipeline integration.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import Config
from pipeline.rag import RAGPipeline
from retrieval.dummy import DummyRetriever
from models.generator import load_generator
from models.embeddings import load_embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config():
    """Test configuration loading."""
    print("🔧 Testing configuration...")
    
    try:
        config = Config()
        assert config.granite_model_id is not None
        assert config.embedding_model_id is not None
        assert config.temperature > 0
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_dummy_retriever():
    """Test dummy retriever functionality."""
    print("📚 Testing dummy retriever...")
    
    try:
        config = Config()
        retriever = DummyRetriever(config)
        
        # Test retrieval
        docs = retriever.retrieve("loan eligibility criteria", top_k=2)
        assert len(docs) > 0
        assert hasattr(docs[0], 'content')
        
        print(f"✅ Retrieved {len(docs)} documents")
        return True
    except Exception as e:
        print(f"❌ Dummy retriever test failed: {e}")
        return False


def test_embeddings():
    """Test embedding model loading and functionality."""
    print("🔤 Testing embeddings...")
    
    try:
        config = Config()
        embedder = load_embedder(config)
        
        # Test embedding
        texts = ["test text", "another test"]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0  # Vector has dimensions
        
        print(f"✅ Generated embeddings with dimension {len(embeddings[0])}")
        return True
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        print("This might be expected if models are not downloaded yet")
        return False


def test_generator():
    """Test text generator loading."""
    print("🤖 Testing generator...")
    
    try:
        config = Config()
        generator, tokenizer = load_generator(config)
        
        # Basic check that models loaded
        assert generator is not None
        assert tokenizer is not None
        
        print("✅ Generator models loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Generator test failed: {e}")
        print("This might be expected if models are not downloaded yet")
        return False


def test_pipeline_init():
    """Test RAG pipeline initialization."""
    print("🔗 Testing pipeline initialization...")
    
    try:
        config = Config()
        retriever = DummyRetriever(config)
        pipeline = RAGPipeline(retriever, config)
        
        # Test health check
        health = pipeline.health_check()
        assert "pipeline" in health
        
        print("✅ Pipeline initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Pipeline initialization test failed: {e}")
        return False


def test_simple_query():
    """Test simple query processing (without model loading)."""
    print("💬 Testing simple query processing...")
    
    try:
        config = Config()
        retriever = DummyRetriever(config)
        pipeline = RAGPipeline(retriever, config)
        
        # Test retrieval part only (models won't be loaded)
        query = "What are the eligibility criteria for personal loans?"
        docs = retriever.retrieve(query, top_k=2)
        
        assert len(docs) > 0
        print(f"✅ Query processed, retrieved {len(docs)} documents")
        return True
    except Exception as e:
        print(f"❌ Simple query test failed: {e}")
        return False


def test_import_structure():
    """Test that all imports work correctly."""
    print("📦 Testing import structure...")
    
    try:
        # Test all major imports
        from config import Config
        from pipeline.rag import RAGPipeline
        from retrieval.dummy import DummyRetriever
        from retrieval.mongo_stub import MongoRetrieverStub
        from models.embeddings import load_embedder
        from models.generator import load_generator
        from prompts import GRANITE_FINANCIAL_PROMPT
        
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def run_smoke_tests():
    """Run all smoke tests."""
    print("🧪 Running RAG Chatbot Smoke Tests")
    print("=" * 50)
    
    tests = [
        test_import_structure,
        test_config,
        test_dummy_retriever,
        test_pipeline_init,
        test_simple_query,
        test_embeddings,
        test_generator,
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            results.append(False)
        print()
    
    # Summary
    print("📋 Test Summary")
    print("-" * 20)
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        print("Note: Model loading tests may fail if models are not downloaded")
        return False


if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
