# ML Module - RAG Chatbot System

## 🎯 Overview

This ML module contains the **RAG (Retrieval-Augmented Generation) Chatbot System** - the core AI component of NexaCred that provides intelligent financial advice using IBM Granite 3.1 8B Instruct model with Indian financial regulations knowledge.

## 🤖 Main System: RAG Chatbot

### 📁 Core Directory Structure

```
ml/
├── rag_chatbot/                 # 🔥 MAIN RAG CHATBOT SYSTEM
│   ├── models/                  # AI model components
│   │   ├── generator.py         # IBM Granite 3.1 8B integration
│   │   └── embeddings.py        # Sentence transformers
│   ├── retrieval/               # Document retrieval system
│   │   ├── dummy.py             # Sample financial regulations
│   │   └── mongo_stub.py        # MongoDB Vector Search (ready)
│   ├── pipeline/                # RAG orchestration
│   │   ├── rag.py               # Main RAG pipeline
│   │   ├── token_utils.py       # Token management
│   │   └── chunking.py          # Text processing
│   ├── api/                     # FastAPI web service
│   │   └── app.py               # REST endpoints with Swagger docs
│   ├── cli.py                   # Interactive command-line interface
│   ├── config.py                # Configuration management
│   ├── prompts.py               # Financial prompt templates
│   ├── test_smoke.py            # Comprehensive functionality tests
│   └── README.md                # Detailed RAG documentation
└── [temporary files...]         # See Temporary/Legacy Files section
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch transformers sentence-transformers
pip install fastapi uvicorn pydantic-settings
pip install huggingface-hub
```

### 1. Interactive CLI

```bash
cd rag_chatbot

# Start interactive chat
python -m rag_chatbot

# Single query
python cli.py --query "What are RBI guidelines for personal loans?"

# Health check
python cli.py --health


### 3. Python Integration

```python
from rag_chatbot.pipeline.rag import RAGPipeline
from rag_chatbot.retrieval.dummy import DummyRetriever
from rag_chatbot.config import Config

# Initialize system
config = Config()
retriever = DummyRetriever(config)
pipeline = RAGPipeline(retriever, config)

# Ask questions
response = pipeline.generate_response(
    "What are the eligibility criteria for credit cards in India?"
)

print(f"Answer: {response['response']}")
print(f"Retrieved docs: {len(response['retrieved_docs'])}")
```

## 🎯 Features

### 🧠 AI Capabilities

- **IBM Granite 3.1 8B Instruct**: Advanced language model for financial responses
- **Intelligent Model Caching**: Downloads once, loads instantly on subsequent runs
- **4-bit Quantization**: Memory-efficient inference with bitsandbytes support
- **Context Management**: Smart token counting and context window optimization

### 🔍 RAG System

- **Semantic Retrieval**: sentence-transformers for document similarity
- **Financial Knowledge**: Specialized in Indian banking, lending, payments
- **MongoDB Ready**: Designed for MongoDB Atlas Vector Search
- **Dummy Retriever**: 5 sample Indian financial regulations for testing

### 🌐 API & Interfaces

- **FastAPI Service**: Production-ready REST API with automatic documentation
- **Interactive CLI**: Chat-style command-line interface
- **Health Monitoring**: System health checks and diagnostics
- **Error Handling**: Graceful fallbacks and informative error messages

## 📚 Knowledge Domains

The RAG system specializes in Indian financial regulations:

- **🏦 Banking**: KYC, AML, account procedures, RBI guidelines
- **💳 Credit Cards**: Interest rates, eligibility, compliance requirements
- **🏠 Loans**: Personal, home, business loan regulations and criteria
- **🔄 P2P Lending**: RBI guidelines for peer-to-peer platforms
- **💰 Digital Payments**: UPI, NEFT, RTGS transaction rules
- **📋 Compliance**: Regulatory requirements, penalties, documentation

## 🔧 Configuration

Key configuration options in `rag_chatbot/config.py`:

```python
# Model Configuration
granite_model_id = "ibm-granite/granite-3.1-8b-instruct"
embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"

# Generation Parameters
temperature = 0.1          # Conservative for financial advice
max_output_length = 512    # Response length limit
retrieval_top_k = 5        # Documents to retrieve

# Hardware Options
use_4bit = False           # Enable for memory efficiency
device_map = "auto"        # Automatic GPU/CPU mapping
```

## 🧪 Testing

```bash
cd rag_chatbot

# Run all tests
python test_smoke.py

# Test specific components
python -c "from config import Config; print('Config loaded:', Config().granite_model_id)"
python -c "from retrieval.dummy import DummyRetriever; from config import Config; r=DummyRetriever(Config()); print('Retriever test:', len(r.retrieve('loans')))"
```

## 📖 API Documentation

### Health Check
```bash
curl http://localhost:8000/health
```

### Query Processing
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the RBI guidelines for digital payments?",
    "use_dummy_retriever": true
  }'
```

### Response Format
```json
{
  "response": "According to RBI guidelines, digital payments...",
  "retrieved_docs": [
    {
      "content": "RBI Digital Payment Guidelines...",
      "metadata": {"source": "rbi_guidelines.pdf"}
    }
  ],
  "metadata": {
    "status": "success",
    "num_retrieved_docs": 3
  }
}
```

## 🚀 Production Deployment

### Environment Variables
```bash
export GRANITE_MODEL_ID="ibm-granite/granite-3.1-8b-instruct"
export EMBEDDING_MODEL_ID="sentence-transformers/all-MiniLM-L6-v2"
export USE_4BIT="true"  # For memory efficiency
```

### Production Server
```bash
# Install production dependencies
pip install uvicorn[standard] gunicorn

# Start with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api.app:app --bind 0.0.0.0:8000
```

### MongoDB Integration (Ready)
```python
# When ready to implement MongoDB Vector Search
from rag_chatbot.retrieval.mongo_stub import MongoRetrieverStub

# The stub contains detailed implementation guidance
retriever = MongoRetrieverStub(config)  # Will raise NotImplementedError with TODO
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB for model files
- **CPU**: Multi-core processor

### Recommended for Production
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: SSD with 50GB+ free space
- **Network**: High-bandwidth for model downloads

---

## 🗂️ Temporary/Legacy Files

> **Note**: The following files are temporary and will be refactored or removed in future versions. They contain experimental or legacy functionality that has been superseded by the main RAG chatbot system.

### 📂 Legacy ML Components

```
ml/
├── credit_scoring.py           # 🚧 Legacy: Basic credit scoring (superseded by RAG)
├── credit_utils.py             # 🚧 Legacy: Utility functions (being integrated)
├── data_preprocessor.py        # 🚧 Legacy: Data preprocessing (being updated)
├── enhanced_preprocessor.py    # 🚧 Legacy: Enhanced preprocessing (temporary)
├── financial_assistant.py      # 🚧 Legacy: Old assistant (replaced by RAG)
├── granite_financial_ai.py     # 🚧 Temporary: Will be refactored to use RAG backend
├── hybrid_credit_system.py     # 🚧 Legacy: Multi-model system (being integrated)
└── train_model.py              # 🚧 Legacy: Traditional ML training (supplementary)
```

### 🔄 Integration Status

- **✅ Completed**: Main RAG chatbot system is production-ready
- **🔄 In Progress**: Integrating legacy ML components with RAG backend
- **📅 Planned**: Consolidation of all ML functionality under RAG system

### 🚫 Removed Files

These duplicate implementations have been removed to eliminate redundancy:

- ~~`indian_financial_rag.py`~~ → Consolidated into `rag_chatbot/`
- ~~`rag_api_server.py`~~ → Replaced by `rag_chatbot/api/app.py`
- ~~`setup_rag_chatbot.py`~~ → Replaced by main setup instructions
- ~~`test_rag_chatbot.py`~~ → Replaced by `rag_chatbot/test_smoke.py`
- ~~`README_RAG_CHATBOT.md`~~ → Replaced by `rag_chatbot/README.md`
- ~~`requirements_rag.txt`~~ → Integrated into main requirements

---

## 🎯 Focus on RAG Chatbot

**The RAG chatbot system in `rag_chatbot/` is the primary AI component.** All development, testing, and production deployment should focus on this system. Legacy files are maintained temporarily for reference and gradual migration.

For detailed technical documentation, see [`rag_chatbot/README.md`](rag_chatbot/README.md).

## 🤝 Contributing

When contributing to the ML module:

1. **Primary Focus**: Work on the `rag_chatbot/` system
2. **Legacy Files**: Mark any changes to legacy files as temporary
3. **Integration**: Help migrate useful functionality from legacy files to RAG system
4. **Documentation**: Update this README when files are consolidated or removed

---

**RAG Chatbot System** - The Future of Financial AI 🚀
