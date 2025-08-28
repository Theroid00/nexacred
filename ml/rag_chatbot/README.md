# Indian Financial Regulation RAG Chatbot

An AI-powered chatbot that uses IBM Granite 3.1 8B Instruct model with Retrieval-Augmented Generation (RAG) to answer questions about Indian financial regulations, loans, credit, debit cards, and peer-to-peer transactions.

## 🎯 Features

- **🤖 IBM Granite 3.1 8B Model**: State-of-the-art language model for response generation
- **🔍 RAG Architecture**: Retrieves relevant context from knowledge base before generation
- **🏦 Indian Financial Focus**: Specialized for Indian banking, lending, and payment regulations
- **📚 MongoDB Integration**: Designed for MongoDB Atlas Vector Search (stub included)
- **🌐 REST API**: FastAPI-based web service with Swagger documentation
- **💬 CLI Interface**: Interactive command-line chat interface
- **⚡ Optimized Performance**: 4-bit quantization support for efficient inference

## 🏗️ Architecture

```
RAG Pipeline:
Query → Embedding → Retrieval → Context + Prompt → Generation → Response
```

**Components:**
- **Models**: IBM Granite 3.1 8B + Sentence Transformers for embeddings
- **Retrieval**: MongoDB Atlas Vector Search (with dummy fallback)
- **Pipeline**: RAG orchestration with context management
- **API**: FastAPI web service
- **CLI**: Interactive command-line interface

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers sentence-transformers
pip install fastapi uvicorn pydantic-settings
pip install huggingface-hub
pip install pymongo  # For MongoDB (when implementing)
```

### 1. Basic Usage (CLI)

```bash
# Interactive chat mode
python -m rag_chatbot

# Single query mode
python cli.py --query "What are RBI guidelines for personal loans?"

# Health check
python cli.py --health
```

### 2. Web API

```bash
# Start server
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Test endpoint
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the eligibility criteria for credit cards in India?"}'
```

### 3. Smoke Tests

```bash
python test_smoke.py
```

## 📋 API Endpoints

- **GET** `/health` - Health check
- **POST** `/infer` - Generate response
- **GET** `/docs` - Swagger UI
- **GET** `/info` - System information

### Example Request

```json
{
  "query": "What are the RBI guidelines for peer-to-peer lending?",
  "use_dummy_retriever": true
}
```

### Example Response

```json
{
  "response": "According to RBI guidelines, peer-to-peer lending platforms must...",
  "retrieved_docs": [
    {
      "content": "RBI guidelines for P2P lending...",
      "metadata": {"source": "rbi_guidelines.pdf"}
    }
  ],
  "metadata": {
    "status": "success",
    "num_retrieved_docs": 3
  }
}
```

## 🔧 Configuration

Configuration via environment variables or `config.py`:

```python
class Config:
    granite_model_id: str = "ibm-granite/granite-3.1-8b-instruct"
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Generation parameters
    temperature: float = 0.1
    top_p: float = 0.9
    max_output_length: int = 512
    
    # Retrieval parameters
    retrieval_top_k: int = 5
    max_input_length: int = 4096
```

## 📚 Example Queries

The chatbot specializes in Indian financial regulations:

- **Loans**: "What are the eligibility criteria for personal loans?"
- **Credit Cards**: "What is the maximum interest rate for credit cards in India?"
- **P2P Lending**: "What are RBI guidelines for peer-to-peer lending platforms?"
- **Digital Payments**: "What are the regulations for UPI transactions?"
- **Banking**: "What are KYC requirements for opening a savings account?"
- **Compliance**: "What are the penalties for non-compliance with AML regulations?"

## 🗄️ MongoDB Integration

The system is designed for MongoDB Atlas Vector Search. Current implementation includes:

**Stub Implementation** (`retrieval/mongo_stub.py`):
```python
# TODO: Implement actual MongoDB connection
class MongoRetrieverStub:
    def retrieve(self, query: str, top_k: int = 5):
        raise NotImplementedError("MongoDB integration pending")
```

**Required Setup** (when implementing):
1. MongoDB Atlas cluster with Vector Search
2. Financial regulation documents indexed with embeddings
3. Search index configuration for semantic similarity

## 🛠️ Development

### Project Structure

```
rag_chatbot/
├── config.py              # Configuration management
├── prompts.py             # Prompt templates
├── cli.py                 # Command-line interface
├── __main__.py           # Package entry point
├── test_smoke.py         # Basic functionality tests
├── models/               # AI models
│   ├── generator.py      # IBM Granite model
│   └── embeddings.py     # Sentence transformers
├── retrieval/            # Document retrieval
│   ├── base.py          # Abstract interfaces
│   ├── dummy.py         # Sample data retriever
│   └── mongo_stub.py    # MongoDB stub
├── pipeline/             # RAG orchestration
│   ├── rag.py           # Main pipeline
│   ├── token_utils.py   # Token management
│   └── chunking.py      # Text chunking
└── api/                  # Web service
    └── app.py           # FastAPI application
```

### Adding New Retrievers

1. Implement `AbstractRetriever` interface
2. Add to `config.py` if needed
3. Update CLI/API to support new retriever

### Model Customization

The system supports:
- **Different LLMs**: Modify `granite_model_id` in config
- **Embedding Models**: Change `embedding_model_id`
- **Quantization**: Enable 4-bit quantization for memory efficiency
- **Custom Prompts**: Edit `prompts.py`

## 🔍 Monitoring & Debugging

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

```bash
# CLI health check
python cli.py --health

# API health check
curl http://localhost:8000/health
```

## 🚧 Known Limitations

1. **MongoDB Integration**: Currently stubbed, needs implementation
2. **Model Size**: Granite 8B requires significant GPU memory
3. **Context Window**: Limited to 4K tokens for input
4. **Language Support**: Optimized for English queries about Indian regulations

## 🔄 Future Enhancements

- [ ] Complete MongoDB Atlas Vector Search integration
- [ ] Multi-language support (Hindi, regional languages)
- [ ] Real-time regulation updates
- [ ] Advanced retrieval strategies (hybrid search, re-ranking)
- [ ] Conversation memory and context tracking
- [ ] Batch processing for document ingestion

## 📄 License

This project is part of the NexaCred system. See main project license.

## 🤝 Contributing

1. Implement MongoDB integration in `mongo_stub.py`
2. Add more comprehensive tests
3. Optimize model performance
4. Extend with additional financial domains

---

**Note**: This is a production-ready foundation with MongoDB integration stubbed for future implementation. The dummy retriever contains sample Indian financial regulation data for testing and development.
