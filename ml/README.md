# NexaCred ML Components

## Overview

The NexaCred ML folder contains a comprehensive suite of machine learning and AI components for credit scoring, risk assessment, and financial assistance. The architecture has been optimized to eliminate redundancy while providing a robust, scalable system.

## Architecture

```
ml/
├── credit_utils.py           # Core utility functions
├── data_preprocessor.py      # Data preprocessing pipeline
├── hybrid_credit_system.py   # Advanced ML models
├── financial_assistant.py    # Integration layer & API
├── granite_agents.py         # AI assistance stub
└── rag_chatbot/             # RAG chatbot system
```

## Core Components

### 1. Credit Utilities (`credit_utils.py`)
**Purpose**: Centralized utility functions for credit scoring calculations

**Key Features**:
- Heuristic credit scoring (300-900 range)
- 5-category classification system (Poor, Fair, Good, Very Good, Exceptional)
- Risk factor extraction and analysis
- Recommendation generation for credit improvement
- Score-to-category mapping utilities

**Functions**:
- `calculate_base_credit_score(data)` - Compute heuristic credit score
- `extract_score_factors(data)` - Extract qualitative factors
- `generate_recommendations_three_class(prediction)` - Generate improvement recommendations
- `score_to_category_five(score)` - Map score to 5-class category
- `score_to_probabilities_five(score)` - Generate probability distribution

### 2. Data Preprocessor (`data_preprocessor.py`)
**Purpose**: Comprehensive data preprocessing pipeline for credit datasets

**Key Features**:
- Missing value imputation with intelligent strategies
- Outlier detection and handling
- Data type validation and conversion
- Feature engineering and creation
- Categorical variable encoding
- Data quality validation

**Main Class**: `NexaCreditDataPreprocessor`
- `preprocess_dataset(file_path, is_training)` - Main preprocessing pipeline
- Handles income normalization, age validation, SSN formatting
- Statistical imputation based on occupation and other factors
- Outlier detection using IQR method

### 3. Hybrid Credit System (`hybrid_credit_system.py`)
**Purpose**: Advanced ML models using traditional algorithms

**Key Features**:
- Multiple ML algorithms (Random Forest, Gradient Boosting, Logistic Regression)
- Model training and evaluation with cross-validation
- Feature importance analysis
- 3-class prediction system (Poor, Standard, Good)
- Credit score explanation and reasoning

**Main Class**: `HybridCreditScoringSystem`
- `train_traditional_models(X_train, y_train, X_val, y_val)` - Train multiple models
- `predict_credit_score(X, model_name)` - Make predictions
- `generate_credit_score_explanation(X, prediction, probabilities)` - Explain predictions

### 4. Financial Assistant (`financial_assistant.py`)
**Purpose**: Main integration layer providing unified API for backend

**Key Features**:
- Orchestrates all ML components
- Provides standardized API for credit scoring
- Loan offer generation based on credit assessment
- Basic fraud detection capabilities
- System health monitoring

**Main Class**: `NexaCredFinancialAssistant`
- `get_score(user_id, customer_data)` - Get credit score with explanation
- `generate_offer(user_id, customer_data, loan_type)` - Generate loan offers
- `detect_fraud(transaction_data)` - Detect fraudulent transactions
- `get_system_status()` - System health check

### 5. Granite Agents (`granite_agents.py`)
**Purpose**: AI assistance stub for financial advice

**Key Features**:
- Basic financial advice generation
- Credit profile analysis
- Knowledge base for financial tips
- Chatbot functionality without external dependencies

**Main Class**: `IBMGraniteFinancialAI`
- `generate_financial_advice(user_query, context)` - Generate advice
- `analyze_credit_profile(customer_data)` - Analyze credit profile

### 6. RAG Chatbot (`rag_chatbot/`)
**Purpose**: Advanced RAG-based chatbot for financial regulations

**Key Features**:
- IBM Granite 3.0 8B Instruct integration
- Vector embeddings for semantic search
- MongoDB integration for regulation retrieval
- Multi-domain financial expertise
- Real-time regulation updates

## Data Flow

```
Customer Data → Data Preprocessor → ML Models → Financial Assistant → Backend API
                     ↓
              Credit Utilities ← → Granite Agents (AI Advice)
                     ↓
              RAG Chatbot (Regulatory Guidance)
```

## Usage Examples

### Basic Credit Scoring
```python
from financial_assistant import NexaCredFinancialAssistant

assistant = NexaCredFinancialAssistant()
customer_data = {
    'annual_income': 800000,
    'debt_to_income_ratio': 0.25,
    'credit_utilization_ratio': 0.15,
    'number_of_late_payments': 0,
    'age': 35
}

result = assistant.get_score("user123", customer_data)
print(f"Credit Score: {result['credit_score']}")
print(f"Category: {result['credit_category']}")
```

### Loan Offer Generation
```python
offer = assistant.generate_offer("user123", customer_data, "personal")
if offer['offer']['approved']:
    print(f"Approved Amount: ₹{offer['offer']['max_amount']}")
    print(f"Interest Rate: {offer['offer']['interest_rate']}%")
```

### Advanced ML Model Training
```python
from hybrid_credit_system import HybridCreditScoringSystem
from data_preprocessor import NexaCreditDataPreprocessor

# Preprocess data
preprocessor = NexaCreditDataPreprocessor()
train_df = preprocessor.preprocess_dataset("train.csv", is_training=True)

# Train models
credit_system = HybridCreditScoringSystem()
results = credit_system.train_traditional_models(X_train, y_train, X_val, y_val)
```

## API Integration

The `financial_assistant.py` module provides the main interface expected by the backend:

- **Backend Import**: `from financial_assistant import NexaCredFinancialAssistant`
- **Granite Import**: `from granite_agents import IBMGraniteFinancialAI`

Both modules are designed to be drop-in replacements for the backend's expected interfaces.

## Dependencies

Core dependencies:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms
- `datetime` - Timestamp management

Optional dependencies:
- `pymongo` - MongoDB integration (RAG chatbot)
- `sentence-transformers` - Embeddings (RAG chatbot)
- `ibm-watson-machine-learning` - IBM Granite integration

## Testing

See the `tests/` folder for comprehensive system tests covering:
- Individual component functionality
- Integration testing
- End-to-end system validation

## Configuration

Most components work out-of-the-box with sensible defaults. For advanced configuration:

- Modify score ranges in `credit_utils.py`
- Adjust ML model parameters in `hybrid_credit_system.py`
- Configure preprocessing steps in `data_preprocessor.py`

## Performance

- **Credit Scoring**: Near-instantaneous with heuristic methods
- **ML Training**: Minutes for traditional algorithms on typical datasets
- **Data Preprocessing**: Handles datasets with 100K+ records efficiently
- **Memory Usage**: Optimized for production environments

## Contributing

When contributing to the ML components:

1. Maintain the separation of concerns between modules
2. Update this README when adding new functionality
3. Ensure backward compatibility with the backend API
4. Add appropriate tests for new features

## Version History

- **v2.0.0** (Current) - Optimized architecture with eliminated redundancy
- **v1.x.x** - Legacy versions (deprecated)
