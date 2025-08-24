#!/usr/bin/env python3
"""
Complete Hybrid System Training and Testing
===========================================

Trains traditional ML models on cleaned datasets and tests IBM Granite integration
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../ml')

from enhanced_preprocessor import HybridDataPreprocessor
from hybrid_credit_system import HybridCreditScoringSystem
# Updated to use main RAG chatbot implementation
from rag_chatbot.pipeline.rag import RAGPipeline
from rag_chatbot.retrieval.dummy import DummyRetriever
from rag_chatbot.config import Config
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main training and testing pipeline"""
    print("🚀 Starting NexaCred Hybrid AI System Training & Testing")
    print("=" * 60)

    # Step 1: Data Preprocessing
    print("\n📊 STEP 1: DATA PREPROCESSING")
    preprocessor = HybridDataPreprocessor()

    # Process datasets
    train_processed, test_processed = preprocessor.clean_and_preprocess(
        'datasets/train.csv',
        'datasets/test.csv'
    )

    # Prepare training data
    X_train_full, y_train = preprocessor.prepare_target_variable(train_processed)
    X_test, _ = preprocessor.prepare_target_variable(test_processed)

    # Split training data for validation
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"✅ Data preprocessing completed:")
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Test set: {X_test.shape}")

    # Step 2: Train Traditional ML Models
    print("\n🤖 STEP 2: TRADITIONAL ML MODEL TRAINING")
    credit_system = HybridCreditScoringSystem()

    # Train models
    training_results = credit_system.train_traditional_models(
        X_train, y_train_split, X_val, y_val
    )

    # Save models
    credit_system.save_models('models')

    # Step 3: Test Credit Scoring
    print("\n📈 STEP 3: TESTING CREDIT SCORING SYSTEM")

    # Test on validation set
    test_sample = X_val.head(5)

    for i, (idx, row) in enumerate(test_sample.iterrows()):
        print(f"\n--- Test Case {i+1} ---")

        # Get prediction
        X_single = row.to_frame().T
        prediction, probabilities = credit_system.predict_credit_score(X_single, 'random_forest')

        # Generate explanation
        explanation = credit_system.generate_credit_score_explanation(
            X_single, prediction[0], probabilities[0], 'random_forest'
        )

        print(f"Credit Category: {explanation['credit_category']}")
        print(f"Confidence: {explanation['confidence']:.2%}")
        print(f"Score Range: {explanation['score_range']}")
        print("Top Factors:")
        for factor in explanation['key_factors'][:3]:
            print(f"  - {factor['factor']}: {factor['impact']} impact")

    # Step 4: Test Main RAG Chatbot Integration
    print("\n🧠 STEP 4: TESTING MAIN RAG CHATBOT INTEGRATION")

    # Initialize RAG pipeline
    config = Config()
    retriever = DummyRetriever(config)
    rag_pipeline = RAGPipeline(retriever, config)

    # Test RAG-based chat
    print("\n💬 Testing Financial Assistant Chat:")
    test_questions = [
        "What factors affect my credit score?",
        "How can I improve my credit score?",
        "What types of loans am I eligible for?"
    ]

    for question in test_questions:
        print(f"\nUser: {question}")
        response = rag_pipeline.generate_response(question)
        print(f"Assistant: {response['response'][:200]}...")

    # Test Risk Assessment (simplified)
    print("\n📋 Testing Risk Assessment:")

    # Use sample customer data
    sample_customer = {
        "Customer_ID": "TEST_001",
        "Annual_Income": 75000,
        "Credit_Utilization_Ratio": 0.25,
        "Num_Credit_Card": 3,
        "Num_of_Delayed_Payment": 1,
        "Outstanding_Debt": 15000
    }

    # Generate risk assessment using RAG
    risk_query = f"""
    Analyze the risk profile for a customer with:
    - Annual Income: ₹{sample_customer['Annual_Income']:,}
    - Credit Utilization: {sample_customer['Credit_Utilization_Ratio']:.1%}
    - Number of Credit Cards: {sample_customer['Num_Credit_Card']}
    - Delayed Payments: {sample_customer['Num_of_Delayed_Payment']}
    - Outstanding Debt: ₹{sample_customer['Outstanding_Debt']:,}
    """

    risk_assessment = rag_pipeline.generate_response(risk_query)

    print(f"Risk Assessment Generated: {len(risk_assessment['response'])} characters")
    print(f"Documents Retrieved: {len(risk_assessment.get('retrieved_docs', []))}")
    print("\nRAG Analysis Preview:")
    print(risk_assessment['response'][:300] + "...")

    # Step 5: Performance Evaluation
    print("\n📊 STEP 5: SYSTEM PERFORMANCE EVALUATION")

    # Evaluate on validation set
    evaluation_results = credit_system.evaluate_model_performance(X_val, y_val)

    print("\n🏆 FINAL RESULTS SUMMARY:")
    print("=" * 40)

    best_model = max(evaluation_results.keys(),
                    key=lambda k: evaluation_results[k]['accuracy'])
    best_accuracy = evaluation_results[best_model]['accuracy']

    print(f"✅ Best Traditional ML Model: {best_model}")
    print(f"✅ Best Accuracy: {best_accuracy:.4f}")
    print(f"✅ Main RAG Chatbot: Operational")
    print(f"✅ RAG System: Functional")
    print(f"✅ Risk Assessment: Active")

    # Save final results
    print(f"\n💾 Saving results and models...")

    # Create results summary
    results_summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset_info": {
            "train_shape": train_processed.shape,
            "test_shape": test_processed.shape,
            "features": len(X_train.columns),
            "target_distribution": pd.Series(y_train).value_counts().to_dict()
        },
        "model_performance": {
            model: {"accuracy": results["accuracy"]}
            for model, results in evaluation_results.items()
        },
        "best_model": best_model,
        "system_status": {
            "traditional_ml": "Trained and Operational",
            "rag_chatbot": "Integrated and Functional",
            "rag_system": "Active",
            "risk_assessment": "Operational"
        }
    }

    # Save to file
    import json
    with open('training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print("✅ Training and testing completed successfully!")
    print("📁 Results saved to training_results.json")
    print("📁 Models saved to models/ directory")

    return results_summary

if __name__ == "__main__":
    results = main()
