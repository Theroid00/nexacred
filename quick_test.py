#!/usr/bin/env python3
"""
Quick System Test - Simplified Version
=====================================

Tests the hybrid system with a smaller dataset sample
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append('ml')

def quick_test():
    print("🧪 Quick System Test - Hybrid Credit Scoring")
    print("=" * 50)

    # Test 1: Check if datasets load
    print("\n📊 Testing dataset loading...")
    try:
        train = pd.read_csv('datasets/train.csv')
        test = pd.read_csv('datasets/test.csv')
        print(f"✅ Train dataset: {train.shape}")
        print(f"✅ Test dataset: {test.shape}")
        print(f"✅ Target distribution: {train['Credit_Score'].value_counts().to_dict()}")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return

    # Test 2: Test preprocessing on small sample
    print("\n🔧 Testing preprocessing on sample...")
    try:
        from enhanced_preprocessor import HybridDataPreprocessor

        # Use small sample for quick test
        train_sample = train.sample(1000, random_state=42)
        test_sample = test.sample(500, random_state=42)

        # Save samples temporarily
        train_sample.to_csv('temp_train_sample.csv', index=False)
        test_sample.to_csv('temp_test_sample.csv', index=False)

        # Test preprocessing
        preprocessor = HybridDataPreprocessor()
        train_proc, test_proc = preprocessor.clean_and_preprocess(
            'temp_train_sample.csv', 'temp_test_sample.csv'
        )

        print(f"✅ Preprocessing successful!")
        print(f"   Processed train: {train_proc.shape}")
        print(f"   Processed test: {test_proc.shape}")

        # Clean up temp files
        os.remove('temp_train_sample.csv')
        os.remove('temp_test_sample.csv')

    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return

    # Test 3: Test ML training on sample
    print("\n🤖 Testing ML model training...")
    try:
        from hybrid_credit_system import HybridCreditScoringSystem
        from sklearn.model_selection import train_test_split

        # Prepare data
        X, y = preprocessor.prepare_target_variable(train_proc)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train models
        credit_system = HybridCreditScoringSystem()
        results = credit_system.train_traditional_models(X_train, y_train, X_val, y_val)

        print(f"✅ ML training successful!")
        for model, metrics in results.items():
            print(f"   {model}: {metrics['validation_accuracy']:.4f}")

    except Exception as e:
        print(f"❌ ML training failed: {e}")
        return

    # Test 4: Test Main RAG Chatbot integration
    print("\n🧠 Testing Main RAG Chatbot integration...")
    try:
        from rag_chatbot.pipeline.rag import RAGPipeline
        from rag_chatbot.retrieval.dummy import DummyRetriever
        from rag_chatbot.config import Config

        config = Config()
        retriever = DummyRetriever(config)
        rag_pipeline = RAGPipeline(retriever, config)

        # Test chat
        response = rag_pipeline.generate_response(
            "What affects credit scores?"
        )
        print(f"✅ RAG chat working!")
        print(f"   Response length: {len(response['response'])} characters")

        # Test risk assessment query
        risk_query = "What are the risk factors for loan defaults?"
        risk_response = rag_pipeline.generate_response(risk_query)
        
        print(f"✅ Risk assessment generation working!")
        print(f"   Documents retrieved: {len(risk_response.get('retrieved_docs', []))}")

    except Exception as e:
        print(f"❌ RAG Chatbot integration failed: {e}")
        return

    print("\n🎉 ALL TESTS PASSED!")
    print("🚀 Hybrid system with main RAG chatbot is working correctly!")

    return True

if __name__ == "__main__":
    quick_test()
