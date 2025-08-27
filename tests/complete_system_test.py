#!/usr/bin/env python3
"""
Complete System Training and Testing - Updated for Optimized ML Structure
=========================================================================

Tests the optimized ML components: data preprocessing, hybrid credit system,
and financial assistant integration
"""

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('../ml')

from data_preprocessor import NexaCreditDataPreprocessor
from hybrid_credit_system import HybridCreditScoringSystem
from financial_assistant import NexaCredFinancialAssistant
from granite_agents import IBMGraniteFinancialAI
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main training and testing pipeline"""
    print("🚀 Starting NexaCred Optimized ML System Training & Testing")
    print("=" * 65)

    # Step 1: Data Preprocessing
    print("\n📊 STEP 1: DATA PREPROCESSING")
    preprocessor = NexaCreditDataPreprocessor()

    # Process datasets
    print("Processing training dataset...")
    train_processed = preprocessor.preprocess_dataset('../datasets/train.csv', is_training=True)

    print("Processing test dataset...")
    test_processed = preprocessor.preprocess_dataset('../datasets/test.csv', is_training=False)

    # Prepare target variable for training
    if 'Credit_Score' in train_processed.columns:
        # Map Credit_Score to numeric categories
        score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
        y_train = train_processed['Credit_Score'].map(score_mapping)
        X_train_full = train_processed.drop(columns=['Credit_Score']).select_dtypes(include=[np.number])

        # Handle any remaining NaN values
        X_train_full = X_train_full.fillna(X_train_full.median())

        # Split training data for validation
        X_train, X_val, y_train_split, y_val = train_test_split(
            X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    else:
        print("❌ Credit_Score column not found in training data")
        return

    print(f"✅ Data preprocessing completed:")
    print(f"   Training set: {X_train.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Target distribution: {y_train.value_counts().to_dict()}")

    # Step 2: Traditional ML Model Training
    print("\n🤖 STEP 2: ML MODEL TRAINING")
    credit_system = HybridCreditScoringSystem()

    # Train traditional ML models
    training_results = credit_system.train_traditional_models(
        X_train, y_train_split, X_val, y_val
    )

    print("\n📈 Training Results Summary:")
    for model_name, metrics in training_results.items():
        print(f"   {model_name}: Validation Accuracy = {metrics['validation_accuracy']:.4f}")

    # Step 3: Test Financial Assistant Integration
    print("\n💼 STEP 3: FINANCIAL ASSISTANT TESTING")
    assistant = NexaCredFinancialAssistant()

    # Test credit scoring
    test_customer = {
        'annual_income': 800000,
        'debt_to_income_ratio': 0.25,
        'credit_utilization_ratio': 0.15,
        'number_of_late_payments': 0,
        'age': 35
    }

    score_result = assistant.get_score("test_user_001", test_customer)
    print(f"✅ Credit Score Test:")
    print(f"   Score: {score_result.get('credit_score', 'N/A')}")
    print(f"   Category: {score_result.get('credit_category', 'N/A')}")
    print(f"   Risk Level: {score_result.get('risk_level', 'N/A')}")

    # Test loan offer generation
    offer_result = assistant.generate_offer("test_user_001", test_customer, "personal")
    print(f"\n✅ Loan Offer Test:")
    if offer_result.get('offer', {}).get('approved'):
        print(f"   Approved: ₹{offer_result['offer']['max_amount']}")
        print(f"   Interest Rate: {offer_result['offer']['interest_rate']}%")
        print(f"   Term: {offer_result['offer']['term_months']} months")
    else:
        print(f"   Status: Declined - {offer_result.get('offer', {}).get('reason', 'Unknown')}")

    # Test fraud detection
    test_transaction = {
        'amount': 50000,
        'daily_transaction_count': 3,
        'location': 'verified',
        'hour': 14,
        'transaction_id': 'txn_001'
    }

    fraud_result = assistant.detect_fraud(test_transaction)
    print(f"\n✅ Fraud Detection Test:")
    print(f"   Risk Score: {fraud_result.get('risk_score', 'N/A')}")
    print(f"   Likelihood: {fraud_result.get('fraud_likelihood', 'N/A')}")
    print(f"   Recommendation: {fraud_result.get('recommendation', 'N/A')}")

    # Step 4: Test Granite AI Integration
    print("\n🧠 STEP 4: AI ASSISTANCE TESTING")
    granite_ai = IBMGraniteFinancialAI()

    # Test financial advice
    advice = granite_ai.generate_financial_advice("How can I improve my credit score?")
    print(f"✅ Financial Advice Test:")
    print(f"   Query: How can I improve my credit score?")
    print(f"   Response: {advice}")

    # Test credit profile analysis
    profile_analysis = granite_ai.analyze_credit_profile(test_customer)
    print(f"\n✅ Credit Profile Analysis:")
    print(f"   Strength: {profile_analysis.get('profile_strength', 'N/A')}")
    print(f"   Insights: {profile_analysis.get('key_insights', [])}")
    print(f"   Improvements: {profile_analysis.get('improvement_areas', [])}")

    # Step 5: System Status Check
    print("\n🔍 STEP 5: SYSTEM STATUS CHECK")
    system_status = assistant.get_system_status()
    print(f"✅ System Status: {system_status.get('status', 'Unknown')}")
    print(f"   Components: {system_status.get('components', {})}")
    print(f"   Version: {system_status.get('version', 'Unknown')}")

    print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    main()
