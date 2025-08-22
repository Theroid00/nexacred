#!/usr/bin/env python3
"""
NexaCred Financial Assistant - Core Engine
==========================================

This module implements the AI financial assistant that powers
real-time credit scoring, loan generation, and fraud detection.
Integrates with IBM Granite for advanced AI capabilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import hashlib
from typing import Dict, List, Optional, Any
import logging
import sys
import os

# Import our ML components
try:
    from granite_agents import IBMGraniteFinancialAI, CreditAnalysisResult
    GRANITE_AVAILABLE = True
except ImportError:
    GRANITE_AVAILABLE = False
    print("Warning: Granite AI not available")

# Import traditional ML model as fallback
try:
    from train_model import CreditScoreModel
    TRADITIONAL_ML_AVAILABLE = True
except ImportError:
    TRADITIONAL_ML_AVAILABLE = False
    print("Warning: Traditional ML model not available")

class NexaCredFinancialAssistant:
    """
    AI Financial Assistant for NexaCred Platform
    Handles real-time credit scoring, loan offers, and fraud detection
    Uses IBM Granite as primary AI with traditional ML as fallback
    """

    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(__name__)

        # Initialize Granite AI (primary)
        if GRANITE_AVAILABLE:
            try:
                self.granite_ai = IBMGraniteFinancialAI()
                self.primary_ai = "granite"
                self.logger.info("✅ Granite AI initialized as primary system")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Granite AI: {e}")
                self.granite_ai = None
                self.primary_ai = "traditional"
        else:
            self.granite_ai = None
            self.primary_ai = "traditional"

        # Initialize traditional ML model (fallback)
        if TRADITIONAL_ML_AVAILABLE:
            try:
                self.credit_model = CreditScoreModel()
                self._initialize_default_model()
                self.logger.info("✅ Traditional ML model initialized as fallback")
            except Exception as e:
                self.logger.warning(f"Failed to initialize traditional ML: {e}")
                self.credit_model = None
        else:
            self.credit_model = None

        # Score mapping configuration
        self.score_ranges = {
            0: (300, 579),  # Poor
            1: (580, 669),  # Fair
            2: (670, 739),  # Good
            3: (740, 799),  # Very Good
            4: (800, 900)   # Exceptional
        }

        # Performance tracking
        self.processed_requests = 0
        self.granite_requests = 0
        self.fallback_requests = 0

    def _initialize_default_model(self):
        """Initialize a default trained model"""
        if not self.credit_model:
            return

        self.logger.info("Initializing traditional credit scoring model...")
        X, y = self.credit_model.generate_synthetic_data(n_samples=1000, n_features=15)
        self.credit_model.train_model(X, y)
        self.logger.info("Traditional credit model initialized successfully")

    def get_score(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate credit score using best available AI system
        Returns comprehensive credit assessment
        """
        self.processed_requests += 1

        try:
            # Try Granite AI first (most advanced)
            if self.granite_ai and self.primary_ai == "granite":
                self.granite_requests += 1
                return self._get_score_granite(user_id, user_data)

            # Fallback to traditional ML
            elif self.credit_model:
                self.fallback_requests += 1
                return self._get_score_traditional(user_id, user_data)

            # Last resort: rule-based scoring
            else:
                return self._get_score_rules(user_id, user_data)

        except Exception as e:
            self.logger.error(f"Error in get_score: {e}")
            return self._get_score_rules(user_id, user_data)

    def _get_score_granite(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get credit score using Granite AI"""
        credit_result = self.granite_ai.analyze_credit_risk(user_data)

        return {
            'user_id': user_id,
            'credit_score': credit_result.credit_score,
            'category': self._score_to_category_name(credit_result.credit_score),
            'risk_level': credit_result.risk_level,
            'probability_of_default': credit_result.probability_of_default,
            'key_factors': credit_result.key_factors,
            'recommendations': credit_result.recommendations,
            'explanation': credit_result.explanation,
            'confidence': credit_result.confidence,
            'model_version': "granite-3.3-8b",
            'calculated_at': datetime.utcnow(),
            'blockchain_hash': self._generate_blockchain_hash(user_id, credit_result.credit_score)
        }

    def _get_score_traditional(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get credit score using traditional ML model"""
        # Convert user data to DataFrame format
        features_df = pd.DataFrame([user_data])

        # Predict category and convert to score
        category = self.credit_model.predict_credit_score_category(features_df)[0]
        score = self.credit_model.convert_category_to_score([category])[0]

        # Calculate probability of default
        pd_estimate = self._estimate_pd_from_score(score)

        # Generate factors and recommendations
        factors = self._analyze_factors_traditional(user_data)
        recommendations = self._generate_recommendations(user_data, score)

        return {
            'user_id': user_id,
            'credit_score': int(score),
            'category': self._score_to_category_name(score),
            'risk_level': self._score_to_risk_level(score),
            'probability_of_default': pd_estimate,
            'key_factors': factors,
            'recommendations': recommendations,
            'explanation': f"Traditional ML analysis with {len(factors)} key factors considered",
            'confidence': 0.82,
            'model_version': "traditional-ml-2.0",
            'calculated_at': datetime.utcnow(),
            'blockchain_hash': self._generate_blockchain_hash(user_id, score)
        }

    def _get_score_rules(self, user_id: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based credit scoring as last resort"""
        score = 500  # Base score
        factors = []

        # Basic scoring rules
        income = user_data.get('annual_income', 0)
        if income > 1000000:
            score += 100
            factors.append("High income")
        elif income > 500000:
            score += 50
            factors.append("Good income")

        dti = user_data.get('debt_to_income_ratio', 0.5)
        if dti < 0.3:
            score += 80
            factors.append("Low debt burden")
        elif dti > 0.6:
            score -= 100
            factors.append("High debt burden")

        # Cap score
        final_score = max(300, min(850, score))
        pd_estimate = self._estimate_pd_from_score(final_score)

        return {
            'user_id': user_id,
            'credit_score': final_score,
            'category': self._score_to_category_name(final_score),
            'risk_level': self._score_to_risk_level(final_score),
            'probability_of_default': pd_estimate,
            'key_factors': factors,
            'recommendations': ["Improve payment history", "Reduce debt utilization"],
            'explanation': "Rule-based scoring - limited AI functionality",
            'confidence': 0.65,
            'model_version': "rules-1.0",
            'calculated_at': datetime.utcnow(),
            'blockchain_hash': self._generate_blockchain_hash(user_id, final_score)
        }

    def generate_offer(self, user_id: str, user_data: Dict[str, Any], loan_type: str = "personal") -> Dict[str, Any]:
        """Generate personalized loan offer"""
        try:
            # Use Granite AI if available
            if self.granite_ai:
                return self.granite_ai.generate_loan_recommendation(user_data, loan_type)

            # Fallback loan generation
            credit_score = self.get_score(user_id, user_data)['credit_score']
            return self._generate_traditional_offer(credit_score, user_data, loan_type)

        except Exception as e:
            self.logger.error(f"Error generating offer: {e}")
            return {'error': True, 'message': 'Failed to generate loan offer'}

    def _generate_traditional_offer(self, credit_score: int, user_data: Dict[str, Any], loan_type: str) -> Dict[str, Any]:
        """Generate loan offer using traditional logic"""
        base_amounts = {
            'personal': 500000,
            'home': 5000000,
            'vehicle': 1000000
        }

        base_rates = {
            'personal': 12.0,
            'home': 8.5,
            'vehicle': 10.5
        }

        # Adjust based on credit score
        if credit_score >= 750:
            rate_adjustment = -2.0
            amount_multiplier = 1.5
        elif credit_score >= 650:
            rate_adjustment = 0.0
            amount_multiplier = 1.0
        else:
            rate_adjustment = 3.0
            amount_multiplier = 0.7

        base_amount = base_amounts.get(loan_type, 500000)
        base_rate = base_rates.get(loan_type, 12.0)

        recommended_amount = int(base_amount * amount_multiplier)
        interest_rate = base_rate + rate_adjustment

        return {
            'eligible': credit_score >= 580,
            'loan_type': loan_type,
            'recommended_amount': recommended_amount,
            'interest_rate': round(interest_rate, 2),
            'max_tenure_months': 60,
            'estimated_emi': self._calculate_emi(recommended_amount, interest_rate, 36),
            'approval_probability': min(0.95, (credit_score - 300) / 550),
            'conditions': ["Income verification", "Identity verification"],
            'explanation': f"Traditional offer based on credit score of {credit_score}"
        }

    def detect_fraud(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect fraud using best available AI system"""
        try:
            # Use Granite AI if available
            if self.granite_ai:
                return self.granite_ai.detect_fraud(event_data)

            # Fallback fraud detection
            return self._detect_fraud_traditional(event_data)

        except Exception as e:
            self.logger.error(f"Error in fraud detection: {e}")
            return {'fraud_probability': 0.1, 'recommended_action': 'Monitor'}

    def _detect_fraud_traditional(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Traditional rule-based fraud detection"""
        fraud_score = 0.0
        indicators = []

        # Velocity check
        if event_data.get('transactions_today', 0) > 15:
            fraud_score += 0.4
            indicators.append("High transaction velocity")

        # Amount check
        amount = event_data.get('amount', 0)
        avg_amount = event_data.get('average_amount', 1000)
        if amount > avg_amount * 3:
            fraud_score += 0.3
            indicators.append("Unusual transaction amount")

        # Time check
        hour = event_data.get('hour', 12)
        if hour < 6 or hour > 22:
            fraud_score += 0.2
            indicators.append("Unusual transaction time")

        fraud_probability = min(1.0, fraud_score)

        if fraud_probability > 0.7:
            action = "Block transaction"
        elif fraud_probability > 0.4:
            action = "Require verification"
        else:
            action = "Allow transaction"

        return {
            'fraud_probability': fraud_probability,
            'risk_indicators': indicators,
            'recommended_action': action,
            'confidence': 0.75,
            'analysis': f"Traditional analysis found {len(indicators)} risk factors"
        }

    def _score_to_category_name(self, score: int) -> str:
        """Convert numeric score to category name"""
        if score >= 800:
            return "Exceptional"
        elif score >= 740:
            return "Very Good"
        elif score >= 670:
            return "Good"
        elif score >= 580:
            return "Fair"
        else:
            return "Poor"

    def _score_to_risk_level(self, score: int) -> str:
        """Convert score to risk level"""
        if score >= 740:
            return "Low"
        elif score >= 670:
            return "Medium-Low"
        elif score >= 580:
            return "Medium"
        else:
            return "High"

    def _estimate_pd_from_score(self, score: int) -> float:
        """Estimate probability of default from credit score"""
        if score >= 750:
            return 0.02
        elif score >= 700:
            return 0.05
        elif score >= 650:
            return 0.08
        elif score >= 600:
            return 0.15
        else:
            return 0.25

    def _analyze_factors_traditional(self, user_data: Dict[str, Any]) -> List[str]:
        """Analyze key factors using traditional logic"""
        factors = []

        # Payment history
        payment_score = user_data.get('payment_history_score', 0)
        if payment_score > 1.5:
            factors.append("Excellent payment history")
        elif payment_score < 0:
            factors.append("Poor payment history")

        # Income
        income = user_data.get('annual_income', 0)
        if income > 1000000:
            factors.append("High income stability")
        elif income < 300000:
            factors.append("Limited income")

        # Debt ratio
        dti = user_data.get('debt_to_income_ratio', 0.5)
        if dti < 0.2:
            factors.append("Low debt burden")
        elif dti > 0.5:
            factors.append("High debt-to-income ratio")

        return factors

    def _generate_recommendations(self, user_data: Dict[str, Any], score: int) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        if score < 650:
            recommendations.append("Focus on improving payment history")

        dti = user_data.get('debt_to_income_ratio', 0.5)
        if dti > 0.4:
            recommendations.append("Reduce existing debt burden")

        utilization = user_data.get('credit_utilization_ratio', 0.5)
        if utilization > 0.3:
            recommendations.append("Lower credit utilization below 30%")

        if not recommendations:
            recommendations.append("Continue maintaining good financial habits")

        return recommendations

    def _calculate_emi(self, principal: float, annual_rate: float, tenure_months: int) -> float:
        """Calculate EMI using standard formula"""
        monthly_rate = annual_rate / (12 * 100)
        if monthly_rate == 0:
            return principal / tenure_months

        emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months / ((1 + monthly_rate) ** tenure_months - 1)
        return round(emi, 2)

    def _generate_blockchain_hash(self, user_id: str, score: int) -> str:
        """Generate blockchain hash for credit score"""
        timestamp = datetime.utcnow().isoformat()
        data = f"{user_id}:{score}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and performance metrics"""
        return {
            'primary_ai': self.primary_ai,
            'granite_available': self.granite_ai is not None,
            'traditional_ml_available': self.credit_model is not None,
            'processed_requests': self.processed_requests,
            'granite_requests': self.granite_requests,
            'fallback_requests': self.fallback_requests,
            'granite_usage_percentage': (self.granite_requests / max(1, self.processed_requests)) * 100,
            'system_health': 'Good' if self.granite_ai else 'Limited'
        }

# Example usage
if __name__ == "__main__":
    # Initialize assistant
    assistant = NexaCredFinancialAssistant()

    # Test data
    test_customer = {
        'annual_income': 850000,
        'debt_to_income_ratio': 0.25,
        'payment_history_score': 1.8,
        'credit_utilization_ratio': 0.15,
        'age': 32,
        'employment_tenure_months': 48
    }

    # Test credit scoring
    print("Testing Credit Scoring...")
    result = assistant.get_score("TEST_USER_001", test_customer)
    print(f"Credit Score: {result['credit_score']}")
    print(f"Category: {result['category']}")
    print(f"Risk Level: {result['risk_level']}")

    # Test loan offer
    print("\nTesting Loan Offer...")
    offer = assistant.generate_offer("TEST_USER_001", test_customer, "personal")
    if offer.get('eligible'):
        print(f"Eligible: {offer['eligible']}")
        print(f"Amount: ₹{offer['recommended_amount']:,}")
        print(f"Rate: {offer['interest_rate']}%")

    # Test fraud detection
    print("\nTesting Fraud Detection...")
    transaction = {
        'amount': 25000,
        'average_amount': 5000,
        'transactions_today': 8,
        'hour': 14
    }
    fraud_result = assistant.detect_fraud(transaction)
    print(f"Fraud Probability: {fraud_result['fraud_probability']:.2%}")
    print(f"Action: {fraud_result['recommended_action']}")

    # System status
    print(f"\nSystem Status: {assistant.get_system_status()}")
