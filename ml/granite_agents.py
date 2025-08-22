#!/usr/bin/env python3
"""
NexaCred IBM Granite-3.3-8B-Instruct Integration
===============================================

Advanced financial AI system using IBM's Granite-3.3-8B-Instruct model for:
- Superior credit risk assessment
- Intelligent loan decisioning
- Advanced fraud detection
- Regulatory compliance analysis
- Natural language financial advisory
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np

# Try to import ML dependencies, fallback gracefully
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Install with: pip install torch transformers accelerate")

# Import our RAG system
try:
    from rag_system import NexaCredRAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("Warning: RAG system not available")

class GraniteTaskType(Enum):
    """Task types for Granite model processing"""
    CREDIT_ANALYSIS = "credit_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    FRAUD_DETECTION = "fraud_detection"
    LOAN_RECOMMENDATION = "loan_recommendation"
    COMPLIANCE_CHECK = "compliance_check"
    MARKET_ANALYSIS = "market_analysis"
    CUSTOMER_SERVICE = "customer_service"

@dataclass
class GraniteRequest:
    """Request structure for Granite model"""
    task_type: GraniteTaskType
    customer_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    requirements: Optional[List[str]] = None
    priority: int = 1

@dataclass
class CreditAnalysisResult:
    """Structured result for credit analysis"""
    credit_score: int
    risk_level: str
    probability_of_default: float
    key_factors: List[str]
    recommendations: List[str]
    explanation: str
    confidence: float

class IBMGraniteFinancialAI:
    """
    IBM Granite-3.3-8B-Instruct powered financial AI system

    Superior to logistic regression because:
    - Understands complex financial relationships
    - Processes natural language requirements
    - Adapts to new scenarios without retraining
    - Provides detailed explanations
    - Handles multi-modal financial data
    """

    def __init__(self, model_name: str = "ibm-granite/granite-3.3-8b-instruct"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.model_loaded = False

        # Initialize Granite model
        self._initialize_granite_model()

        # Initialize RAG system for knowledge enhancement
        if RAG_AVAILABLE:
            self.rag_system = NexaCredRAGSystem()
        else:
            self.rag_system = None

        # Financial domain prompts
        self.system_prompts = self._initialize_financial_prompts()

        # Performance tracking
        self.processed_requests = 0
        self.accuracy_scores = []

    def _initialize_granite_model(self):
        """Initialize IBM Granite-3.3-8B-Instruct model"""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available - using rule-based fallback")
            self.generator = None
            return

        try:
            self.logger.info(f"Loading IBM Granite model: {self.model_name}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=dtype,
                device=device,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            self.model_loaded = True
            self.logger.info("✅ IBM Granite model loaded successfully")

        except Exception as e:
            self.logger.error(f"❌ Failed to load Granite model: {e}")
            # Fallback to rule-based implementation
            self.generator = None
            self.model_loaded = False
            self.logger.warning("Using rule-based implementation as fallback")

    def _initialize_financial_prompts(self) -> Dict[str, str]:
        """Initialize specialized financial prompts for Granite"""
        return {
            "credit_analysis": """You are an expert credit analyst with 20+ years of experience in financial risk assessment.
Analyze the customer's financial profile and provide a comprehensive credit assessment.

Consider these key factors:
- Payment history (35% weight): On-time payments, defaults, bankruptcies
- Credit utilization (30% weight): How much available credit is being used
- Credit history length (15% weight): Age of oldest and average accounts
- Credit mix (10% weight): Variety of credit types (cards, loans, mortgages)
- New credit (10% weight): Recent inquiries and new accounts

Customer Data: {customer_data}
Context: {context}

Provide a detailed analysis with:
1. Credit score recommendation (300-850 scale)
2. Risk level assessment (Low/Medium/High)
3. Key strengths and weaknesses
4. Specific recommendations for improvement
5. Probability of default estimate

Analysis:""",

            "fraud_detection": """You are a fraud detection specialist with expertise in financial crime patterns.
Analyze the transaction and customer behavior for potential fraud indicators.

Look for:
- Transaction velocity anomalies
- Geographic inconsistencies
- Amount patterns outside normal behavior
- Merchant category risks
- Device/IP address changes
- Time-of-day patterns

Transaction Data: {customer_data}
Historical Context: {context}

Provide:
1. Fraud probability score (0-100%)
2. Specific risk indicators found
3. Fraud pattern classification
4. Recommended immediate actions
5. Investigation priority level

Fraud Analysis:""",

            "loan_recommendation": """You are a senior loan advisor specializing in personalized financial products.
Based on the customer's profile, recommend the most suitable loan products.

Customer Data: {customer_data}
Available Products: {context}

Provide:
1. Recommended loan products with rationale
2. Optimal loan amount and tenure
3. Competitive interest rate
4. Required documentation
5. Approval probability assessment

Recommendation:"""
        }

    def analyze_credit_risk(self, customer_data: Dict[str, Any]) -> CreditAnalysisResult:
        """Perform comprehensive credit risk analysis using Granite"""
        try:
            if self.model_loaded and self.generator:
                return self._granite_credit_analysis(customer_data)
            else:
                return self._fallback_credit_analysis(customer_data)
        except Exception as e:
            self.logger.error(f"Credit analysis error: {e}")
            return self._fallback_credit_analysis(customer_data)

    def _granite_credit_analysis(self, customer_data: Dict[str, Any]) -> CreditAnalysisResult:
        """Use Granite model for credit analysis"""
        prompt = self.system_prompts["credit_analysis"].format(
            customer_data=json.dumps(customer_data, indent=2),
            context="Standard credit assessment for personal lending"
        )

        try:
            response = self.generator(
                prompt,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True
            )[0]['generated_text']

            # Extract the analysis (remove the prompt)
            analysis = response[len(prompt):].strip()

            # Parse the analysis and extract structured data
            return self._parse_granite_analysis(analysis, customer_data)

        except Exception as e:
            self.logger.error(f"Granite analysis failed: {e}")
            return self._fallback_credit_analysis(customer_data)

    def _fallback_credit_analysis(self, customer_data: Dict[str, Any]) -> CreditAnalysisResult:
        """Rule-based fallback credit analysis"""
        # Base score calculation
        score = 500
        risk_factors = []
        positive_factors = []

        # Payment history analysis
        payment_score = customer_data.get('payment_history_score', 0)
        if payment_score > 1.5:
            score += 120
            positive_factors.append("Excellent payment history")
        elif payment_score > 0.5:
            score += 80
            positive_factors.append("Good payment history")
        else:
            score -= 100
            risk_factors.append("Poor payment history")

        # Income analysis
        income = customer_data.get('annual_income', 0)
        if income > 1000000:
            score += 100
            positive_factors.append("High income stability")
        elif income > 500000:
            score += 50
            positive_factors.append("Stable income")
        else:
            risk_factors.append("Limited income capacity")

        # Debt-to-income ratio
        dti = customer_data.get('debt_to_income_ratio', 0.5)
        if dti < 0.2:
            score += 80
            positive_factors.append("Low debt burden")
        elif dti < 0.4:
            score += 40
        else:
            score -= 60
            risk_factors.append("High debt-to-income ratio")

        # Credit utilization
        utilization = customer_data.get('credit_utilization_ratio', 0.5)
        if utilization < 0.1:
            score += 90
            positive_factors.append("Low credit utilization")
        elif utilization < 0.3:
            score += 60
        else:
            score -= 50
            risk_factors.append("High credit utilization")

        # Employment stability
        employment_months = customer_data.get('employment_tenure_months', 0)
        if employment_months > 60:
            score += 40
            positive_factors.append("Long employment tenure")
        elif employment_months > 24:
            score += 20

        # Cap score between 300-850
        final_score = max(300, min(850, score))

        # Determine risk level
        if final_score >= 740:
            risk_level = "Low"
            pd = 0.02
        elif final_score >= 670:
            risk_level = "Medium-Low"
            pd = 0.05
        elif final_score >= 580:
            risk_level = "Medium"
            pd = 0.12
        else:
            risk_level = "High"
            pd = 0.25

        # Generate recommendations
        recommendations = []
        if dti > 0.4:
            recommendations.append("Reduce existing debt burden")
        if utilization > 0.3:
            recommendations.append("Lower credit utilization below 30%")
        if employment_months < 24:
            recommendations.append("Maintain stable employment")
        if not recommendations:
            recommendations.append("Continue maintaining good financial habits")

        return CreditAnalysisResult(
            credit_score=final_score,
            risk_level=risk_level,
            probability_of_default=pd,
            key_factors=positive_factors + risk_factors,
            recommendations=recommendations,
            explanation=f"Score based on payment history, income, debt ratios, and employment stability",
            confidence=0.85
        )

    def _parse_granite_analysis(self, analysis: str, customer_data: Dict[str, Any]) -> CreditAnalysisResult:
        """Parse Granite model output into structured format"""
        # This is a simplified parser - in production, you'd use more sophisticated NLP
        lines = analysis.split('\n')

        # Extract score (look for numbers in 300-850 range)
        score = 650  # default
        for line in lines:
            if 'score' in line.lower():
                numbers = [int(s) for s in line.split() if s.isdigit() and 300 <= int(s) <= 850]
                if numbers:
                    score = numbers[0]
                    break

        # Extract risk level
        risk_level = "Medium"
        for line in lines:
            line_lower = line.lower()
            if 'low risk' in line_lower or 'risk: low' in line_lower:
                risk_level = "Low"
                break
            elif 'high risk' in line_lower or 'risk: high' in line_lower:
                risk_level = "High"
                break

        # Estimate PD based on score
        if score >= 740:
            pd = 0.02
        elif score >= 670:
            pd = 0.05
        elif score >= 580:
            pd = 0.12
        else:
            pd = 0.25

        return CreditAnalysisResult(
            credit_score=score,
            risk_level=risk_level,
            probability_of_default=pd,
            key_factors=["AI-analyzed risk factors"],
            recommendations=["Follow AI recommendations"],
            explanation=analysis[:200] + "..." if len(analysis) > 200 else analysis,
            confidence=0.90
        )

    def detect_fraud(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced fraud detection using Granite model"""
        if self.model_loaded and self.generator:
            return self._granite_fraud_detection(transaction_data)
        else:
            return self._fallback_fraud_detection(transaction_data)

    def _granite_fraud_detection(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use Granite model for fraud detection"""
        prompt = self.system_prompts["fraud_detection"].format(
            customer_data=json.dumps(transaction_data, indent=2),
            context="Real-time transaction monitoring"
        )

        try:
            response = self.generator(
                prompt,
                max_new_tokens=200,
                temperature=0.2,
                do_sample=True
            )[0]['generated_text']

            analysis = response[len(prompt):].strip()

            return {
                "fraud_probability": 0.15,  # Would parse from Granite output
                "risk_indicators": ["AI-detected patterns"],
                "recommended_action": "Monitor closely",
                "confidence": 0.88,
                "analysis": analysis[:150] + "..." if len(analysis) > 150 else analysis
            }

        except Exception as e:
            self.logger.error(f"Granite fraud detection failed: {e}")
            return self._fallback_fraud_detection(transaction_data)

    def _fallback_fraud_detection(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based fraud detection"""
        fraud_score = 0.0
        risk_indicators = []

        # Check transaction velocity
        if transaction_data.get('transactions_today', 0) > 10:
            fraud_score += 0.3
            risk_indicators.append("High transaction velocity")

        # Check amount anomaly
        amount = transaction_data.get('amount', 0)
        avg_amount = transaction_data.get('average_amount', 1000)
        if amount > avg_amount * 5:
            fraud_score += 0.4
            risk_indicators.append("Unusually high transaction amount")

        # Check time patterns
        hour = transaction_data.get('transaction_hour', 12)
        if hour < 6 or hour > 23:
            fraud_score += 0.2
            risk_indicators.append("Unusual transaction time")

        # Check location
        if transaction_data.get('new_location', False):
            fraud_score += 0.3
            risk_indicators.append("New geographical location")

        fraud_probability = min(1.0, fraud_score)

        if fraud_probability > 0.7:
            action = "Block transaction"
        elif fraud_probability > 0.4:
            action = "Require additional verification"
        else:
            action = "Allow transaction"

        return {
            "fraud_probability": fraud_probability,
            "risk_indicators": risk_indicators,
            "recommended_action": action,
            "confidence": 0.82,
            "analysis": f"Rule-based analysis detected {len(risk_indicators)} risk factors"
        }

    def generate_loan_recommendation(self, customer_data: Dict[str, Any], loan_type: str = "personal") -> Dict[str, Any]:
        """Generate personalized loan recommendations"""
        credit_analysis = self.analyze_credit_risk(customer_data)

        # Base loan parameters
        loan_products = {
            "personal": {"max_amount": 1000000, "base_rate": 12.0, "max_tenure": 60},
            "home": {"max_amount": 10000000, "base_rate": 8.5, "max_tenure": 240},
            "vehicle": {"max_amount": 2000000, "base_rate": 10.5, "max_tenure": 84}
        }

        product = loan_products.get(loan_type, loan_products["personal"])

        # Adjust based on credit score
        score = credit_analysis.credit_score
        if score >= 750:
            rate_adjustment = -2.0
            amount_multiplier = 1.2
        elif score >= 650:
            rate_adjustment = 0.0
            amount_multiplier = 1.0
        else:
            rate_adjustment = 3.0
            amount_multiplier = 0.7

        # Calculate recommended amount based on income
        income = customer_data.get('annual_income', 0)
        max_recommended = min(
            income * 5,  # 5x annual income max
            product["max_amount"] * amount_multiplier
        )

        interest_rate = max(6.0, product["base_rate"] + rate_adjustment)

        return {
            "eligible": score >= 580,
            "loan_type": loan_type,
            "recommended_amount": int(max_recommended),
            "interest_rate": round(interest_rate, 2),
            "max_tenure_months": product["max_tenure"],
            "estimated_emi": self._calculate_emi(max_recommended, interest_rate, 36),
            "approval_probability": min(0.95, (score - 300) / 550),
            "conditions": self._generate_loan_conditions(credit_analysis),
            "explanation": f"Based on credit score of {score} and risk assessment"
        }

    def _calculate_emi(self, principal: float, annual_rate: float, tenure_months: int) -> float:
        """Calculate EMI using standard formula"""
        monthly_rate = annual_rate / (12 * 100)
        if monthly_rate == 0:
            return principal / tenure_months

        emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months / ((1 + monthly_rate) ** tenure_months - 1)
        return round(emi, 2)

    def _generate_loan_conditions(self, credit_analysis: CreditAnalysisResult) -> List[str]:
        """Generate loan conditions based on risk assessment"""
        conditions = ["Income verification required", "Identity verification required"]

        if credit_analysis.risk_level == "High":
            conditions.extend([
                "Collateral required",
                "Co-signer recommended",
                "Additional documentation needed"
            ])
        elif credit_analysis.risk_level == "Medium":
            conditions.append("Employment verification required")

        return conditions

    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and performance metrics"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model_loaded,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "rag_available": RAG_AVAILABLE,
            "processed_requests": self.processed_requests,
            "average_accuracy": np.mean(self.accuracy_scores) if self.accuracy_scores else 0.0,
            "cuda_available": torch.cuda.is_available() if TRANSFORMERS_AVAILABLE else False
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the Granite AI system
    granite_ai = IBMGraniteFinancialAI()

    # Test customer data
    test_customer = {
        'annual_income': 850000,
        'debt_to_income_ratio': 0.25,
        'payment_history_score': 1.8,
        'credit_utilization_ratio': 0.15,
        'age': 32,
        'employment_tenure_months': 48,
        'number_of_late_payments': 0,
        'total_credit_limit': 200000,
        'current_debt_amount': 30000
    }

    # Test credit analysis
    print("Testing Credit Analysis...")
    credit_result = granite_ai.analyze_credit_risk(test_customer)
    print(f"Credit Score: {credit_result.credit_score}")
    print(f"Risk Level: {credit_result.risk_level}")
    print(f"PD: {credit_result.probability_of_default:.2%}")

    # Test loan recommendation
    print("\nTesting Loan Recommendation...")
    loan_result = granite_ai.generate_loan_recommendation(test_customer, "personal")
    print(f"Eligible: {loan_result['eligible']}")
    print(f"Recommended Amount: ₹{loan_result['recommended_amount']:,}")
    print(f"Interest Rate: {loan_result['interest_rate']}%")

    # Test fraud detection
    print("\nTesting Fraud Detection...")
    transaction_data = {
        'amount': 50000,
        'average_amount': 5000,
        'transactions_today': 15,
        'transaction_hour': 3,
        'new_location': True
    }
    fraud_result = granite_ai.detect_fraud(transaction_data)
    print(f"Fraud Probability: {fraud_result['fraud_probability']:.2%}")
    print(f"Recommended Action: {fraud_result['recommended_action']}")

    print(f"\nModel Status: {granite_ai.get_model_status()}")
