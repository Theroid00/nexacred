#!/usr/bin/env python3
"""
IBM Granite Integration for Financial AI
========================================

Integrates IBM Granite models for:
- RAG-based financial assistant chatbot
- Automatic risk report generation
- Loan recommendation systems
- Natural language explanations
"""

import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import sqlite3
import os

class IBMGraniteFinancialAI:
    """
    IBM Granite model integration for financial AI tasks
    """

    def __init__(self, api_key: str = None, model_id: str = "ibm/granite-13b-instruct-v2"):
        """
        Initialize IBM Granite AI system

        Args:
            api_key: IBM Cloud API key
            model_id: Granite model identifier
        """
        self.api_key = api_key or os.getenv('IBM_CLOUD_API_KEY')
        self.model_id = model_id
        self.base_url = "https://us-south.ml.cloud.ibm.com/ml/v1"
        self.access_token = None
        self.knowledge_base = self._initialize_knowledge_base()

    def _get_access_token(self) -> str:
        """Get IBM Cloud access token"""
        if not self.api_key:
            # For development/testing, return mock token
            return "mock_token_for_testing"

        url = "https://iam.cloud.ibm.com/identity/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = f"grant_type=urn:iam:grants:api-key&apikey={self.api_key}"

        try:
            response = requests.post(url, headers=headers, data=data)
            return response.json()["access_token"]
        except Exception as e:
            print(f"Warning: Could not get IBM access token: {e}")
            return "mock_token_for_testing"

    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize financial knowledge base for RAG"""
        return {
            "credit_scoring": {
                "factors": [
                    "Payment history (35% of score)",
                    "Credit utilization (30% of score)",
                    "Length of credit history (15% of score)",
                    "Credit mix (10% of score)",
                    "New credit inquiries (10% of score)"
                ],
                "score_ranges": {
                    "Poor": "300-579",
                    "Fair": "580-669",
                    "Good": "670-739",
                    "Very Good": "740-799",
                    "Excellent": "800-850"
                }
            },
            "loan_policies": {
                "personal_loan": {
                    "min_score": 580,
                    "max_amount": 50000,
                    "interest_rate_range": "6-36%"
                },
                "mortgage": {
                    "min_score": 620,
                    "down_payment_min": "3%",
                    "interest_rate_range": "3-8%"
                },
                "auto_loan": {
                    "min_score": 560,
                    "interest_rate_range": "4-20%"
                }
            },
            "risk_factors": [
                "High debt-to-income ratio",
                "Recent missed payments",
                "High credit utilization",
                "Multiple recent credit inquiries",
                "Short credit history",
                "Limited credit mix"
            ]
        }

    def chat_with_financial_assistant(self, user_message: str, context: Dict[str, Any] = None) -> str:
        """
        RAG-based financial assistant chat

        Args:
            user_message: User's question or message
            context: Additional context (credit score, financial data, etc.)

        Returns:
            AI assistant response
        """
        # Retrieve relevant knowledge
        relevant_info = self._retrieve_relevant_knowledge(user_message)

        # Construct prompt with RAG context
        system_prompt = f"""You are a knowledgeable financial advisor assistant. Use the following knowledge base to provide accurate, helpful financial advice.

Knowledge Base:
{json.dumps(relevant_info, indent=2)}

User Context: {json.dumps(context or {}, indent=2)}

Guidelines:
- Provide accurate, actionable financial advice
- Reference specific credit score ranges and factors
- Be helpful but not overly promotional
- Mention risks and considerations
- Keep responses conversational but professional
"""

        prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"

        # Generate response using Granite (or mock for testing)
        response = self._generate_granite_response(prompt)

        return response

    def generate_risk_report(self, credit_data: Dict[str, Any], credit_prediction: int,
                           confidence: float) -> Dict[str, Any]:
        """
        Generate comprehensive risk assessment report

        Args:
            credit_data: Customer's financial data
            credit_prediction: ML model prediction (0=Poor, 1=Standard, 2=Good)
            confidence: Model confidence score

        Returns:
            Structured risk report
        """
        # Map prediction to category
        categories = {0: "Poor", 1: "Standard", 2: "Good"}
        credit_category = categories.get(credit_prediction, "Unknown")

        # Generate risk assessment prompt
        prompt = f"""Generate a comprehensive risk assessment report for a credit application.

Customer Financial Profile:
{json.dumps(credit_data, indent=2)}

ML Model Assessment:
- Credit Category: {credit_category}
- Model Confidence: {confidence:.2%}

Please provide a detailed risk report covering:
1. Overall Risk Level (Low/Medium/High)
2. Key Risk Factors
3. Mitigating Factors  
4. Loan Recommendations
5. Monitoring Requirements

Format as structured JSON with clear explanations."""

        # Generate report using Granite
        granite_response = self._generate_granite_response(prompt)

        # Structure the response
        risk_report = {
            "timestamp": datetime.now().isoformat(),
            "customer_id": credit_data.get("Customer_ID", "Unknown"),
            "credit_category": credit_category,
            "model_confidence": confidence,
            "overall_risk_level": self._determine_risk_level(credit_prediction, confidence),
            "ai_analysis": granite_response,
            "key_metrics": self._extract_key_metrics(credit_data),
            "recommendations": self._generate_loan_recommendations(credit_prediction, credit_data)
        }

        return risk_report

    def generate_loan_application_analysis(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze loan application and generate decision support
        """
        prompt = f"""Analyze this loan application and provide decision support.

Application Data:
{json.dumps(application_data, indent=2)}

Financial Knowledge:
{json.dumps(self.knowledge_base['loan_policies'], indent=2)}

Provide analysis covering:
1. Application Strengths
2. Areas of Concern
3. Approval Recommendation (Approve/Conditional/Deny)
4. Suggested Terms (if approved)
5. Required Documentation
6. Risk Mitigation Strategies

Be thorough and professional."""

        analysis = self._generate_granite_response(prompt)

        return {
            "analysis_date": datetime.now().isoformat(),
            "application_id": application_data.get("application_id", "Unknown"),
            "ai_analysis": analysis,
            "risk_score": self._calculate_application_risk_score(application_data),
            "decision_factors": self._extract_decision_factors(application_data)
        }

    def _retrieve_relevant_knowledge(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant information from knowledge base"""
        # Simple keyword-based retrieval (would use embeddings in production)
        relevant_sections = {}

        query_lower = query.lower()

        if any(term in query_lower for term in ['credit score', 'score', 'rating']):
            relevant_sections['credit_scoring'] = self.knowledge_base['credit_scoring']

        if any(term in query_lower for term in ['loan', 'mortgage', 'borrow']):
            relevant_sections['loan_policies'] = self.knowledge_base['loan_policies']

        if any(term in query_lower for term in ['risk', 'danger', 'concern']):
            relevant_sections['risk_factors'] = self.knowledge_base['risk_factors']

        return relevant_sections

    def _generate_granite_response(self, prompt: str) -> str:
        """
        Generate response using IBM Granite model
        (Mock implementation for development)
        """
        # In production, this would call the actual IBM Granite API
        if not self.access_token:
            self.access_token = self._get_access_token()

        # Mock response for development/testing
        if "risk report" in prompt.lower():
            return self._generate_mock_risk_analysis()
        elif "loan application" in prompt.lower():
            return self._generate_mock_loan_analysis()
        else:
            return self._generate_mock_chat_response(prompt)

    def _generate_mock_risk_analysis(self) -> str:
        """Generate mock risk analysis for testing"""
        return """Based on the financial profile analysis:

**Overall Risk Assessment: MEDIUM**

**Key Risk Factors:**
- Credit utilization above optimal threshold
- Limited credit history length
- Recent credit inquiries indicating credit-seeking behavior

**Mitigating Factors:**
- Stable employment history
- Consistent payment patterns
- Adequate debt-to-income ratio

**Recommendations:**
- Approve with standard terms
- Monthly payment monitoring recommended
- Credit counseling resources provided
- 6-month performance review suggested

**Monitoring Requirements:**
- Track payment performance monthly
- Monitor credit utilization changes
- Review employment status quarterly"""

    def _generate_mock_loan_analysis(self) -> str:
        """Generate mock loan analysis for testing"""
        return """**Loan Application Analysis**

**Strengths:**
- Good payment history demonstrates reliability
- Stable income source provides repayment capacity
- Reasonable debt-to-income ratio

**Areas of Concern:**
- Credit utilization could be improved
- Limited savings for emergency reserves

**Recommendation: CONDITIONAL APPROVAL**

**Suggested Terms:**
- Interest rate: 8.5% (standard tier)
- Term: 36 months
- Monthly payment monitoring

**Required Documentation:**
- Employment verification letter
- Bank statements (3 months)
- Proof of income

**Risk Mitigation:**
- Debt consolidation counseling
- Automatic payment setup
- Quarterly check-ins"""

    def _generate_mock_chat_response(self, prompt: str) -> str:
        """Generate mock chat response for testing"""
        if "credit score" in prompt.lower():
            return """Your credit score is influenced by five main factors:

1. **Payment History (35%)** - Always pay on time
2. **Credit Utilization (30%)** - Keep below 30% of limits  
3. **Credit History Length (15%)** - Longer is better
4. **Credit Mix (10%)** - Mix of cards, loans, etc.
5. **New Credit (10%)** - Limit new applications

To improve your score, focus on timely payments and reducing credit card balances. Would you like specific recommendations based on your current profile?"""

        return "I'm here to help with your financial questions. Could you please provide more details about what you'd like to know?"

    def _determine_risk_level(self, prediction: int, confidence: float) -> str:
        """Determine overall risk level"""
        if prediction == 2 and confidence > 0.8:
            return "Low"
        elif prediction == 1 or (prediction == 2 and confidence <= 0.8):
            return "Medium"
        else:
            return "High"

    def _extract_key_metrics(self, credit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key financial metrics"""
        return {
            "debt_to_income": credit_data.get("Debt_to_Income_Ratio", "N/A"),
            "credit_utilization": credit_data.get("Credit_Utilization_Ratio", "N/A"),
            "annual_income": credit_data.get("Annual_Income", "N/A"),
            "credit_accounts": credit_data.get("Num_Credit_Card", "N/A"),
            "payment_history": credit_data.get("Num_of_Delayed_Payment", "N/A")
        }

    def _generate_loan_recommendations(self, prediction: int, credit_data: Dict[str, Any]) -> List[str]:
        """Generate loan recommendations based on credit profile"""
        recommendations = []

        if prediction == 2:  # Good credit
            recommendations = [
                "Qualified for premium loan products",
                "Competitive interest rates available",
                "Consider refinancing existing debt",
                "Eligible for higher credit limits"
            ]
        elif prediction == 1:  # Standard credit
            recommendations = [
                "Standard loan terms available",
                "Focus on payment history improvement",
                "Consider secured loan options",
                "Build credit before major purchases"
            ]
        else:  # Poor credit
            recommendations = [
                "Consider credit repair services",
                "Secured credit cards recommended",
                "Co-signer may be required",
                "Focus on debt reduction first"
            ]

        return recommendations

    def _calculate_application_risk_score(self, application_data: Dict[str, Any]) -> float:
        """Calculate numerical risk score for application"""
        # Simple risk scoring algorithm
        base_score = 0.5

        # Adjust based on available factors
        if application_data.get("Credit_Score") == "Good":
            base_score -= 0.2
        elif application_data.get("Credit_Score") == "Poor":
            base_score += 0.3

        return max(0.0, min(1.0, base_score))

    def _extract_decision_factors(self, application_data: Dict[str, Any]) -> List[str]:
        """Extract key decision factors from application"""
        factors = []

        if application_data.get("Annual_Income"):
            factors.append(f"Annual Income: ${application_data['Annual_Income']:,}")

        if application_data.get("Credit_Score"):
            factors.append(f"Credit Category: {application_data['Credit_Score']}")

        if application_data.get("Employment_Type"):
            factors.append(f"Employment: {application_data['Employment_Type']}")

        return factors
