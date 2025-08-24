#!/usr/bin/env python3
"""NexaCred Financial Assistant

Provides high-level orchestration utilities wrapping the heuristic CreditScoreModel
and (optionally) IBM Granite integration. This replaces multiple previously
inconsistent assistant implementations with a single lightweight module.

Public class used by tests / backend:
  - NexaCredFinancialAssistant

Methods required by tests:
  - get_score(user_id, customer_data)
  - generate_offer(user_id, customer_data, loan_type)
  - detect_fraud(transaction_data)
  - get_system_status()

Design goals:
  * Pure-Python & dependency‑light (Granite optional)
  * Deterministic given random_state
  * Avoid duplication (reuse train_model + credit_utils)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import os
import math
from datetime import datetime, timezone

try:  # package context
    from .train_model import CreditScoreModel
    from .granite_agents import IBMGraniteFinancialAI  # optional
except ImportError:  # script context
    from train_model import CreditScoreModel  # type: ignore
    try:
        from granite_agents import IBMGraniteFinancialAI  # type: ignore
    except Exception:  # Granite optional
        IBMGraniteFinancialAI = None  # type: ignore


CATEGORY_NAMES = ['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
RISK_LEVELS = {
    0: 'High',
    1: 'Elevated',
    2: 'Moderate',
    3: 'Low',
    4: 'Very Low',
}


@dataclass
class ScoreResult:
    user_id: str
    category: str
    category_index: int
    credit_score: int
    risk_level: str
    probability_distribution: List[float]
    calculated_at: str


class NexaCredFinancialAssistant:
    """Facade exposing scoring, loan recommendation, and fraud heuristics."""

    def __init__(self, random_state: int = 42, enable_granite: bool | None = None):
        self.random_state = random_state
        self.model = CreditScoreModel(random_state=random_state)
        # Train model once on synthetic data to satisfy interface expectations
        X, y = self.model.generate_synthetic_data(n_samples=300, n_features=15)
        self.model.train_model(X, y)

        if enable_granite is None:
            # Auto-enable if environment variable / dependency available
            enable_granite = bool(os.getenv('ENABLE_GRANITE', '')) and 'IBMGraniteFinancialAI' in globals()
        self.granite: Optional[object] = None
        if enable_granite and 'IBMGraniteFinancialAI' in globals() and IBMGraniteFinancialAI is not None:  # type: ignore
            try:
                self.granite = IBMGraniteFinancialAI()  # type: ignore
            except Exception:
                self.granite = None

    # ------------------------------------------------------------------
    # Core credit scoring
    # ------------------------------------------------------------------
    def get_score(self, user_id: str, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        cat_idx = int(self.model.predict_credit_score_category(customer_data)[0])
        score = self.model.convert_category_to_score([cat_idx])[0]
        proba = self.model.predict_credit_score_proba(customer_data)[0].tolist()
        result = ScoreResult(
            user_id=user_id,
            category=CATEGORY_NAMES[cat_idx],
            category_index=cat_idx,
            credit_score=score,
            risk_level=RISK_LEVELS.get(cat_idx, 'Unknown'),
            probability_distribution=proba,
            calculated_at=datetime.now(timezone.utc).isoformat(),
        )
        return {
            'user_id': result.user_id,
            'credit_score': result.credit_score,
            'category': result.category,
            'risk_level': result.risk_level,
            'probabilities': result.probability_distribution,
            'calculated_at': result.calculated_at,
        }

    # ------------------------------------------------------------------
    # Loan recommendation
    # ------------------------------------------------------------------
    def generate_offer(self, user_id: str, customer_data: Dict[str, Any], loan_type: str) -> Dict[str, Any]:
        score_info = self.get_score(user_id, customer_data)
        score = score_info['credit_score']
        category = score_info['category']

        base_limits = {
            'personal': 500000,
            'auto': 1200000,
            'mortgage': 20000000,
            'education': 800000,
        }
        base_amount = base_limits.get(loan_type.lower(), 300000)

        # Scale by score (linear within 300-900)
        scale = (score - 300) / 600  # 0..1
        recommended_amount = int(base_amount * (0.4 + 0.6 * scale))

        # Interest rate heuristic (inverse to score)
        rate = round(18 - 8 * scale, 2)
        term_months = 36 if loan_type == 'personal' else 120 if loan_type == 'auto' else 240

        # Simple EMI: P * r / (1 - (1+r)^-n) with monthly r
        monthly_r = rate / 1200
        n = term_months
        if monthly_r == 0:
            emi = recommended_amount / n
        else:
            emi = recommended_amount * monthly_r / (1 - (1 + monthly_r) ** -n)

        eligible = category not in ['Poor']

        return {
            'user_id': user_id,
            'loan_type': loan_type,
            'eligible': eligible,
            'recommended_amount': recommended_amount if eligible else 0,
            'interest_rate': rate,
            'term_months': term_months,
            'estimated_emi': round(emi, 2),
            'category': category,
        }

    # ------------------------------------------------------------------
    # Fraud heuristics
    # ------------------------------------------------------------------
    def detect_fraud(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        amount = float(transaction.get('amount', 0))
        avg = float(transaction.get('average_amount', max(amount, 1)))
        ratio = amount / max(avg, 1)
        velocity = int(transaction.get('transactions_today', 1))
        hour = int(transaction.get('transaction_hour', 12))
        new_location = bool(transaction.get('new_location', False))

        # Basic risk components (all float, clamp via exp transform)
        score = 0.0
        score += min(1.5, math.log1p(max(0, ratio - 1)))  # large deviation
        score += 0.3 if velocity > 10 else 0.1 if velocity > 5 else 0
        if hour < 6 or hour > 22:
            score += 0.25
        if new_location:
            score += 0.4

        fraud_prob = max(0.0, min(1.0, 1 - math.exp(-score)))
        action = 'review' if fraud_prob > 0.4 else 'allow'
        if fraud_prob > 0.75:
            action = 'block'

        return {
            'fraud_probability': fraud_prob,
            'recommended_action': action,
            'anomaly_ratio': ratio,
            'velocity': velocity,
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_system_status(self) -> Dict[str, Any]:
        return {
            'primary_ai': 'Granite' if self.granite else 'HeuristicModel',
            'granite_available': bool(self.granite),
            'model_version': self.model.model_version,
            'system_health': 'OK',
        }


__all__ = ["NexaCredFinancialAssistant"]
