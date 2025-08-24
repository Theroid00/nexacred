#!/usr/bin/env python3
"""Shared credit scoring utility functions.

Centralizes heuristic scoring logic, factor extraction, and recommendation text
previously duplicated across multiple modules (credit_scoring, train_model,
hybrid_credit_system). Keeping this lightweight avoids altering existing public
APIs while removing redundancy.
"""
from __future__ import annotations
from typing import Dict, Any, List

_NUMERIC_SCORE_MIN = 300
_NUMERIC_SCORE_MAX = 900  # train_model uses up to 900; credit_scoring caps at 850

# Base starting score used by heuristic components
_BASE_SCORE = 500


def calculate_base_credit_score(data: Dict[str, Any]) -> int:
    """Compute base heuristic credit score portion shared by modules.

    Factors considered (aligned with previous duplicated logic):
      - Annual Income
      - Debt-to-Income Ratio
      - Credit Utilization Ratio
      - Number of Late Payments (Num_of_Delayed_Payment / number_of_late_payments)
      - Age

    The function returns a bounded integer score (300-900) so callers can add
    module‑specific adjustments (e.g. payment history score, employment tenure)
    and re-bound if needed.
    """
    score = _BASE_SCORE
    # Income
    income = _to_float(data.get('annual_income') or data.get('Annual_Income'), 0)
    if income > 1_000_000:
        score += 110
    elif income > 600_000:
        score += 70
    elif income > 300_000:
        score += 40
    else:
        score -= 20

    # Debt-to-income ratio
    dti = _to_float(data.get('debt_to_income_ratio') or data.get('Debt_to_Income_Ratio'), 0.4)
    if dti < 0.2:
        score += 90
    elif dti < 0.35:
        score += 50
    elif dti > 0.6:
        score -= 80

    # Credit utilization
    utilization = _to_float(data.get('credit_utilization_ratio') or data.get('Credit_Utilization_Ratio'), 0.5)
    if utilization < 0.1:
        score += 80
    elif utilization < 0.3:
        score += 40
    elif utilization > 0.7:
        score -= 70

    # Late payments
    late = _to_float(data.get('number_of_late_payments') or data.get('Num_of_Delayed_Payment'), 0)
    if late == 0:
        score += 60
    elif late > 5:
        score -= 80
    elif late > 2:
        score -= 40

    # Age
    age = _to_float(data.get('age') or data.get('Age'), 30)
    if 25 <= age <= 65:
        score += 20

    return max(_NUMERIC_SCORE_MIN, min(_NUMERIC_SCORE_MAX, int(score)))


def extract_score_factors(data: Dict[str, Any]) -> List[str]:
    """Return qualitative factor descriptions based on base heuristics."""
    factors: List[str] = []
    income = _to_float(data.get('annual_income') or data.get('Annual_Income'), 0)
    if income > 1_000_000:
        factors.append('High income stability')

    dti = _to_float(data.get('debt_to_income_ratio') or data.get('Debt_to_Income_Ratio'), 0.4)
    if dti < 0.25:
        factors.append('Low debt-to-income ratio')
    elif dti > 0.55:
        factors.append('Elevated debt burden')

    util = _to_float(data.get('credit_utilization_ratio') or data.get('Credit_Utilization_Ratio'), 0.5)
    if util < 0.2:
        factors.append('Healthy credit utilization')
    elif util > 0.6:
        factors.append('High credit utilization')

    late = _to_float(data.get('number_of_late_payments') or data.get('Num_of_Delayed_Payment'), 0)
    if late == 0:
        factors.append('Perfect payment history')
    elif late > 3:
        factors.append('Multiple late payments')

    return factors


def generate_recommendations_three_class(prediction: int) -> List[str]:
    """Return recommendations for 3-class (0=Poor,1=Standard,2=Good) scheme with original phrasing."""
    if prediction == 0:  # Poor
        return [
            "Focus on making all payments on time to improve payment history",
            "Reduce credit utilization ratio below 30%",
            "Consider debt consolidation to manage outstanding debt",
            "Avoid opening new credit accounts in the near term",
            "Monitor credit report regularly for errors"
        ]
    if prediction == 1:  # Standard
        return [
            "Continue making timely payments to build positive payment history",
            "Keep credit utilization below 30% across all accounts",
            "Consider increasing credit limits to improve utilization ratio",
            "Maintain a good mix of credit types",
            "Pay down existing debt systematically"
        ]
    # Good
    return [
        "Maintain excellent payment history",
        "Keep credit utilization low (under 10% is ideal)",
        "Consider rewards credit cards for additional benefits",
        "Monitor credit regularly to maintain good standing",
        "You may qualify for premium financial products"
    ]


def score_to_category_five(score: int) -> int:
    """Map numeric score to 5-class category (0..4) used in train_model."""
    if score >= 800:
        return 4
    if score >= 740:
        return 3
    if score >= 670:
        return 2
    if score >= 580:
        return 1
    return 0


def score_to_probabilities_five(score: int):
    """Return probability distribution for 5 classes reflecting score bands."""
    # Distribution logic copied from train_model then centralized.
    if score >= 800:
        return [0.05, 0.05, 0.10, 0.30, 0.50]
    if score >= 740:
        return [0.05, 0.10, 0.20, 0.50, 0.15]
    if score >= 670:
        return [0.10, 0.15, 0.50, 0.20, 0.05]
    if score >= 580:
        return [0.15, 0.50, 0.25, 0.08, 0.02]
    return [0.70, 0.20, 0.08, 0.02, 0.00]


def _to_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)
