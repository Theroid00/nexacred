#!/usr/bin/env python3
"""
NexaCred Credit Scoring Model
=============================

Lightweight heuristic credit scoring model exposing a stable interface for
system tests. (Original file had duplicated imports and corrupted logic.)

Provides 5 credit categories (0..4):
  0 Poor (300-579)
  1 Fair (580-669)
  2 Good (670-739)
  3 Very Good (740-799)
  4 Exceptional (800-900)

Public methods expected by tests:
  - generate_synthetic_data(n_samples, n_features)
  - train_model(X, y)
  - predict_credit_score_category(X)
  - predict_credit_score_proba(X)
  - convert_category_to_score(categories)
  - save_model(filepath)
  - load_model(filepath)
  - get_model_info()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from typing import Iterable, Sequence, List, Dict, Any

# Centralized mapping utilities
try:  # package context
    from .credit_utils import (
        score_to_category_five,
        score_to_probabilities_five,
        calculate_base_credit_score,
    )
except ImportError:  # script / loose execution
    from credit_utils import (  # type: ignore
        score_to_category_five,
        score_to_probabilities_five,
        calculate_base_credit_score,
    )


class CreditScoreModel:
    """Heuristic credit scoring model (acts like a trained model for demos)."""

    def __init__(self, random_state: int = 42):
        self.model_version = "2.1.0"
        self.trained_at: datetime | None = None
        self.random_state = random_state
        self.last_accuracy: float | None = None
        np.random.seed(random_state)

        # Canonical feature list (synthetic generator will populate subset)
        self.feature_names: List[str] = [
            'payment_history_score',
            'credit_utilization_ratio',
            'length_of_credit_history_months',
            'number_of_credit_accounts',
            'recent_credit_inquiries',
            'annual_income',
            'debt_to_income_ratio',
            'employment_tenure_months',
            'number_of_late_payments',
            'total_credit_limit',
            'current_debt_amount',
            'number_of_bankruptcies',
            'age',
            'education_level',
            'homeownership_status',
            'monthly_expenses',
            'savings_account_balance',
            'checking_account_balance',
            'investment_portfolio_value',
            'loan_default_history',
        ]

        self.score_ranges: Dict[int, tuple[int, int]] = {
            0: (300, 579),  # Poor
            1: (580, 669),  # Fair
            2: (670, 739),  # Good
            3: (740, 799),  # Very Good
            4: (800, 900),  # Exceptional
        }

    # ------------------------------------------------------------------
    # Data generation & internal scoring
    # ------------------------------------------------------------------
    def generate_synthetic_data(self, n_samples: int = 1000, n_features: int = 20):
        """Generate synthetic dataset and corresponding category labels.

        Returns: (X_df, y_categories)
        """
        n_features = max(5, min(n_features, len(self.feature_names)))
        selected = self.feature_names[:n_features]

        rng = np.random.default_rng(self.random_state)
        data: Dict[str, Any] = {}

        if 'payment_history_score' in selected:
            data['payment_history_score'] = rng.normal(1.0, 1.2, n_samples)  # ~0-3 typical
        if 'credit_utilization_ratio' in selected:
            data['credit_utilization_ratio'] = rng.beta(2, 5, n_samples)  # 0-1
        if 'length_of_credit_history_months' in selected:
            data['length_of_credit_history_months'] = rng.gamma(2, 24, n_samples)
        if 'number_of_credit_accounts' in selected:
            data['number_of_credit_accounts'] = rng.integers(1, 15, n_samples)
        if 'recent_credit_inquiries' in selected:
            data['recent_credit_inquiries'] = rng.integers(0, 6, n_samples)
        if 'annual_income' in selected:
            data['annual_income'] = rng.lognormal(mean=13, sigma=0.45, size=n_samples)  # rupees-ish
        if 'debt_to_income_ratio' in selected:
            data['debt_to_income_ratio'] = rng.beta(2, 4, n_samples)
        if 'employment_tenure_months' in selected:
            data['employment_tenure_months'] = rng.integers(0, 240, n_samples)
        if 'number_of_late_payments' in selected:
            data['number_of_late_payments'] = rng.poisson(1.0, n_samples)
        if 'total_credit_limit' in selected:
            data['total_credit_limit'] = rng.lognormal(12, 0.6, n_samples)
        if 'current_debt_amount' in selected:
            data['current_debt_amount'] = rng.lognormal(11.5, 0.7, n_samples)
        if 'number_of_bankruptcies' in selected:
            data['number_of_bankruptcies'] = rng.integers(0, 2, n_samples)
        if 'age' in selected:
            data['age'] = rng.normal(36, 9, n_samples)
        if 'monthly_expenses' in selected:
            data['monthly_expenses'] = rng.lognormal(10, 0.4, n_samples)
        if 'savings_account_balance' in selected:
            data['savings_account_balance'] = rng.lognormal(11, 0.8, n_samples)
        if 'checking_account_balance' in selected:
            data['checking_account_balance'] = rng.lognormal(10.5, 0.6, n_samples)
        if 'investment_portfolio_value' in selected:
            data['investment_portfolio_value'] = rng.lognormal(12, 1.0, n_samples)
        if 'loan_default_history' in selected:
            data['loan_default_history'] = rng.integers(0, 2, n_samples)

        # Fill any uninitialized selected feature with noise
        for f in selected:
            if f not in data:
                data[f] = rng.normal(0, 1, n_samples)

        X = pd.DataFrame({k: data[k] for k in selected})

        # Derive categories using internal scoring
        categories = np.array([self._score_to_category(self._calculate_credit_score(X.iloc[i].to_dict())) for i in range(n_samples)])
        return X, categories

    def _calculate_credit_score(self, features: Dict[str, Any]) -> int:
        """Heuristic score using centralized base + lightweight adjustments."""
        base = calculate_base_credit_score({
            'annual_income': features.get('annual_income'),
            'debt_to_income_ratio': features.get('debt_to_income_ratio'),
            'credit_utilization_ratio': features.get('credit_utilization_ratio'),
            'number_of_late_payments': features.get('number_of_late_payments'),
            'age': features.get('age'),
        })
        # Additional adjustments
        payment_hist = features.get('payment_history_score', 0) or 0
        base += int(payment_hist * 60)  # amplify on-time pattern
        inquiries = features.get('recent_credit_inquiries', 0) or 0
        base -= int(max(0, inquiries - 1) * 15)
        tenure = features.get('employment_tenure_months', 0) or 0
        if tenure > 60:
            base += 25
        return int(max(300, min(900, base)))

    def _score_to_category(self, score: int) -> int:
        return score_to_category_five(score)

    def _score_to_probabilities(self, score: int) -> List[float]:
        return score_to_probabilities_five(score)

    # ------------------------------------------------------------------
    # Training & inference API
    # ------------------------------------------------------------------
    def train_model(self, X: pd.DataFrame, y: Sequence[int] | None = None) -> float:
        """"Train" model (heuristic - no parameters). Returns pseudo-accuracy.

        If y provided, compute agreement between heuristic categories and y.
        If y is None, generates categories internally and returns 1.0.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        preds = np.array([self._score_to_category(self._calculate_credit_score(X.iloc[i].to_dict())) for i in range(len(X))])
        if y is not None:
            y_arr = np.array(y)
            if len(y_arr) != len(preds):
                raise ValueError("Length mismatch between X and y")
            # Align label space if y contains out-of-range values (defensive)
            valid_mask = np.isin(y_arr, list(self.score_ranges.keys()))
            if not np.all(valid_mask):
                y_arr = y_arr[valid_mask]
                preds = preds[valid_mask]
            accuracy = float(np.mean(preds == y_arr)) if len(y_arr) else 0.0
        else:
            accuracy = 1.0
        self.trained_at = datetime.now()
        self.last_accuracy = accuracy
        return accuracy

    def predict_credit_score_category(self, X: pd.DataFrame | Dict[str, Any]) -> np.ndarray:
        if self.trained_at is None:
            raise ValueError("Model must be trained before predictions")
        if isinstance(X, dict):  # single record
            return np.array([self._score_to_category(self._calculate_credit_score(X))])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be DataFrame or dict")
        return np.array([self._score_to_category(self._calculate_credit_score(X.iloc[i].to_dict())) for i in range(len(X))])

    def predict_credit_score_proba(self, X: pd.DataFrame | Dict[str, Any]) -> np.ndarray:
        if self.trained_at is None:
            raise ValueError("Model must be trained before predictions")
        if isinstance(X, dict):
            score = self._calculate_credit_score(X)
            return np.array([self._score_to_probabilities(score)])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be DataFrame or dict")
        return np.array([self._score_to_probabilities(self._calculate_credit_score(X.iloc[i].to_dict())) for i in range(len(X))])

    def convert_category_to_score(self, categories: Iterable[int]) -> List[int]:
        """Map category labels back to representative numeric scores (midpoints)."""
        out: List[int] = []
        for c in categories:
            low, high = self.score_ranges.get(int(c), (300, 579))
            out.append(int((low + high) / 2))
        return out

    # ------------------------------------------------------------------
    # Persistence & metadata
    # ------------------------------------------------------------------
    def save_model(self, filepath: str):
        model_data = {
            'model_version': self.model_version,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'feature_names': self.feature_names,
            'score_ranges': self.score_ranges,
            'random_state': self.random_state,
            'last_accuracy': self.last_accuracy,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        self.model_version = model_data.get('model_version', self.model_version)
        ts = model_data.get('trained_at')
        self.trained_at = datetime.fromisoformat(ts) if ts else None
        self.feature_names = model_data.get('feature_names', self.feature_names)
        self.score_ranges = {int(k): tuple(v) for k, v in model_data.get('score_ranges', self.score_ranges).items()}
        self.random_state = model_data.get('random_state', self.random_state)
        self.last_accuracy = model_data.get('last_accuracy')

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_version': self.model_version,
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'feature_count': len(self.feature_names),
            'last_accuracy': self.last_accuracy,
        }


# ----------------------------------------------------------------------
# Demonstration entrypoint
# ----------------------------------------------------------------------

def main():  # pragma: no cover - demo utility
    print("=" * 60)
    print("🤖 NEXACRED CREDIT SCORING MODEL (Heuristic Demo)")
    print("=" * 60)
    model = CreditScoreModel()
    X, y = model.generate_synthetic_data(n_samples=300, n_features=15)
    acc = model.train_model(X, y)
    print(f"Trained heuristic model. Pseudo-accuracy: {acc:.4f}")
    sample = X.head(3)
    cats = model.predict_credit_score_category(sample)
    probs = model.predict_credit_score_proba(sample)
    scores = model.convert_category_to_score(cats)
    for i, (c, s, p) in enumerate(zip(cats, scores, probs), 1):
            print(f"Sample {i}: category={c} score≈{s} probs={np.round(p,3)}")
    out_path = os.path.join(os.path.dirname(__file__), 'credit_model.json')
    model.save_model(out_path)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
