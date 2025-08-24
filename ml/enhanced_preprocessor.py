#!/usr/bin/env python3
"""Compatibility wrapper providing HybridDataPreprocessor expected by tests.

Wraps NexaCreditDataPreprocessor (data_preprocessor.py) and exposes:
  - clean_and_preprocess(train_path, test_path) -> (train_df, test_df)
  - prepare_target_variable(df) -> (X, y)

Simplifies legacy pipeline while removing redundant older implementations.
"""
from __future__ import annotations

from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np

try:  # package context
    from .data_preprocessor import NexaCreditDataPreprocessor
except ImportError:
    from data_preprocessor import NexaCreditDataPreprocessor  # type: ignore


_CREDIT_MAP = {
    'Poor': 'Poor',
    'Standard': 'Standard', 'Fair': 'Standard',
    'Good': 'Good', 'Very Good': 'Good', 'Exceptional': 'Good'
}
_LABEL_MAP = {'Poor': 0, 'Standard': 1, 'Good': 2}

class HybridDataPreprocessor:
    def __init__(self):
        self.base = NexaCreditDataPreprocessor()

    def clean_and_preprocess(self, train_path: str, test_path: str):
        train_df = self.base.preprocess_dataset(train_path, is_training=True)
        test_df = self.base.preprocess_dataset(test_path, is_training=False)
        return train_df, test_df

    def prepare_target_variable(self, df: pd.DataFrame):
        if 'Credit_Score' not in df.columns:
            raise ValueError("Credit_Score column missing")
        mapped = df['Credit_Score'].astype(str).str.strip().map(_CREDIT_MAP)
        # Drop rows with unmapped categories
        mask = mapped.notna()
        df2 = df.loc[mask].copy()
        mapped = mapped.loc[mask]
        y = mapped.map(_LABEL_MAP).astype(int).values
        # Drop target and non-numeric columns for X
        feature_df = df2.drop(columns=['Credit_Score'])
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        X = feature_df[numeric_cols].fillna(feature_df[numeric_cols].median())
        return X, y

__all__ = ["HybridDataPreprocessor"]

