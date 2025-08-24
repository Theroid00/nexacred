#!/usr/bin/env python3
"""
Hybrid Credit Scoring System
============================

Combines traditional ML models for structured credit scoring with IBM Granite for:
- RAG-based financial assistant
- Risk report generation
- Loan recommendation systems
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Centralized credit utilities (fallback if package context absent)
try:
    from .credit_utils import generate_recommendations_three_class
except ImportError:  # pragma: no cover - direct script execution
    from credit_utils import generate_recommendations_three_class

class HybridCreditScoringSystem:
    """
    Hybrid system combining traditional ML for credit scoring 
    and IBM Granite for financial assistance
    """
    
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
        self.feature_importance = {}
        self.preprocessing_pipeline = None
        self.is_trained = False
        
    def train_traditional_models(self, X_train: pd.DataFrame, y_train: np.ndarray, 
                                X_val: pd.DataFrame, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Train multiple traditional ML models for credit scoring
        
        Returns:
            Dictionary with model performance metrics
        """
        print("🚀 Training traditional ML models for credit scoring...")
        
        # Initialize models
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            ),  # removed unsupported max_depth param
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        }
        
        results = {}
        
        # Train each model
        for model_name, model in models_to_train.items():
            print(f"   Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Store model and metrics
            self.models[model_name] = model
            self.model_metrics[model_name] = {
                'train_accuracy': train_acc,
                'validation_accuracy': val_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_val, val_pred, output_dict=True)
            }
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[model_name] = importance_df
            
            results[model_name] = {
                'validation_accuracy': val_acc,
                'cv_score': cv_scores.mean()
            }
            
            print(f"     ✓ {model_name}: Val Acc = {val_acc:.4f}, CV = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.is_trained = True
        
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['validation_accuracy'])
        print(f"\n🏆 Best performing model: {best_model} (Accuracy: {results[best_model]['validation_accuracy']:.4f})")
        
        return results
    
    def predict_credit_score(self, X: pd.DataFrame, model_name: str = 'random_forest') -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict credit scores using trained traditional ML model
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained or model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained or not found")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return predictions, probabilities
    
    def generate_credit_score_explanation(self, X: pd.DataFrame, prediction: int, 
                                        probabilities: np.ndarray, model_name: str = 'random_forest') -> Dict[str, Any]:
        """
        Generate explanation for credit score prediction
        """
        model = self.models[model_name]
        
        # Credit score mapping
        score_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        credit_category = score_mapping[prediction]
        confidence = probabilities[prediction]
        
        explanation = {
            'credit_category': credit_category,
            'confidence': float(confidence),
            'score_range': self._get_score_range(prediction),
            'key_factors': self._get_key_factors(X, model_name),
            'recommendations': generate_recommendations_three_class(prediction)
        }
        
        return explanation
    
    def _get_score_range(self, prediction: int) -> str:
        """Get credit score range for prediction"""
        ranges = {
            0: '300-579 (Poor)',
            1: '580-669 (Standard)', 
            2: '670-850 (Good)'
        }
        return ranges.get(prediction, 'Unknown')
    
    def _get_key_factors(self, X: pd.DataFrame, model_name: str) -> List[Dict[str, Any]]:
        """Get key factors influencing the prediction"""
        if model_name not in self.feature_importance:
            return []
        
        importance_df = self.feature_importance[model_name].head(5)
        factors = []
        
        for _, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Get feature value if available
            feature_value = None
            if hasattr(X, 'columns') and feature in X.columns:
                feature_value = X[feature].iloc[0] if len(X) > 0 else None
            
            factors.append({
                'factor': feature,
                'importance': float(importance),
                'value': feature_value,
                'impact': 'High' if importance > 0.1 else 'Medium' if importance > 0.05 else 'Low'
            })
        
        return factors
    
    def save_models(self, model_dir: str = 'models'):
        """Save trained models and metadata"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f'{model_dir}/{model_name}_model.pkl')
        
        # Save metrics and metadata
        metadata = {
            'model_metrics': self.model_metrics,
            'feature_importance': {k: v.to_dict() for k, v in self.feature_importance.items()},
            'trained_at': datetime.now().isoformat(),
            'is_trained': self.is_trained
        }
        
        with open(f'{model_dir}/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Models saved to {model_dir}/")
    
    def load_models(self, model_dir: str = 'models'):
        """Load trained models and metadata"""
        import os
        
        # Load models
        for filename in os.listdir(model_dir):
            if filename.endswith('_model.pkl'):
                model_name = filename.replace('_model.pkl', '')
                self.models[model_name] = joblib.load(f'{model_dir}/{filename}')
        
        # Load metadata
        metadata_path = f'{model_dir}/model_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_metrics = metadata.get('model_metrics', {})
                self.is_trained = metadata.get('is_trained', False)
                
                # Reconstruct feature importance DataFrames
                feature_importance_data = metadata.get('feature_importance', {})
                for model_name, importance_dict in feature_importance_data.items():
                    self.feature_importance[model_name] = pd.DataFrame(importance_dict)
        
        print(f"✅ Models loaded from {model_dir}/")
    
    def evaluate_model_performance(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_prob.tolist()
            }
            
            print(f"\n📊 {model_name.upper()} Performance:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Classification Report:")
            print(classification_report(y_test, y_pred))
        
        return evaluation_results
