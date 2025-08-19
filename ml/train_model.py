#!/usr/bin/env python3
"""
Nexacred ML Model Training Script
=================================

This script trains a machine learning model for credit scoring using scikit-learn.
It generates synthetic data and trains a LogisticRegression model to predict credit scores.

In a production environment, this would use real financial data with proper
feature engineering and data preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class CreditScoreModel:
    """
    A machine learning model for credit scoring
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.model_version = "1.0.0"
        self.trained_at = None
        
    def generate_synthetic_data(self, n_samples=10000, n_features=20):
        """
        Generate synthetic financial data for training
        
        In production, this would be replaced with real financial data including:
        - Payment history
        - Credit utilization ratio
        - Length of credit history
        - Types of credit accounts
        - Recent credit inquiries
        - Income information
        - Debt-to-income ratio
        - Employment history
        """
        print(f"Generating {n_samples} synthetic data samples with {n_features} features...")
        
        # Generate synthetic classification data
        X, y_binary = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=2,
            flip_y=0.02,  # Add some noise
            class_sep=0.8,
            random_state=self.random_state
        )
        
        # Convert to DataFrame with meaningful feature names
        self.feature_names = [
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
            'loan_default_history'
        ]
        
        # Ensure we have the right number of feature names
        if len(self.feature_names) > n_features:
            self.feature_names = self.feature_names[:n_features]
        elif len(self.feature_names) < n_features:
            # Add generic feature names if needed
            for i in range(len(self.feature_names), n_features):
                self.feature_names.append(f'feature_{i+1}')
        
        df = pd.DataFrame(X, columns=self.feature_names)
        
        # Convert binary classification to credit score categories
        # 0: Poor (300-579), 1: Fair (580-669), 2: Good (670-739), 3: Very Good (740-799), 4: Exceptional (800-850)
        credit_score_categories = np.where(
            y_binary == 0,
            np.random.choice([0, 1], size=np.sum(y_binary == 0), p=[0.7, 0.3]),  # More poor/fair scores for class 0
            np.random.choice([2, 3, 4], size=np.sum(y_binary == 1), p=[0.4, 0.4, 0.2])  # More good/excellent scores for class 1
        )
        
        # Add some realistic constraints and transformations
        df['payment_history_score'] = np.clip(df['payment_history_score'], -3, 3)
        df['credit_utilization_ratio'] = np.clip(np.abs(df['credit_utilization_ratio']) * 0.5, 0, 1)
        df['length_of_credit_history_months'] = np.clip(np.abs(df['length_of_credit_history_months']) * 50, 6, 600)
        df['annual_income'] = np.clip(np.abs(df['annual_income']) * 30000 + 25000, 15000, 300000)
        df['age'] = np.clip(np.abs(df['age']) * 10 + 25, 18, 80)
        
        print("Synthetic data generation completed!")
        print(f"Feature columns: {list(df.columns)}")
        print(f"Credit score categories distribution:")
        unique, counts = np.unique(credit_score_categories, return_counts=True)
        for cat, count in zip(unique, counts):
            category_names = ['Poor (300-579)', 'Fair (580-669)', 'Good (670-739)', 'Very Good (740-799)', 'Exceptional (800-850)']
            print(f"  {category_names[cat]}: {count} samples ({count/len(credit_score_categories)*100:.1f}%)")
        
        return df, credit_score_categories
    
    def train_model(self, X, y):
        """
        Train the credit scoring model
        """
        print("Training credit scoring model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Create a pipeline with preprocessing and model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                multi_class='ovr',  # One-vs-Rest for multiclass
                solver='liblinear'
            ))
        ])
        
        # Perform hyperparameter tuning
        print("Performing hyperparameter tuning...")
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2']
        }
        
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Use the best model
        self.pipeline = grid_search.best_estimator_
        self.model = self.pipeline.named_steps['classifier']
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Training completed at: {datetime.now()}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:")
        target_names = ['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance (for LogisticRegression, we can use coefficients)
        self.print_feature_importance(X_train)
        
        self.trained_at = datetime.now()
        
        return accuracy
    
    def print_feature_importance(self, X_train):
        """
        Print feature importance based on model coefficients
        """
        if self.model is None:
            return
        
        print(f"\nFeature Importance (based on model coefficients):")
        
        # For multiclass logistic regression, we'll use the mean of absolute coefficients
        if len(self.model.coef_) > 1:
            # Multiclass case
            mean_coef = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            # Binary case
            mean_coef = np.abs(self.model.coef_[0])
        
        # Sort features by importance
        feature_importance = list(zip(self.feature_names, mean_coef))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 10 most important features:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1:2d}. {feature:<30} {importance:.4f}")
    
    def predict_credit_score_category(self, X):
        """
        Predict credit score category for given features
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict(X)
    
    def predict_credit_score_proba(self, X):
        """
        Predict credit score category probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.pipeline.predict_proba(X)
    
    def convert_category_to_score(self, category):
        """
        Convert category prediction to actual credit score
        """
        score_ranges = {
            0: (300, 579),  # Poor
            1: (580, 669),  # Fair  
            2: (670, 739),  # Good
            3: (740, 799),  # Very Good
            4: (800, 850)   # Exceptional
        }
        
        if isinstance(category, (list, np.ndarray)):
            scores = []
            for cat in category:
                min_score, max_score = score_ranges[cat]
                # Generate a random score within the range
                score = np.random.randint(min_score, max_score + 1)
                scores.append(score)
            return np.array(scores)
        else:
            min_score, max_score = score_ranges[category]
            return np.random.randint(min_score, max_score + 1)
    
    def save_model(self, filepath):
        """
        Save the trained model to disk
        """
        if self.pipeline is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'model_version': self.model_version,
            'trained_at': self.trained_at,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.feature_names = model_data['feature_names']
        self.model_version = model_data.get('model_version', 'unknown')
        self.trained_at = model_data.get('trained_at', 'unknown')
        self.random_state = model_data.get('random_state', 42)
        self.model = self.pipeline.named_steps['classifier']
        
        print(f"Model loaded from: {filepath}")
        print(f"Model version: {self.model_version}")
        print(f"Trained at: {self.trained_at}")

def main():
    """
    Main training function
    """
    print("=" * 60)
    print("NEXACRED CREDIT SCORING MODEL TRAINING")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Initialize the model
    credit_model = CreditScoreModel(random_state=42)
    
    try:
        # Generate synthetic data
        X, y = credit_model.generate_synthetic_data(n_samples=10000, n_features=20)
        
        print(f"\nDataset shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Train the model
        accuracy = credit_model.train_model(X, y)
        
        # Save the model
        model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
        credit_model.save_model(model_path)
        
        # Test the saved model by loading it
        print(f"\nTesting model loading...")
        test_model = CreditScoreModel()
        test_model.load_model(model_path)
        
        # Make a sample prediction
        print(f"\nMaking sample predictions...")
        sample_data = X.iloc[:5]  # Take first 5 samples
        predictions = test_model.predict_credit_score_category(sample_data)
        probabilities = test_model.predict_credit_score_proba(sample_data)
        credit_scores = test_model.convert_category_to_score(predictions)
        
        print(f"Sample predictions:")
        category_names = ['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
        for i in range(len(predictions)):
            print(f"  Sample {i+1}: Category = {category_names[predictions[i]]}, Score = {credit_scores[i]}")
            print(f"    Probabilities: {probabilities[i]}")
        
        print(f"\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Model saved to: {model_path}")
        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Completed at: {datetime.now()}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
