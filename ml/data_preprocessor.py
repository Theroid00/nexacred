#!/usr/bin/env python3
"""
NexaCred Data Preprocessor
==========================

Comprehensive data preprocessing pipeline for credit scoring datasets.
Handles missing values, outliers, data type conversions, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

class NexaCreditDataPreprocessor:
    """
    Advanced data preprocessing pipeline for NexaCred credit scoring system
    """

    def __init__(self):
        self.feature_mappings = {}
        self.imputation_values = {}
        self.outlier_bounds = {}
        self.preprocessing_stats = {}

    def preprocess_dataset(self, file_path: str, is_training: bool = True) -> pd.DataFrame:
        """
        Main preprocessing pipeline for credit datasets

        Args:
            file_path: Path to CSV file
            is_training: Whether this is training data (has target variable)

        Returns:
            Preprocessed DataFrame
        """
        print(f"🔄 Starting preprocessing for: {file_path}")

        # Load data
        df = pd.read_csv(file_path, low_memory=False)
        print(f"📊 Loaded dataset: {df.shape[0]} records, {df.shape[1]} features")

        # Store original stats
        self.preprocessing_stats['original_shape'] = df.shape
        self.preprocessing_stats['original_missing'] = df.isnull().sum().sum()

        # Step 1: Clean basic data issues
        df = self._clean_basic_issues(df)

        # Step 2: Handle missing values
        df = self._handle_missing_values(df, is_training)

        # Step 3: Fix data types and formats
        df = self._fix_data_types(df)

        # Step 4: Handle outliers
        df = self._handle_outliers(df)

        # Step 5: Feature engineering
        df = self._engineer_features(df)

        # Step 6: Encode categorical variables
        df = self._encode_categorical(df, is_training)

        # Step 7: Final validation
        df = self._final_validation(df, is_training)

        # Store final stats
        self.preprocessing_stats['final_shape'] = df.shape
        self.preprocessing_stats['final_missing'] = df.isnull().sum().sum()

        print(f"✅ Preprocessing completed: {df.shape[0]} records, {df.shape[1]} features")

        return df

    def _clean_basic_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean obvious data quality issues"""
        print("🧹 Cleaning basic data issues...")

        df_clean = df.copy()

        # Fix age issues (negative ages, extreme values)
        if 'Age' in df_clean.columns:
            # Convert Age to numeric, handling string values
            df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')

            # Fix impossible age values
            invalid_ages = (df_clean['Age'] < 18) | (df_clean['Age'] > 100)
            df_clean.loc[invalid_ages, 'Age'] = np.nan

            print(f"   Fixed {invalid_ages.sum()} invalid age values")

        # Clean SSN format (ensure consistency)
        if 'SSN' in df_clean.columns:
            df_clean['SSN'] = df_clean['SSN'].astype(str).str.replace(r'[^\d-]', '', regex=True)

        # Clean income values (remove non-numeric characters)
        income_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt',
                      'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

        for col in income_cols:
            if col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    # Remove currency symbols and clean numeric values
                    df_clean[col] = df_clean[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Clean percentage values
        if 'Credit_Utilization_Ratio' in df_clean.columns:
            # Convert percentage strings to decimals if needed
            if df_clean['Credit_Utilization_Ratio'].dtype == 'object':
                df_clean['Credit_Utilization_Ratio'] = pd.to_numeric(
                    df_clean['Credit_Utilization_Ratio'].astype(str).str.replace('%', ''),
                    errors='coerce'
                ) / 100

        return df_clean

    def _handle_missing_values(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Handle missing values with intelligent imputation"""
        print("🔧 Handling missing values...")

        df_filled = df.copy()

        # Strategy 1: Forward fill for time series data (same customer)
        if 'Customer_ID' in df_filled.columns:
            df_filled = df_filled.sort_values(['Customer_ID', 'Month'])

            # Forward fill within same customer
            for col in df_filled.select_dtypes(include=[np.number]).columns:
                if col not in ['ID']:
                    df_filled[col] = df_filled.groupby('Customer_ID')[col].fillna(method='ffill')

        # Strategy 2: Statistical imputation
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in ['ID', 'Customer_ID'] and df_filled[col].isnull().sum() > 0:

                if col in ['Monthly_Inhand_Salary', 'Annual_Income']:
                    # Income: Use median within occupation groups
                    if 'Occupation' in df_filled.columns:
                        df_filled[col] = df_filled.groupby('Occupation')[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                    # Fill remaining with overall median
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)

                elif col in ['Age']:
                    # Age: Use mode (most common age)
                    df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

                else:
                    # Other numeric: Use median
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)

        # Strategy 3: Categorical imputation
        categorical_cols = df_filled.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if col not in ['ID', 'Customer_ID', 'Name', 'SSN'] and df_filled[col].isnull().sum() > 0:

                if col == 'Occupation':
                    df_filled[col].fillna('Other', inplace=True)

                elif col == 'Type_of_Loan':
                    df_filled[col].fillna('Personal Loan', inplace=True)

                elif col in ['Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']:
                    # Use mode (most frequent value)
                    mode_val = df_filled[col].mode()[0] if len(df_filled[col].mode()) > 0 else 'Unknown'
                    df_filled[col].fillna(mode_val, inplace=True)

                else:
                    df_filled[col].fillna('Unknown', inplace=True)

        # Store imputation values for future use
        if is_training:
            self.imputation_values = {
                col: df_filled[col].median() if df_filled[col].dtype in ['int64', 'float64']
                else df_filled[col].mode()[0] if len(df_filled[col].mode()) > 0 else 'Unknown'
                for col in df_filled.columns
                if col not in ['ID', 'Customer_ID', 'Name', 'SSN']
            }

        missing_after = df_filled.isnull().sum().sum()
        print(f"   Reduced missing values to: {missing_after}")

        return df_filled

    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix and standardize data types"""
        print("🔧 Fixing data types...")

        df_typed = df.copy()

        # Numeric columns that should be integers
        int_cols = ['Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
                   'Delay_from_due_date', 'Age']

        for col in int_cols:
            if col in df_typed.columns:
                df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce').fillna(0).astype(int)

        # Numeric columns that should be floats
        float_cols = ['Monthly_Inhand_Salary', 'Annual_Income', 'Outstanding_Debt',
                     'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly',
                     'Monthly_Balance', 'Num_Credit_Inquiries']

        for col in float_cols:
            if col in df_typed.columns:
                df_typed[col] = pd.to_numeric(df_typed[col], errors='coerce').fillna(0).astype(float)

        # Clean string columns
        string_cols = ['Name', 'Occupation', 'Type_of_Loan', 'Credit_Mix',
                      'Payment_of_Min_Amount', 'Payment_Behaviour']

        for col in string_cols:
            if col in df_typed.columns:
                df_typed[col] = df_typed[col].astype(str).str.strip().str.title()

        # Handle special columns
        if 'Credit_History_Age' in df_typed.columns:
            # Extract numeric years from text like "5 Years and 2 Months"
            df_typed['Credit_History_Years'] = df_typed['Credit_History_Age'].str.extract(r'(\d+)').astype(float)

        if 'Num_of_Loan' in df_typed.columns:
            # Convert loan count to numeric
            df_typed['Num_of_Loan'] = pd.to_numeric(df_typed['Num_of_Loan'], errors='coerce').fillna(0)

        if 'Num_of_Delayed_Payment' in df_typed.columns:
            # Convert delayed payment count to numeric
            df_typed['Num_of_Delayed_Payment'] = pd.to_numeric(df_typed['Num_of_Delayed_Payment'], errors='coerce').fillna(0)

        if 'Changed_Credit_Limit' in df_typed.columns:
            # Convert credit limit changes to numeric
            df_typed['Changed_Credit_Limit'] = pd.to_numeric(df_typed['Changed_Credit_Limit'], errors='coerce').fillna(0)

        return df_typed

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        print("📊 Handling outliers...")

        df_clean = df.copy()

        # Define columns to check for outliers
        outlier_cols = ['Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt',
                       'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
                       'Num_Bank_Accounts', 'Num_Credit_Card']

        outliers_fixed = 0

        for col in outlier_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Count outliers
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))
                outliers_count = outliers.sum()

                if outliers_count > 0:
                    # Cap outliers instead of removing them
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound

                    outliers_fixed += outliers_count

                # Store bounds for future use
                self.outlier_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}

        print(f"   Fixed {outliers_fixed} outlier values")

        return df_clean

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new engineered features"""
        print("⚙️ Engineering features...")

        df_eng = df.copy()

        # 1. Debt-to-Income Ratio
        if 'Outstanding_Debt' in df_eng.columns and 'Monthly_Inhand_Salary' in df_eng.columns:
            df_eng['Debt_to_Income_Ratio'] = df_eng['Outstanding_Debt'] / (df_eng['Monthly_Inhand_Salary'] * 12 + 1)

        # 2. Credit Account Diversity
        if 'Num_Bank_Accounts' in df_eng.columns and 'Num_Credit_Card' in df_eng.columns:
            df_eng['Total_Accounts'] = df_eng['Num_Bank_Accounts'] + df_eng['Num_Credit_Card']

        # 3. Payment Stress Indicator
        if 'Total_EMI_per_month' in df_eng.columns and 'Monthly_Inhand_Salary' in df_eng.columns:
            df_eng['EMI_to_Income_Ratio'] = df_eng['Total_EMI_per_month'] / (df_eng['Monthly_Inhand_Salary'] + 1)

        # 4. Investment Capacity
        if 'Amount_invested_monthly' in df_eng.columns and 'Monthly_Inhand_Salary' in df_eng.columns:
            df_eng['Investment_to_Income_Ratio'] = df_eng['Amount_invested_monthly'] / (df_eng['Monthly_Inhand_Salary'] + 1)

        # 5. Age Group Categories
        if 'Age' in df_eng.columns:
            df_eng['Age_Group'] = pd.cut(df_eng['Age'],
                                       bins=[0, 25, 35, 50, 65, 100],
                                       labels=['Young', 'Young_Adult', 'Middle_Age', 'Senior', 'Elderly'])

        # 6. Income Level Categories
        if 'Annual_Income' in df_eng.columns:
            df_eng['Income_Level'] = pd.qcut(df_eng['Annual_Income'],
                                           q=5,
                                           labels=['Low', 'Low_Medium', 'Medium', 'Medium_High', 'High'])

        # 7. Credit Utilization Risk Level
        if 'Credit_Utilization_Ratio' in df_eng.columns:
            df_eng['Utilization_Risk'] = pd.cut(df_eng['Credit_Utilization_Ratio'],
                                              bins=[0, 0.1, 0.3, 0.7, 1.0],
                                              labels=['Low', 'Medium', 'High', 'Very_High'])

        return df_eng

    def _encode_categorical(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Encode categorical variables"""
        print("🏷️ Encoding categorical variables...")

        df_encoded = df.copy()

        # Target encoding for Credit_Score (if present)
        if 'Credit_Score' in df_encoded.columns:
            credit_score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
            df_encoded['Credit_Score_Numeric'] = df_encoded['Credit_Score'].map(credit_score_mapping)

            if is_training:
                self.feature_mappings['Credit_Score'] = credit_score_mapping

        # Binary encodings
        binary_mappings = {
            'Payment_of_Min_Amount': {'No': 0, 'Yes': 1},
        }

        for col, mapping in binary_mappings.items():
            if col in df_encoded.columns:
                df_encoded[f'{col}_Encoded'] = df_encoded[col].map(mapping).fillna(0)
                if is_training:
                    self.feature_mappings[col] = mapping

        # One-hot encode high-cardinality categoricals
        categorical_cols = ['Credit_Mix', 'Payment_Behaviour', 'Type_of_Loan']

        for col in categorical_cols:
            if col in df_encoded.columns:
                # Get top categories to avoid too many dummy variables
                top_categories = df_encoded[col].value_counts().head(10).index.tolist()
                df_encoded[col] = df_encoded[col].apply(lambda x: x if x in top_categories else 'Other')

                # Create dummy variables
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)

        return df_encoded

    def _final_validation(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Final data validation and cleanup"""
        print("✅ Final validation...")

        # Remove rows with too many missing values (if any remain)
        missing_threshold = 0.5  # Remove rows missing more than 50% of features
        missing_ratio = df.isnull().sum(axis=1) / len(df.columns)
        df_clean = df[missing_ratio <= missing_threshold].copy()

        # Remove duplicate rows
        if 'Customer_ID' in df_clean.columns and 'Month' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['Customer_ID', 'Month'], keep='first')
        else:
            df_clean = df_clean.drop_duplicates(keep='first')

        # Ensure no infinite values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

        rows_removed = len(df) - len(df_clean)
        if rows_removed > 0:
            print(f"   Removed {rows_removed} problematic rows")

        return df_clean

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations"""
        return {
            'original_shape': self.preprocessing_stats.get('original_shape'),
            'final_shape': self.preprocessing_stats.get('final_shape'),
            'missing_values_reduced': self.preprocessing_stats.get('original_missing', 0) - self.preprocessing_stats.get('final_missing', 0),
            'feature_mappings': self.feature_mappings,
            'outlier_bounds': self.outlier_bounds,
            'preprocessing_timestamp': datetime.now().isoformat()
        }

def main():
    """Main preprocessing function"""
    print("=" * 60)
    print("🔄 NEXACRED DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    try:
        # Initialize preprocessor
        preprocessor = NexaCreditDataPreprocessor()

        # Process training data
        print("\n📊 Processing training dataset...")
        train_df = preprocessor.preprocess_dataset('train.csv', is_training=True)

        # Save processed training data
        train_df.to_csv('train_processed.csv', index=False)
        print(f"✅ Saved processed training data: train_processed.csv")

        # Process test data
        print("\n📊 Processing test dataset...")
        test_df = preprocessor.preprocess_dataset('test.csv', is_training=False)

        # Save processed test data
        test_df.to_csv('test_processed.csv', index=False)
        print(f"✅ Saved processed test data: test_processed.csv")

        # Print summary
        summary = preprocessor.get_preprocessing_summary()
        print(f"\n📈 PREPROCESSING SUMMARY:")
        print(f"Original training shape: {summary['original_shape']}")
        print(f"Final training shape: {summary['final_shape']}")
        print(f"Missing values reduced: {summary['missing_values_reduced']}")

        print(f"\n✅ Data preprocessing completed successfully!")

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
