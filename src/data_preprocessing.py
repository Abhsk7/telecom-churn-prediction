"""
Data Preprocessing and Feature Selection Pipeline
Telecom Customer Churn Prediction - Verveox Project 5
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Define paths
DATA_PATH = "data/telco_customer_churn.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_clean_raw_data(data_path=DATA_PATH):
    """Loads CSV, handles missing values, and prepares target variable."""
    df = pd.read_csv(data_path)
    
    # 1. Drop customerID (unique identifier, no predictive value)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    
    # 2. Fix TotalCharges blank spaces and cast to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # 3. Map target column Churn: Yes -> 1, No -> 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def preprocess_and_split(k_features=18):
    """
    Performs one-hot encoding, train-test split, feature scaling,
    and feature selection without data leakage.
    """
    print("=" * 60)
    print("STARTING DATA PREPROCESSING & FEATURE SELECTION")
    print("=" * 60)
    
    df = load_and_clean_raw_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Separate Features (X) and Target (y)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Identify numerical and categorical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [col for col in X.columns if col not in numerical_cols]
    
    # One-Hot Encode categorical features (drop_first=True to avoid dummy variable trap)
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    all_feature_names = list(X_encoded.columns)
    print(f"Total features after One-Hot Encoding: {len(all_feature_names)}")
    
    # Train-Test Split (80% Train, 20% Test)
    # stratify=y guarantees the same 73/27 churn ratio in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")
    
    # Standardize numerical features
    scaler = StandardScaler()
    
    # Scale numerical columns strictly fitting on X_train only (prevents data leakage)
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Feature Selection using SelectKBest (ANOVA F-value) as required by Verveox specification
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Get selected feature names and their importance scores
    selected_mask = selector.get_support()
    selected_features = [feature for feature, selected in zip(all_feature_names, selected_mask) if selected]
    
    print("\n" + "-" * 50)
    print(f"Top {k_features} Features Selected by SelectKBest:")
    print("-" * 50)
    feature_scores = pd.DataFrame({
        'Feature': all_feature_names,
        'F_Score': selector.scores_
    }).sort_values(by='F_Score', ascending=False)
    
    for idx, row in feature_scores.head(k_features).iterrows():
        print(f"  - {row['Feature']:<35} (Score: {row['F_Score']:.2f})")
    
    # Save the preprocessor bundle for the Streamlit app
    preprocessor_bundle = {
        'scaler': scaler,
        'selector': selector,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'all_feature_names': all_feature_names,
        'selected_features': selected_features,
        'raw_feature_names': list(X.columns)
    }
    
    bundle_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor_bundle, bundle_path)
    print(f"\nPreprocessor bundle saved to: {bundle_path}")
    
    return X_train_selected, X_test_selected, y_train, y_test, selected_features

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, selected_features = preprocess_and_split()
    print("\nData preprocessing & feature selection completed successfully!")