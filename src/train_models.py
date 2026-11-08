"""
Traditional Machine Learning Training and Hyperparameter Tuning
Models: Logistic Regression, KNN, Random Forest
Verveox Project 5 Specification
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Import our preprocessing pipeline
from data_preprocessing import preprocess_and_split

MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def train_and_tune_models():
    # 1. Load preprocessed and scaled data
    X_train, X_test, y_train, y_test, selected_features = preprocess_and_split()
    
    # 2. Set up 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 3. Define models and parameter grids for GridSearchCV
    model_configs = {
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5]
            }
        }
    }
    
    results = []
    
    print("\n" + "=" * 60)
    print("STARTING GRIDSEARCHCV & MODEL TRAINING")
    print("=" * 60)
    
    for name, config in model_configs.items():
        print(f"\n---> Tuning {name}...")
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        print(f"  Best Parameters: {grid.best_params_}")
        print(f"  Best 5-Fold CV ROC-AUC: {grid.best_score_:.4f}")
        
        # Evaluate on the unseen test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(roc_auc, 4),
            "Best_Params": str(grid.best_params_)
        })
        
        # Save each trained model to disk
        model_path = os.path.join(MODELS_DIR, f"{name.lower()}.joblib")
        joblib.dump(best_model, model_path)
        print(f"  Saved model to: {model_path}")
        
    # Convert results into DataFrame for clean summary
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION SUMMARY (TRADITIONAL ML)")
    print("=" * 60)
    print(results_df[["Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]].to_string(index=False))
    
    # Save results summary to CSV for comparison phase
    summary_path = os.path.join(REPORTS_DIR, "traditional_models_metrics.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\nMetrics saved to: {summary_path}")
    
    return results_df

if __name__ == "__main__":
    train_and_tune_models()