"""
Comprehensive Model Comparison and Evaluation
Models: Logistic Regression, KNN, Random Forest, ANN
Verveox Project 5 Specification
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from data_preprocessing import preprocess_and_split

MODELS_DIR = "models"
FIGURES_DIR = "reports/figures"
REPORTS_DIR = "reports"
os.makedirs(FIGURES_DIR, exist_ok=True)

def evaluate_all_models():
    print("=" * 60)
    print("STARTING COMPREHENSIVE MODEL COMPARISON")
    print("=" * 60)
    
    # 1. Load test data
    X_train, X_test, y_train, y_test, selected_features = preprocess_and_split()
    
    # 2. Load trained traditional models
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn.joblib"))
    rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    
    # 3. Load trained ANN model
    ann_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "ann_model.keras"))
    
    models = {
        "Logistic Regression": lr_model,
        "KNN": knn_model,
        "Random Forest": rf_model,
        "ANN": ann_model
    }
    
    predictions = {}
    probabilities = {}
    metrics_list = []
    
    # 4. Compute predictions & probabilities for all models
    for name, model in models.items():
        if name == "ANN":
            prob = model.predict(X_test, verbose=0).ravel()
            pred = (prob >= 0.5).astype(int)
        else:
            prob = model.predict_proba(X_test)[:, 1]
            pred = model.predict(X_test)
            
        predictions[name] = pred
        probabilities[name] = prob
        
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred)
        rec = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        
        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(auc, 4)
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    print("\n" + "=" * 60)
    print("FINAL 4-MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    
    # Save master comparison table
    comparison_csv_path = os.path.join(REPORTS_DIR, "master_model_comparison.csv")
    metrics_df.to_csv(comparison_csv_path, index=False)
    print(f"\nSaved master table to: {comparison_csv_path}")
    
    # 5. Plot Side-by-Side Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    
    for idx, (name, pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
            xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn']
        )
        axes[idx].set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
        
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrices to: {cm_path}")
    
    # 6. Plot Combined ROC Curves
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for (name, prob), color in zip(probabilities.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.500)")
    plt.title("ROC Curves Comparison (All Models)", fontsize=14, fontweight='bold')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC Curves to: {roc_path}")
    
    # 7. Plot Metrics Comparison Bar Chart
    plot_df = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette="tab10")
    plt.title("Model Performance Metrics Comparison", fontsize=14, fontweight='bold')
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, "model_comparison_bar_chart.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved Bar Chart to: {bar_path}")
    
    # 8. Select and Save the Champion Model
    # Both Random Forest and Logistic Regression are high performers;
    # Random Forest offers the highest AUC (0.8402) and highest Precision (0.6721).
    # Logistic Regression has the highest overall F1 (0.5939) and Accuracy (0.8020).
    best_model_name = "Logistic Regression"
    champion_model = lr_model
    
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(champion_model, best_model_path)
    print("\n" + "=" * 60)
    print(f"CHAMPION MODEL SELECTED: {best_model_name}")
    print(f"Saved to: {best_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_all_models()