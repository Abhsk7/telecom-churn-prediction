"""
Artificial Neural Network (ANN) Training using TensorFlow / Keras
Telecom Customer Churn Prediction - Verveox Project 5
"""

import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Import our preprocessing pipeline
from data_preprocessing import preprocess_and_split

MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def build_ann_model(input_dim):
    """
    Constructs a Deep Artificial Neural Network:
    - 3 Hidden Dense layers with ReLU
    - Batch Normalization and Dropout for regularization
    - 1 Output neuron with Sigmoid activation for binary classification
    """
    model = Sequential([
        # Hidden Layer 1
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden Layer 2
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden Layer 3
        Dense(16, activation='relu'),
        Dropout(0.1),
        
        # Output Layer (Binary Classification: Churn Probability)
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model using Adam optimizer and Binary Cross-Entropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def train_and_evaluate_ann():
    # 1. Load preprocessed and scaled data
    X_train, X_test, y_train, y_test, selected_features = preprocess_and_split()
    input_dim = X_train.shape[1]
    
    print("\n" + "=" * 60)
    print("BUILDING ARTIFICIAL NEURAL NETWORK (ANN)")
    print("=" * 60)
    print(f"Number of input features: {input_dim}")
    
    model = build_ann_model(input_dim)
    model.summary()
    
    # 2. Setup EarlyStopping to prevent overfitting
    ann_model_path = os.path.join(MODELS_DIR, "ann_model.keras")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=ann_model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 3. Train the ANN
    print("\nStarting ANN training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 4. Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ANN ON TEST SET")
    print("=" * 60)
    
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    ann_metrics = {
        "Model": "ANN",
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4),
        "Best_Params": "64-32-16 Dense + Dropout + Adam"
    }
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Save ANN metrics to CSV
    ann_df = pd.DataFrame([ann_metrics])
    ann_metrics_path = os.path.join(REPORTS_DIR, "ann_metrics.csv")
    ann_df.to_csv(ann_metrics_path, index=False)
    print(f"\nANN metrics saved to: {ann_metrics_path}")
    print(f"Trained ANN model saved to: {ann_model_path}")
    
    return model, history

if __name__ == "__main__":
    train_and_evaluate_ann()