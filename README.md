# 📡 Telecom Customer Churn Prediction System

**AI/ML Internship Project 5**  
**Organization:** Verveox Technologies  
**Domain:** Artificial Intelligence & Machine Learning  
**Application:** End-to-End Predictive Analytics Dashboard  

---

## 📌 Executive Summary
Customer churn occurs when telecom subscribers discontinue their services. Because acquiring new customers costs 5 to 25 times more than retaining existing ones, accurately identifying at-risk customers enables telecom providers to execute targeted retention strategies (discounts, contract upgrades, customer support).

This project implements an academically sound, end-to-end Machine Learning and Deep Learning system that predicts customer churn probabilities using real-world customer subscription, billing, and demographic attributes.

---

## 🛠️ Technology Stack
- **Language:** Python 3.11
- **Data Engineering:** Pandas, NumPy
- **Visualizations:** Matplotlib, Seaborn
- **Traditional Machine Learning:** Scikit-learn (Logistic Regression, K-Nearest Neighbors, Random Forest)
- **Deep Learning:** TensorFlow / Keras (Artificial Neural Network - ANN)
- **Model Optimization:** 5-Fold Stratified Cross-Validation, GridSearchCV, ANOVA F-Score Feature Selection (`SelectKBest`)
- **Web Dashboard:** Streamlit
- **Serialization:** Joblib, Keras HDF5/Native format

---

## 📂 Project Architecture

```text
verveox project 1/
│
├── data/
│   └── telco_customer_churn.csv            # 7,043 real IBM Telco customer records
│
├── models/
│   ├── best_model.joblib                   # Champion model (Logistic Regression)
│   ├── logistic_regression.joblib          # Tuned Logistic Regression model
│   ├── knn.joblib                          # Tuned K-Nearest Neighbors model
│   ├── random_forest.joblib                # Tuned Random Forest Classifier
│   ├── ann_model.keras                     # 3-Layer Deep Artificial Neural Network
│   └── preprocessor.joblib                 # Scaler, Feature Selector & Column metadata
│
├── src/
│   ├── eda.py                              # Data inspection, cleaning & chart generation
│   ├── data_preprocessing.py               # Preprocessing pipeline & feature selection
│   ├── train_models.py                     # Traditional ML training with GridSearchCV
│   ├── train_ann.py                        # TensorFlow ANN model training
│   └── evaluate_models.py                  # Evaluation, comparison tables & ROC curves
│
├── app/
│   └── app.py                              # Interactive Streamlit Web Application
│
├── reports/
│   ├── figures/                            # High-resolution evaluation and EDA plots
│   ├── master_model_comparison.csv         # Comparative metrics table
│   └── traditional_models_metrics.csv      # Baseline traditional ML metrics
│
├── requirements.txt                        # Dependency specifications
└── README.md                               # Project documentation