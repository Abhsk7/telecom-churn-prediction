# 🎓 Telecom Customer Churn Prediction System
## Comprehensive Internship Project Report & Viva / Interview Guide
**Verveox Technologies AI/ML Internship — Project 5 Specification**

---

## 1. PROJECT OVERVIEW
- **Project Title:** Telecom Customer Churn Prediction System Using Machine Learning and Artificial Neural Networks
- **Objective:** Build an end-to-end binary classification system to accurately predict customer churn risk, quantify churn probability, and provide data-driven retention recommendations.
- **Dataset:** IBM Telco Customer Churn Dataset (7,043 genuine customer records, 21 attributes).
- **Target Variable:** `Churn` (Binary: `1` = Churned, `0` = Remained).

---

## 2. DATA ENGINEERING & PREPROCESSING METHODOLOGY

### A. Data Cleaning:
- **Identifier Removal:** Dropped `customerID` as it contains arbitrary unique alphanumeric strings with zero predictive capability.
- **Handling Invalid Values (`TotalCharges`):** 11 rows had blank whitespace characters (`" "`) representing brand new customers with tenure = 0 months. Converted these to numeric and imputed with `0.0`.

### B. Categorical Encoding:
- Categorical features (such as `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`) were converted using **One-Hot Encoding** (`pd.get_dummies`) with `drop_first=True` to prevent multicollinearity (the dummy variable trap).

### C. Feature Scaling:
- Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were standardized using **`StandardScaler`** ($z = \frac{x - \mu}{\sigma}$).
- **Crucial Rule:** The scaler was fitted strictly on the training set (`X_train`) and only applied (`transform`) to the test set to completely prevent **Data Leakage**.

### D. Feature Selection:
- Applied statistical **`SelectKBest`** with ANOVA F-statistic (`f_classif`) to extract the top 18 most statistically significant features.
- Primary predictors identified: `Contract_One year`, `tenure`, `MonthlyCharges`, `TotalCharges`, and `TechSupport_Yes`.

---

## 3. MODEL ARCHITECTURE & EVALUATION

### Model Benchmark on Unseen Test Data (1,409 Samples):

| Algorithm | Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear / Probabilistic | **80.20%** | 65.18% | 54.55% | **0.5939** | 0.8389 |
| **ANN (TensorFlow)** | Deep Learning | 79.28% | 62.13% | **56.15%** | 0.5899 | 0.8346 |
| **Random Forest** | Ensemble Trees | 79.49% | **67.21%** | 44.39% | 0.5346 | **0.8402** |
| **KNN** | Distance-Based | 78.78% | 61.06% | 55.35% | 0.5806 | 0.8118 |

---

## 4. TOP 10 VIVA & INTERVIEW QUESTIONS WITH ANSWERS

### Q1: Why is Accuracy alone an unreliable metric for Customer Churn?
**Answer:**  
In telecom churn, the dataset is imbalanced (~73.5% non-churn vs ~26.5% churn). If a model simply predicts that every customer will stay, it would achieve 73.5% accuracy while completely failing to identify a single churning customer. Therefore, we prioritize **Recall** (catching churners), **Precision** (avoiding false alarms), **F1-Score** (harmonic mean), and **ROC-AUC** (discriminative power across all thresholds).

### Q2: Why did you scale numerical features using StandardScaler?
**Answer:**  
Distance-based algorithms like **K-Nearest Neighbors (KNN)** compute Euclidean distances between data points. If `TotalCharges` ranges from $0 to $8,000 while `tenure` ranges from 0 to 72, the distance metric would be overwhelmingly dominated by TotalCharges. Feature scaling ensures all features contribute proportionally. It also prevents exploding or vanishing gradients during **ANN** backpropagation and **Logistic Regression** convergence.

### Q3: What is Data Leakage and how did your pipeline prevent it?
**Answer:**  
Data leakage occurs when information from the test dataset contaminates the training phase, giving falsely optimistic test scores that fail in production. We prevented this by splitting data into train and test sets **before** scaling and feature selection. The `StandardScaler` and `SelectKBest` were fitted **strictly on `X_train`** (`fit_transform`) and only used to transform `X_test` (`transform`).

### Q4: Why did you use Stratified K-Fold Cross-Validation?
**Answer:**  
Because churn is an imbalanced binary class, regular random splitting could lead to folds with disproportionate numbers of churners. **Stratified K-Fold** guarantees that every fold preserves the exact 73.5% to 26.5% target distribution, providing a realistic estimate of model generalization during hyperparameter tuning.

### Q5: How does your Artificial Neural Network (ANN) differ from traditional ML?
**Answer:**  
Traditional ML models (Logistic Regression, KNN) map features using explicit mathematical boundaries or distance metrics. Our **ANN** utilizes multiple stacked dense layers with non-linear activation functions (ReLU). The hidden layers learn higher-level hierarchical representations of feature interactions. The output layer uses a Sigmoid activation to compress the output into a calibrated probability distribution between 0.0 and 1.0.

### Q6: What was the purpose of Dropout and EarlyStopping in the ANN?
**Answer:**  
- **Dropout (0.3 and 0.2):** Randomly deactivates a fraction of neurons during training to prevent co-adaptation of features and combat overfitting.
- **EarlyStopping:** Monitors validation loss (`val_loss`) during training and halts execution when loss ceases to improve for 10 consecutive epochs (`patience=10`), automatically restoring the best network weights.

### Q7: Why was Logistic Regression selected as the final Champion Model?
**Answer:**  
While Random Forest had slightly higher precision and the ANN had slightly higher recall, **Logistic Regression achieved the highest overall F1-Score (0.5939)** and **highest Accuracy (80.20%)**, along with an outstanding **ROC-AUC of 0.8389**. It strikes the best operational balance between catching churners and minimizing unnecessary retention discount expenditures.

### Q8: How does the trained model integrate into the Streamlit dashboard?
**Answer:**  
We saved the fitted preprocessing pipeline (`preprocessor.joblib`) and trained models (`best_model.joblib`, `ann_model.keras`). When an end-user inputs customer attributes in Streamlit, the app packages the inputs into a single-row DataFrame, transforms it using the saved `StandardScaler` and `SelectKBest`, and feeds the processed vector directly to the selected model for real-time inference.

### Q9: What are the biggest real-world drivers of customer churn discovered during EDA?
**Answer:**  
1. **Contract Type:** Month-to-month contracts have the highest churn rate by far.
2. **Tenure:** The first 6 months have peak churn risk; customer lifetime value stabilizes after month 12.
3. **Monthly Charges:** High bills without value-added security or tech support accelerate customer departure.

### Q10: How would you enhance this system for production in a telecom company?
**Answer:**  
1. Implement automated cost-sensitive classification by assigning higher financial penalties to False Negatives (lost customers) than False Positives (unnecessary promotional discounts).
2. Integrate real-time call center logs using NLP sentiment analysis.
3. Deploy the application in a Docker container on Google Cloud Run or AWS ECS with automated CI/CD retraining pipelines.