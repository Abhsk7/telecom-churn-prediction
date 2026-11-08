"""
Exploratory Data Analysis (EDA) for Telecom Customer Churn
Verveox Technologies AI/ML Internship Project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11})

# Define paths
DATA_PATH = "data/telco_customer_churn.csv"
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def load_and_inspect_data():
    print("=" * 60)
    print("1. LOADING AND BASIC INSPECTION")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("First 3 rows:")
    print(df.head(3))
    print("\nColumn Data Types:")
    print(df.dtypes)
    return df

def clean_data_for_eda(df):
    print("\n" + "=" * 60)
    print("2. CHECKING MISSING & INVALID VALUES")
    print("=" * 60)
    
    # Check null values reported by pandas
    print("Initial Null values per column:")
    null_counts = df.isnull().sum()
    print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "No explicit NaN values found.")
    
    # TotalCharges is often loaded as object/string because of spaces ' '
    space_count = (df['TotalCharges'] == ' ').sum()
    print(f"\nBlank spaces found in TotalCharges: {space_count}")
    
    # Convert TotalCharges to float, turning spaces into NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    
    # For customers with tenure == 0, their TotalCharges is 0
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    print(f"Handled TotalCharges: Converted to float, filled missing with 0.0 (new customers with 0 tenure).")
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    return df

def analyze_churn_distribution(df):
    print("\n" + "=" * 60)
    print("3. CHURN TARGET DISTRIBUTION")
    print("=" * 60)
    
    churn_counts = df['Churn'].value_counts()
    churn_percent = df['Churn'].value_counts(normalize=True) * 100
    
    print("Churn counts:")
    for label, count in churn_counts.items():
        print(f"  {label}: {count} ({churn_percent[label]:.2f}%)")
        
    print("\nInsight: The dataset is imbalanced (~73% Stayed vs ~27% Churned).")
    print("This means Accuracy alone is not enough; we must evaluate Recall and F1-score.")

def generate_eda_visualizations(df):
    print("\n" + "=" * 60)
    print("4. GENERATING EDA CHARTS")
    print("=" * 60)
    
    # Chart 1: Churn Distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x='Churn', palette=['#2ecc71', '#e74c3c'])
    plt.title("Overall Churn Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Customer Churned?")
    plt.ylabel("Number of Customers")
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    chart1_path = os.path.join(FIGURES_DIR, "churn_distribution.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f" Saved: {chart1_path}")

    # Chart 2: Tenure vs Churn
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', palette=['#2ecc71', '#e74c3c'], kde=True)
    plt.title("Customer Tenure vs Churn", fontsize=14, fontweight='bold')
    plt.xlabel("Tenure (Months with Company)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    chart2_path = os.path.join(FIGURES_DIR, "tenure_vs_churn.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f" Saved: {chart2_path}")

    # Chart 3: Monthly Charges vs Churn
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df[df['Churn'] == 'No']['MonthlyCharges'], label='Stayed', shade=True, color='#2ecc71')
    sns.kdeplot(data=df[df['Churn'] == 'Yes']['MonthlyCharges'], label='Churned', shade=True, color='#e74c3c')
    plt.title("Monthly Charges Distribution by Churn Status", fontsize=14, fontweight='bold')
    plt.xlabel("Monthly Charges ($)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    chart3_path = os.path.join(FIGURES_DIR, "monthly_charges_vs_churn.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f" Saved: {chart3_path}")

    # Chart 4: Contract Type vs Churn
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Contract', hue='Churn', palette=['#2ecc71', '#e74c3c'])
    plt.title("Churn Rate by Contract Type", fontsize=14, fontweight='bold')
    plt.xlabel("Contract Type")
    plt.ylabel("Customer Count")
    plt.tight_layout()
    chart4_path = os.path.join(FIGURES_DIR, "contract_vs_churn.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f" Saved: {chart4_path}")

    print("\nAll EDA charts successfully generated and saved to reports/figures/!")

if __name__ == "__main__":
    df = load_and_inspect_data()
    df_cleaned = clean_data_for_eda(df)
    analyze_churn_distribution(df_cleaned)
    generate_eda_visualizations(df_cleaned)