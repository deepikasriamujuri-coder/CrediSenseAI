import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_visualizations():
    cleaned_path = os.path.join("data", "processed", "credit_risk_cleaned.csv")
    img_dir = os.path.join("static", "images", "eda")
    os.makedirs(img_dir, exist_ok=True)
    
    if not os.path.exists(cleaned_path):
        print(f"Error: Cleaned dataset not found at {cleaned_path}")
        return False
        
    df = pd.read_csv(cleaned_path)
    
    # Set modern styling style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 150
    })
    
    # Color palette
    colors = ["#2A9D8F", "#E76F51"]  # Teal (0) and Coral (1)
    
    # 1. Target Class Distribution
    plt.figure(figsize=(6, 4.5))
    counts = df["loan_status"].value_counts()
    percentages = df["loan_status"].value_counts(normalize=True) * 100
    ax = sns.barplot(x=counts.index, y=counts.values, palette=colors, hue=counts.index, legend=False)
    plt.title("Credit Risk Class Distribution (loan_status)")
    plt.xlabel("Risk Class (0: Lower Risk, 1: Higher Risk)")
    plt.ylabel("Record Count")
    plt.xticks([0, 1], ["0 (Lower Risk)", "1 (Higher Risk)"])
    for i, p in enumerate(counts.values):
        ax.annotate(f"{p}\n({percentages[i]:.1f}%)", (i, p/2), ha='center', va='center', color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "target_distribution.png"), dpi=200)
    plt.close()
    
    # 2. Missing-value analysis
    plt.figure(figsize=(8, 4))
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_data = pd.DataFrame({"Missing Count": missing, "Percentage": missing_pct})
    missing_data = missing_data[missing_data["Missing Count"] > 0]
    
    # If no missing values are present in processed, analyze raw just to save a chart, or plot empty bars.
    # Since we didn't fill missing values yet (it is left to the preprocessing pipeline), missing values still exist!
    if len(missing_data) > 0:
        ax = sns.barplot(x=missing_data.index, y=missing_data["Percentage"], palette="crest", hue=missing_data.index, legend=False)
        plt.title("Missing Values Percentage by Feature")
        plt.ylabel("Missing (%)")
        plt.xlabel("Features")
        for i, val in enumerate(missing_data["Percentage"].values):
            ax.text(i, val + 0.5, f"{val:.2f}%", ha='center', va='bottom', fontweight='bold')
        plt.ylim(0, 12)
    else:
        plt.text(0.5, 0.5, "No Missing Values in Cleaned Dataset", ha='center', va='center', fontsize=12)
        plt.title("Missing Values Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "missing_values.png"), dpi=200)
    plt.close()
    
    # Helper list of categorical and numerical features
    num_features = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
    cat_features = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    
    # 3. Numerical Feature Distributions (Melted boxplot/violin style or individual subplots)
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(num_features, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(y=df[col], color="#2A9D8F")
        plt.title(f"{col} Distribution")
        plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "numerical_distributions.png"), dpi=200)
    plt.close()
    
    # 4. Categorical Feature Distributions
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(cat_features, 1):
        plt.subplot(2, 2, i)
        sns.countplot(x=df[col], palette="Set2", order=df[col].value_counts().index, hue=df[col], legend=False)
        plt.title(f"{col} Distribution")
        plt.xticks(rotation=15)
        plt.xlabel("")
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "categorical_distributions.png"), dpi=200)
    plt.close()
    
    # 5. Correlation Heatmap (only numerical features)
    plt.figure(figsize=(8, 6))
    corr = df[num_features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "correlation_heatmap.png"), dpi=200)
    plt.close()
    
    # Helper function for bivariate stacked percentage bars
    def plot_stacked_pct(col, filename, title):
        ct = pd.crosstab(df[col], df["loan_status"], normalize="index") * 100
        plt.figure(figsize=(8, 5))
        ax = ct.plot(kind="bar", stacked=True, color=colors, figsize=(8, 5))
        plt.title(title)
        plt.ylabel("Percentage (%)")
        plt.xlabel(col)
        plt.xticks(rotation=15)
        plt.legend(["0 (Lower Risk)", "1 (Higher Risk)"], title="Risk Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add labels inside bars
        for container in ax.containers:
            labels = [f"{v.get_height():.1f}%" if v.get_height() > 5 else '' for v in container]
            ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, filename), dpi=200)
        plt.close()
        
    # 6. Loan status by home ownership
    plot_stacked_pct("person_home_ownership", "status_by_home_ownership.png", "Credit Risk Status by Home Ownership")
    
    # 7. Loan status by loan intent
    plot_stacked_pct("loan_intent", "status_by_loan_intent.png", "Credit Risk Status by Loan Intent")
    
    # 8. Loan status by loan grade
    plot_stacked_pct("loan_grade", "status_by_loan_grade.png", "Credit Risk Status by Loan Grade (Leaked Target Proxy)")
    
    # 9. Loan status by previous default history
    plot_stacked_pct("cb_person_default_on_file", "status_by_previous_default.png", "Credit Risk Status by Previous Default File")
    
    # Helper function to plot individual distributions
    def plot_dist(col, filename, title, xlabel):
        plt.figure(figsize=(7, 4.5))
        sns.histplot(data=df, x=col, hue="loan_status", kde=True, bins=35, palette=colors, element="step", stat="density", common_norm=False)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.legend(["1 (Higher Risk)", "0 (Lower Risk)"], title="Risk Class")
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, filename), dpi=200)
        plt.close()

    # 10. Income distribution (using log-scale or clipping for visibility due to extreme outliers)
    plt.figure(figsize=(7, 4.5))
    # We clip the visualization display of income to 200k to ensure readability
    sns.histplot(data=df[df["person_income"] <= 200000], x="person_income", hue="loan_status", kde=True, bins=35, palette=colors, element="step", stat="density", common_norm=False)
    plt.title("Income Distribution (Filtered to <= $200k for Readability)")
    plt.xlabel("Annual Income (USD)")
    plt.ylabel("Density")
    plt.legend(["1 (Higher Risk)", "0 (Lower Risk)"], title="Risk Class")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "income_distribution.png"), dpi=200)
    plt.close()
    
    # 11. Loan amount distribution
    plot_dist("loan_amnt", "loan_amount_distribution.png", "Loan Amount Requested Distribution", "Loan Amount ($)")
    
    # 12. Interest rate distribution (ignoring NaNs for kdeplot)
    plt.figure(figsize=(7, 4.5))
    sns.histplot(data=df.dropna(subset=["loan_int_rate"]), x="loan_int_rate", hue="loan_status", kde=True, bins=35, palette=colors, element="step", stat="density", common_norm=False)
    plt.title("Interest Rate Distribution")
    plt.xlabel("Interest Rate (%)")
    plt.ylabel("Density")
    plt.legend(["1 (Higher Risk)", "0 (Lower Risk)"], title="Risk Class")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "interest_rate_distribution.png"), dpi=200)
    plt.close()
    
    # 13. Loan-to-income ratio distribution
    plot_dist("loan_percent_income", "loan_to_income_distribution.png", "Loan-to-Income Ratio Distribution", "Loan / Annual Income Ratio")
    
    # 14. Age distribution
    plot_dist("person_age", "age_distribution.png", "Applicant Age Distribution", "Age (Years)")
    
    # 15. Employment length distribution (ignoring NaNs)
    plt.figure(figsize=(7, 4.5))
    sns.histplot(data=df.dropna(subset=["person_emp_length"]), x="person_emp_length", hue="loan_status", kde=True, bins=30, palette=colors, element="step", stat="density", common_norm=False)
    plt.title("Employment Length Distribution")
    plt.xlabel("Employment Length (Years)")
    plt.ylabel("Density")
    plt.legend(["1 (Higher Risk)", "0 (Lower Risk)"], title="Risk Class")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "employment_length_distribution.png"), dpi=200)
    plt.close()
    
    print(f"All 15 EDA visualizations generated successfully and saved to: {img_dir}")
    return True

if __name__ == "__main__":
    generate_visualizations()
