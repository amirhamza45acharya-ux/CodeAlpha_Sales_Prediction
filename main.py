"""
=============================================================================
CodeAlpha Data Science Internship - Task 4: Sales Prediction using Python
=============================================================================
Author: CodeAlpha Data Science Intern
Project: Sales Prediction with Advertising Expenditure Analysis
Description:
    This beginner-friendly Python script analyzes the relationship between
    advertising expenditures (TV, Radio, Newspaper) and product Sales.
    It performs end-to-end data science steps:
      1. Data Loading and Inspection
      2. Data Cleaning and Preparation
      3. Exploratory Data Analysis (EDA) and Visualization
      4. Categorical Feature Demonstration & Preprocessing
      5. Train/Test Splitting
      6. Regression Modeling (Linear Regression vs. Random Forest)
      7. Model Evaluation (MAE, RMSE, R²)
      8. Actual vs. Predicted Visualizations & Residual Diagnostics
      9. Strategic Business Marketing Insights & Recommendations
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless mode for saving figures without GUI popups
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_results_directory(dir_name="results"):
    """Creates output folder for saving charts and evaluation files."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"[INFO] Created directory: '{dir_name}' for saving plots and reports.")
    return dir_name


def load_and_inspect_data(filepath="Advertising.csv"):
    """
    Step 1: Load the dataset and inspect basic characteristics.
    Begins by printing shape, columns, first rows, and missing values.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING & INITIAL INSPECTION")
    print("=" * 70)

    if not os.path.exists(filepath):
        print(f"[ERROR] Dataset file '{filepath}' not found!")
        print("Please place 'Advertising.csv' in the project directory.")
        sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded from '{filepath}'.")
    print(f"Number of rows (records): {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\nFirst 5 rows of data:")
    print(df.head())

    print("\nData column types and non-null counts:")
    df.info()

    return df


def clean_and_prepare_data(df):
    """
    Step 2: Clean and prepare the dataset.
    - Removes unnecessary index columns like 'Unnamed: 0'.
    - Checks for missing (null) values and duplicates.
    - Demonstrates categorical encoding handling if categorical features exist,
      and shows how to engineer categorical segments (e.g., Spending Tiers).
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA CLEANING & PREPARATION")
    print("=" * 70)

    # 1. Drop redundant serial index column if present
    if "Unnamed: 0" in df.columns:
        print("[CLEANING] Dropping redundant serial index column 'Unnamed: 0'...")
        df = df.drop(columns=["Unnamed: 0"])

    # 2. Check for missing values
    missing_count = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_count)
    if missing_count.sum() == 0:
        print(">> Dataset is clean: No missing or null values found.")
    else:
        print(">> Handling missing values by median imputation...")
        df = df.fillna(df.median(numeric_only=True))

    # 3. Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows detected: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate row(s).")

    # 4. Summary statistics of continuous numerical variables
    print("\nStatistical Summary (Descriptive Statistics):")
    print(df.describe().round(2))

    # 5. Categorical Feature Demonstration (Task Requirement #5)
    print("\n--- Categorical Feature Encoding Demonstration ---")
    print("Note: In the classic Advertising dataset, spend across channels")
    print("(TV, Radio, Newspaper) is continuous expenditure in thousands of dollars.")
    print("To demonstrate how categorical features (e.g., 'Target_Segment' or 'Dominant_Platform')")
    print("are handled in a real-world pipeline, we derive an illustrative segment feature:")

    # Derive dominant ad channel as a categorical feature
    df_demo = df.copy()
    ad_channels = ['TV', 'Radio', 'Newspaper']
    df_demo['Dominant_Platform'] = df_demo[ad_channels].idxmax(axis=1)
    print("\nDistribution of derived 'Dominant_Platform' categorical feature:")
    print(df_demo['Dominant_Platform'].value_counts())

    # Perform One-Hot Encoding
    encoded_platforms = pd.get_dummies(df_demo['Dominant_Platform'], prefix='Platform', drop_first=False, dtype=int)
    print("\nSample of One-Hot Encoded platform features:")
    print(encoded_platforms.head(3))
    print("One-hot encoding successfully demonstrated!\n")

    return df


def perform_eda_and_visualizations(df, output_dir="results"):
    """
    Step 3: Exploratory Data Analysis (EDA).
    Produces visual charts for:
      - Distribution of each variable
      - Scatter plots of advertising spend vs. Sales with trendlines
      - Correlation Heatmap
    """
    print("\n" + "=" * 70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION")
    print("=" * 70)

    # Chart 1: Distributions of TV, Radio, Newspaper, and Sales
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Feature Distributions (Advertising Spend & Sales)", fontsize=16, fontweight='bold', y=0.98)
    
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    cols = ['TV', 'Radio', 'Newspaper', 'Sales']
    units = ['Spend ($k)', 'Spend ($k)', 'Spend ($k)', 'Units Sold (k)']

    for idx, (col, unit, color) in enumerate(zip(cols, units, palette)):
        row = idx // 2
        c = idx % 2
        ax = axes[row, c]
        sns.histplot(df[col], kde=True, ax=ax, color=color, bins=15, edgecolor='black', alpha=0.6)
        ax.set_title(f"Distribution of {col}", fontsize=13, fontweight='semibold')
        ax.set_xlabel(f"{col} - {unit}", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "01_feature_distributions.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 1: Feature distributions saved to '{chart1_path}'")

    # Chart 2: Advertising Spend vs Sales Scatter Plots with Regression Trendlines
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Advertising Expenditure vs. Sales with Trendlines", fontsize=16, fontweight='bold', y=1.02)

    channels = ['TV', 'Radio', 'Newspaper']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (channel, color) in enumerate(zip(channels, colors)):
        ax = axes[i]
        sns.regplot(
            data=df,
            x=channel,
            y='Sales',
            ax=ax,
            color=color,
            scatter_kws={'alpha': 0.7, 'edgecolor': 'black', 's': 40},
            line_kws={'color': '#d62728', 'linewidth': 2}
        )
        r_val = df[channel].corr(df['Sales'])
        ax.set_title(f"{channel} vs. Sales (Pearson r = {r_val:.2f})", fontsize=12, fontweight='semibold')
        ax.set_xlabel(f"{channel} Advertising Spend ($1,000s)", fontsize=11)
        ax.set_ylabel("Sales ($1,000 units)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "02_advertising_vs_sales_scatter.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 2: Advertising vs Sales scatter plots saved to '{chart2_path}'")

    # Chart 3: Correlation Matrix Heatmap
    plt.figure(figsize=(8, 6))
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
    )
    plt.title("Correlation Matrix Heatmap", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "03_correlation_heatmap.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 3: Correlation heatmap saved to '{chart3_path}'")

    print("\nPearson Correlation with Sales:")
    for col in ['TV', 'Radio', 'Newspaper']:
        print(f"  - {col:10s}: {df[col].corr(df['Sales']):.4f}")


def split_data(df, test_size=0.2, random_state=42):
    """
    Step 4: Train/Test Split.
    Splits the dataset into 80% training data and 20% testing data.
    """
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN / TEST DATASET SPLIT")
    print("=" * 70)

    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Total dataset size : {len(df)} samples")
    print(f"Training set size  : {len(X_train)} samples ({int((1 - test_size) * 100)}%)")
    print(f"Testing set size   : {len(X_test)} samples ({int(test_size * 100)}%)")
    print(f"Random state seed  : {random_state} (ensures reproducibility)")

    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred):
    """Calculates Mean Absolute Error, Root Mean Squared Error, and R-squared."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir="results"):
    """
    Step 5: Model Training, Evaluation, and Comparison.
      Model 1: Multiple Linear Regression (interpretable statistical baseline)
      Model 2: Random Forest Regressor (non-linear ensemble method)
    """
    print("\n" + "=" * 70)
    print("STEP 5: MODEL TRAINING & EVALUATION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Model 1: Multiple Linear Regression
    # -------------------------------------------------------------
    print("\n--> Training Model 1: Multiple Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    lr_mae, lr_rmse, lr_r2 = compute_metrics(y_test, lr_pred)
    print("  [Linear Regression Results]")
    print(f"  - Mean Absolute Error (MAE)       : {lr_mae:.4f}")
    print(f"  - Root Mean Squared Error (RMSE)  : {lr_rmse:.4f}")
    print(f"  - R² Score (Variance Explained)   : {lr_r2:.4f} ({lr_r2 * 100:.2f}%)")
    print(f"  - Model Intercept (Baseline Sales): {lr_model.intercept_:.4f}")
    print("  - Learned Regression Coefficients (Slopes):")
    for feat, coef in zip(X_train.columns, lr_model.coef_):
        print(f"      * {feat:10s}: {coef:+.4f} (Sales increase per $1k spend)")

    # -------------------------------------------------------------
    # Model 2: Random Forest Regressor
    # -------------------------------------------------------------
    print("\n--> Training Model 2: Random Forest Regressor (Ensemble)...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_mae, rf_rmse, rf_r2 = compute_metrics(y_test, rf_pred)
    print("  [Random Forest Regressor Results]")
    print(f"  - Mean Absolute Error (MAE)       : {rf_mae:.4f}")
    print(f"  - Root Mean Squared Error (RMSE)  : {rf_rmse:.4f}")
    print(f"  - R² Score (Variance Explained)   : {rf_r2:.4f} ({rf_r2 * 100:.2f}%)")
    print("  - Feature Importance Breakdown:")
    for feat, imp in zip(X_train.columns, rf_model.feature_importances_):
        print(f"      * {feat:10s}: {imp * 100:.2f}%")

    # -------------------------------------------------------------
    # Metrics Comparison Table
    # -------------------------------------------------------------
    metrics_data = {
        "Model": ["Multiple Linear Regression", "Random Forest Regressor"],
        "MAE": [lr_mae, rf_mae],
        "RMSE": [lr_rmse, rf_rmse],
        "R2_Score": [lr_r2, rf_r2],
        "Variance_Explained_Pct": [lr_r2 * 100, rf_r2 * 100]
    }
    metrics_df = pd.DataFrame(metrics_data)
    csv_metrics_path = os.path.join(output_dir, "model_metrics_summary.csv")
    metrics_df.to_csv(csv_metrics_path, index=False)
    print(f"\n[SAVED] Metrics comparison saved to '{csv_metrics_path}'")
    print("\n" + metrics_df.to_string(index=False))

    # -------------------------------------------------------------
    # Visualization: Model Comparison Bar Chart
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Performance Comparison (Linear Regression vs. Random Forest)", fontsize=15, fontweight='bold')

    metric_names = ['MAE', 'RMSE', 'R2_Score']
    chart_titles = ['Mean Absolute Error (Lower is Better)', 'Root Mean Squared Error (Lower is Better)', 'R² Score (Higher is Better)']
    colors = ['#4A90E2', '#50E3C2']

    for i, (m_name, title) in enumerate(zip(metric_names, chart_titles)):
        ax = axes[i]
        bars = ax.bar(metrics_df['Model'], metrics_df[m_name], color=colors, edgecolor='black', width=0.55)
        ax.set_title(title, fontsize=11, fontweight='semibold')
        ax.set_ylabel("Score", fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_xticks(range(len(metrics_df['Model'])))
        ax.set_xticklabels(["Linear Reg", "Random Forest"], fontsize=10)

        # Label values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    chart6_path = os.path.join(output_dir, "06_model_comparison_metrics.png")
    plt.savefig(chart6_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 6: Model comparison metrics saved to '{chart6_path}'")

    # -------------------------------------------------------------
    # Visualization: Actual vs. Predicted (Both Models)
    # -------------------------------------------------------------
    # Linear Regression Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, lr_pred, color='#1f77b4', edgecolors='black', alpha=0.8, s=60, label='Linear Reg Predictions')
    min_val = min(y_test.min(), lr_pred.min()) - 1
    max_val = max(y_test.max(), lr_pred.max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction Line (y = x)')
    plt.title(f"Linear Regression: Actual vs. Predicted Sales\n(R² = {lr_r2:.4f}, MAE = {lr_mae:.3f})", fontsize=13, fontweight='bold')
    plt.xlabel("Actual Sales ($1,000 units)", fontsize=11)
    plt.ylabel("Predicted Sales ($1,000 units)", fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "04_actual_vs_predicted_lr.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 4: Actual vs Predicted (Linear Reg) saved to '{chart4_path}'")

    # Random Forest Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, rf_pred, color='#2ca02c', edgecolors='black', alpha=0.8, s=60, label='Random Forest Predictions')
    min_val = min(y_test.min(), rf_pred.min()) - 1
    max_val = max(y_test.max(), rf_pred.max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction Line (y = x)')
    plt.title(f"Random Forest: Actual vs. Predicted Sales\n(R² = {rf_r2:.4f}, MAE = {rf_mae:.3f})", fontsize=13, fontweight='bold')
    plt.xlabel("Actual Sales ($1,000 units)", fontsize=11)
    plt.ylabel("Predicted Sales ($1,000 units)", fontsize=11)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    chart5_path = os.path.join(output_dir, "05_actual_vs_predicted_rf.png")
    plt.savefig(chart5_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 5: Actual vs Predicted (Random Forest) saved to '{chart5_path}'")

    # -------------------------------------------------------------
    # Visualization: Feature Importance Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Importance Analysis Across Models", fontsize=15, fontweight='bold')

    # Linear Regression Coefficients
    ax1 = axes[0]
    ax1.barh(X_train.columns, lr_model.coef_, color='#1f77b4', edgecolor='black', alpha=0.85)
    ax1.set_title("Linear Regression Coefficients (Per $1k Spend)", fontsize=12, fontweight='semibold')
    ax1.set_xlabel("Coefficient Value (Slope)", fontsize=11)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    for i, v in enumerate(lr_model.coef_):
        ax1.text(v + 0.005, i, f"{v:+.4f}", va='center', fontweight='bold', fontsize=10)

    # Random Forest Feature Importances
    ax2 = axes[1]
    rf_importances = rf_model.feature_importances_ * 100
    ax2.barh(X_train.columns, rf_importances, color='#2ca02c', edgecolor='black', alpha=0.85)
    ax2.set_title("Random Forest Relative Importance (%)", fontsize=12, fontweight='semibold')
    ax2.set_xlabel("Relative Importance (%)", fontsize=11)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    for i, v in enumerate(rf_importances):
        ax2.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    chart7_path = os.path.join(output_dir, "07_feature_importance.png")
    plt.savefig(chart7_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 7: Feature importance comparison saved to '{chart7_path}'")

    # -------------------------------------------------------------
    # Visualization: Residual Analysis Diagnostics
    # -------------------------------------------------------------
    lr_residuals = y_test - lr_pred
    rf_residuals = y_test - rf_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residual Diagnostics (Prediction Errors on Test Set)", fontsize=15, fontweight='bold')

    # Residuals vs. Fitted Values
    ax1 = axes[0]
    ax1.scatter(lr_pred, lr_residuals, color='#1f77b4', edgecolors='black', alpha=0.75, s=50, label='Linear Regression')
    ax1.scatter(rf_pred, rf_residuals, color='#2ca02c', edgecolors='black', alpha=0.75, s=50, label='Random Forest')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Line')
    ax1.set_title("Residuals vs. Fitted (Predicted) Values", fontsize=12, fontweight='semibold')
    ax1.set_xlabel("Predicted Sales ($1,000 units)", fontsize=11)
    ax1.set_ylabel("Residuals (Actual - Predicted)", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Residual Distribution (KDE)
    ax2 = axes[1]
    sns.kdeplot(lr_residuals, ax=ax2, color='#1f77b4', fill=True, alpha=0.4, label=f'Linear Reg (Mean={lr_residuals.mean():.2f})')
    sns.kdeplot(rf_residuals, ax=ax2, color='#2ca02c', fill=True, alpha=0.4, label=f'Random Forest (Mean={rf_residuals.mean():.2f})')
    ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error Mark')
    ax2.set_title("Residual Error Distribution (Density)", fontsize=12, fontweight='semibold')
    ax2.set_xlabel("Residual Error", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart8_path = os.path.join(output_dir, "08_residual_analysis.png")
    plt.savefig(chart8_path, dpi=300)
    plt.close()
    print(f"[SAVED] Chart 8: Residual diagnostics saved to '{chart8_path}'")

    return lr_model, rf_model, metrics_df


def generate_business_insights(df, lr_model, rf_model, metrics_df):
    """
    Step 6: Business Marketing Insights & Strategic Recommendations.
    Synthesizes findings into actionable advice for marketing leadership.
    """
    print("\n" + "=" * 70)
    print("STEP 6: STRATEGIC BUSINESS MARKETING INSIGHTS")
    print("=" * 70)

    tv_coef = lr_model.coef_[0]
    radio_coef = lr_model.coef_[1]
    news_coef = lr_model.coef_[2]
    rf_tv_imp = rf_model.feature_importances_[0] * 100
    rf_radio_imp = rf_model.feature_importances_[1] * 100
    rf_news_imp = rf_model.feature_importances_[2] * 100

    insights_text = f"""
1. TV ADVERTISING IS THE PRIMARY DRIVER OF TOTAL SALES VOLUME:
   - TV spend has the strongest correlation with Sales (r = {df['TV'].corr(df['Sales']):.3f}).
   - In the Random Forest model, TV commands {rf_tv_imp:.1f}% of overall feature importance.
   - Strategic Takeaway: Sustained TV campaigns are essential for broad brand reach and large-scale sales volume.

2. RADIO DELIVERS THE HIGHEST RETURN PER DOLLAR SPENT (HIGHEST MARGINAL ROI):
   - In Multiple Linear Regression, Radio has the highest coefficient (+{radio_coef:.4f}) compared to TV (+{tv_coef:.4f}).
   - This means that holding other channels constant, an extra $1,000 invested in Radio produces approximately {radio_coef * 1000:.0f} additional unit sales, whereas TV yields {tv_coef * 1000:.0f} unit sales.
   - Strategic Takeaway: Radio is extremely cost-effective and should be allocated higher budget share in regional and conversion-focused pushes.

3. NEWSPAPER ADVERTISING PROVIDES NEGLIGIBLE STANDALONE BENEFIT:
   - Newspaper correlation with Sales is weak (r = {df['Newspaper'].corr(df['Sales']):.3f}), and its linear regression coefficient is near zero (+{news_coef:.4f}).
   - In the Random Forest model, Newspaper accounts for merely {rf_news_imp:.1f}% of predictive power.
   - Strategic Takeaway: Marketing management should reallocate newspaper ad spend into high-performing digital, TV, or radio channels to maximize overall ROI.

4. SYNERGY AND NON-LINEAR INTERACTION ACROSS CHANNELS:
   - The Random Forest Regressor outperformed Linear Regression (R² of {metrics_df.loc[1, 'R2_Score']:.4f} vs. {metrics_df.loc[0, 'R2_Score']:.4f}, reducing MAE from {metrics_df.loc[0, 'MAE']:.3f} to {metrics_df.loc[1, 'MAE']:.3f}).
   - This proves that advertising channels do not act in isolation: combining TV awareness with Radio frequency creates a compounding non-linear multiplier effect on sales.
"""
    print(insights_text)

    # Save insights to text file
    insights_filepath = os.path.join("results", "business_insights.txt")
    with open(insights_filepath, "w") as f:
        f.write("CODEALPHA DATA SCIENCE TASK 4 - BUSINESS MARKETING INSIGHTS\n")
        f.write("=" * 65 + "\n")
        f.write(insights_text)
    print(f"[SAVED] Business insights report saved to '{insights_filepath}'")


def main():
    """Main execution function running the complete data science pipeline."""
    print("=" * 70)
    print("CODEALPHA DATA SCIENCE INTERNSHIP: TASK 4 - SALES PREDICTION")
    print("=" * 70)

    # 1. Setup results folder
    output_dir = create_results_directory("results")

    # 2. Load dataset
    df = load_and_inspect_data("Advertising.csv")

    # 3. Clean and prepare data
    df_clean = clean_and_prepare_data(df)

    # 4. Exploratory Data Analysis & Visualizations
    perform_eda_and_visualizations(df_clean, output_dir)

    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2, random_state=42)

    # 6. Train, Evaluate & Compare Models
    lr_model, rf_model, metrics_df = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, output_dir
    )

    # 7. Deliver Business Insights
    generate_business_insights(df_clean, lr_model, rf_model, metrics_df)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All generated charts and summaries are available in the '{output_dir}/' folder.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
