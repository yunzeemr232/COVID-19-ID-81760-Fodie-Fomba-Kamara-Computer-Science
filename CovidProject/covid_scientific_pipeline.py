"""
================================================================================
COVID-19 MULTI-DIMENSIONAL EPIDEMIOLOGICAL DATA SCIENCE & MACHINE LEARNING PIPELINE
Environment Requirement: Anaconda (Python 3.x with NumPy, Pandas, Scikit-Learn, Matplotlib)
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Set matplotlib style for high-resolution scannable rendering
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_epidemiological_pipeline():
    print("================================================================")
    # -------------------------------------------------------------------------
    # LAYER 1: DATA INGESTION & INTEGRITY CHECK
    # -------------------------------------------------------------------------
    print("[1/4] Ingesting all 10 epidemiological datasets...")
    
    # Load files safely
    case_df = pd.read_csv('Case.csv')
    patient_df = pd.read_csv('PatientInfo.csv')
    policy_df = pd.read_csv('Policy.csv')
    region_df = pd.read_csv('Region.csv')
    search_df = pd.read_csv('SearchTrend.csv')
    time_df = pd.read_csv('Time.csv')
    time_age_df = pd.read_csv('TimeAge.csv')
    time_gender_df = pd.read_csv('TimeGender.csv')
    time_prov_df = pd.read_csv('TimeProvince.csv')
    weather_df = pd.read_csv('Weather.csv')
    
    # Strip any leading spaces from column names (e.g., ' case_id' in Case.csv)
    case_df.columns = case_df.columns.str.strip()
    
    # Standardize all tracking time formats to datetime objects
    for df in [patient_df, policy_df, search_df, time_df, time_age_df, time_gender_df, time_prov_df, weather_df]:
        for col in ['date', 'confirmed_date', 'start_date', 'end_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

    # -------------------------------------------------------------------------
    # LAYER 2: FEATURE ENGINEERING & DATASETS CONNECTION
    # -------------------------------------------------------------------------
    print("[2/4] Linking files & constructing master analytical matrix...")
    
    # A. Aggregate Infrastructure Demographics from Region.csv at the Province Level
    # (Excluding the summary rows where city == province to get real aggregated means)
    prov_region = region_df[region_df['province'] != region_df['city']].groupby('province').agg({
        'elementary_school_count': 'sum',
        'kindergarten_count': 'sum',
        'university_count': 'sum',
        'academy_ratio': 'mean',
        'elderly_population_ratio': 'mean',
        'elderly_alone_ratio': 'mean',
        'nursing_home_count': 'sum'
    }).reset_index()
    
    # B. Derive Regional Structural Cluster Metrics from Case.csv
    prov_clusters = case_df.groupby('province').agg({
        'case_id': 'count',
        'confirmed': 'sum',
        'group': 'sum'  # Number of group-based institutional transmission vectors
    }).rename(columns={'case_id': 'total_cluster_count', 'confirmed': 'cluster_confirmed_total', 'group': 'group_transmission_count'}).reset_index()

    # C. Calculate Active Chronological Policy Count from Policy.csv for each date
    unique_dates = time_prov_df['date'].unique()
    policy_counts = []
    for d in unique_dates:
        # Count how many policies were actively implemented on this specific date
        active_count = ((policy_df['start_date'] <= d) & ((policy_df['end_date'].isna()) | (policy_df['end_date'] >= d))).sum()
        policy_counts.append({'date': d, 'active_policy_count': active_count})
    policy_time_df = pd.DataFrame(policy_counts)

    # D. Calculate Non-cumulative Daily New Cases from Cumulative Time Records
    time_prov_df = time_prov_df.sort_values(by=['province', 'date'])
    time_prov_df['daily_new_confirmed'] = time_prov_df.groupby('province')['confirmed'].diff().fillna(0).clip(lower=0)
    
    # E. Assemble Base Matrix (TimeProvince + Weather + SearchTrend + Policy Timeline)
    base_matrix = pd.merge(time_prov_df, weather_df, on=['province', 'date'], how='inner')
    base_matrix = pd.merge(base_matrix, search_df, on='date', how='inner')
    base_matrix = pd.merge(base_matrix, policy_time_df, on='date', how='inner')
    
    # F. Inject Infrastructure & Cluster Structural Profiles
    base_matrix = pd.merge(base_matrix, prov_region, on='province', how='inner')
    base_matrix = pd.merge(base_matrix, prov_clusters, on='province', how='inner')

    # Drop non-predictive metadata tracking IDs or redundant system labels
    features_matrix = base_matrix.drop(columns=['time', 'code', 'most_wind_direction'])
    # Interpolate minor localized null parameters in weather readings
    features_matrix = features_matrix.ffill().bfill()
    
    print(f" -> Master Matrix successfully compiled. Dimensions: {features_matrix.shape}")

    # -------------------------------------------------------------------------
    # LAYER 3: EXPLORATORY DATA SCIENCE VISUALIZATION (Matplotlib Backend)
    # -------------------------------------------------------------------------
    print("[3/4] Generating diagnostic visualization layers...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # Graph A: Nationwide Cumulative Pathogen Progress (Time.csv)
    axes[0, 0].plot(time_df['date'], time_df['confirmed'], color='crimson', lw=2.5, label='Confirmed Positives')
    axes[0, 0].plot(time_df['date'], time_df['released'], color='forestgreen', lw=2, label='Released / Recovered')
    axes[0, 0].fill_between(time_df['date'], time_df['confirmed'], time_df['released'], color='red', alpha=0.1)
    axes[0, 0].set_title("Nationwide Epidemic Tracking Matrix", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Patient Counts")
    axes[0, 0].legend()
    
    # Graph B: Age Cohort Mortality Breakdown Analytics (TimeAge.csv)
    latest_age = time_age_df[time_age_df['date'] == time_age_df['date'].max()]
    cfr_age = (latest_age['deceased'] / latest_age['confirmed'].replace(0, 1)) * 100
    axes[0, 1].bar(latest_age['age'], cfr_age, color='darkorange', edgecolor='black', alpha=0.8)
    axes[0, 1].set_title("Case Fatality Rate (%) by Age Demographics", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("Fatality Rate %")
    axes[0, 1].set_xlabel("Age Group")
    
    # Graph C: Search Trend Volume Index vs Spikes (SearchTrend.csv & Time.csv)
    axes[1, 0].plot(search_df['date'], search_df['coronavirus'], color='purple', label='Search Index: "coronavirus"')
    axes[1, 0].set_title("Public Worry Index vs Pathogen Expansion Window", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Relative Query Search Volume")
    axes[1, 0].set_xlim(pd.Timestamp('2020-01-01'), search_df['date'].max())
    axes[1, 0].legend(loc='upper left')
    
    # Graph D: Patient State Outcome Distribution Analysis (PatientInfo.csv)
    state_counts = patient_df['state'].value_counts()
    axes[1, 1].pie(state_counts, labels=state_counts.index, autopct='%1.1f%%', 
                   colors=['lightgreen', 'salmon', 'darkgray'], startangle=140, explode=[0, 0.1, 0.2][:len(state_counts)])
    axes[1, 1].set_title("Patient Status Outcome Distribution", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('epidemiological_analytics_report.png', dpi=300)
    print(" -> Scientific analytical dashboard plots exported cleanly as 'epidemiological_analytics_report.png'.")
    plt.show()

    # -------------------------------------------------------------------------
    # LAYER 4: SCIENTIFIC MACHINE LEARNING PIPELINE (Scikit-Learn)
    # -------------------------------------------------------------------------
    print("[4/4] Activating Scikit-Learn Predictive Machine Learning Engine...")
    
    # Define Target Objective: Forecast localized non-cumulative Daily New Cases
    y = features_matrix['daily_new_confirmed'].values
    
    # Drop targets, string dates, and non-numeric categorical province values
    X_raw = features_matrix.drop(columns=['date', 'province', 'confirmed', 'released', 'deceased', 'daily_new_confirmed'])
    
    # Convert remaining categories to numeric identifiers using One-Hot encoding safely
    X = pd.get_dummies(X_raw, drop_first=True)
    feature_names = X.columns.tolist()
    X = X.values
    
    # Implement cross-validation train/test partitioning split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Construct an Anaconda-optimized processing pipeline execution sequence
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1))
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Generate predictive vector output matrices
    y_pred = pipeline.predict(X_test)
    
    # Evaluate model accuracy metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n================================================================")
    print("                MACHINE LEARNING MODEL REPORT                   ")
    print("================================================================")
    print(f"Model Regressor Algorithm : Random Forest Ensemble Architecture")
    print(f"Mean Squared Error (MSE)  : {mse:.4f}")
    print(f"Root Mean Sq. Error(RMSE) : {rmse:.4f} cases")
    print(f"Coefficient of Det. ($R^2$)   : {r2:.4f}")
    print("================================================================")
    
    # Extract feature importance weights to identify the key driver of transmission spikes
    importances = pipeline.named_steps['regressor'].feature_importances_
    indices = np.argsort(importances)[::-1][:5]  # View top 5 features
    
    print("\nTOP 5 OUTBREAK DRIVERS DISCOVERED BY RANDOM FOREST:")
    for rank, idx in enumerate(indices):
        print(f" Rank {rank+1}: Feature '{feature_names[idx]}' -> Weight Importance: {importances[idx]:.4f}")
    print("================================================================")

if __name__ == "__main__":
    run_epidemiological_pipeline()