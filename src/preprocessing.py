import pandas as pd
import numpy as np
import os

def load_raw_data(path="data/raw/trials_raw.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nStatus distribution:\n{df['status'].value_counts()}")
    return df

def create_target(df):
    df["target"] = (df["status"] == "TERMINATED").astype(int)
    return df

def clean_phase(df):
    phase_map = {
        "PHASE1": 1,
        "PHASE2": 2,
        "PHASE3": 3,
        "PHASE4": 4,
        "NA": 0,
        "EARLY_PHASE1": 0
    }
    df["phase_clean"] = df["phase"].map(phase_map)
    df["phase_clean"] = df["phase_clean"].fillna(0)
    return df

def clean_enrollment(df):
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    median_enrollment = df["enrollment"].median()
    df["enrollment_clean"] = df["enrollment"].fillna(median_enrollment)
    df["log_enrollment"] = np.log1p(df["enrollment_clean"])
    return df

def clean_sponsor(df):
    df["is_industry"] = (df["sponsor_class"] == "INDUSTRY").astype(int)
    df["is_industry"] = df["is_industry"].fillna(0)
    return df

def calculate_duration(df):
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    df["planned_duration_days"] = (
        df["completion_date"] - df["start_date"]
    ).dt.days
    
    df["duration_missing"] = df["planned_duration_days"].isnull().astype(int)
    
    median_duration = df["planned_duration_days"].median()
    df["planned_duration_days"] = df["planned_duration_days"].fillna(median_duration)
    
    cap = df["planned_duration_days"].quantile(0.99)
    df["planned_duration_days"] = df["planned_duration_days"].clip(upper=cap)
    
    return df

def clean_enrollment(df):
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    median_enrollment = df["enrollment"].median()
    df["enrollment_clean"] = df["enrollment"].fillna(median_enrollment)
    
    cap = df["enrollment_clean"].quantile(0.99)
    df["enrollment_clean"] = df["enrollment_clean"].clip(upper=cap)
    df["log_enrollment"] = np.log1p(df["enrollment_clean"])
    
    return df

def clean_study_type(df):
    df["is_interventional"] = (df["study_type"] == "INTERVENTIONAL").astype(int)
    return df

def get_final_features(df):
    features = [
        "phase_clean",
        "log_enrollment",
        "is_industry",
        "duration_missing",
        "is_interventional"
    ]
    target = "target"
    
    df_final = df[features + [target, "nct_id"]].copy()
    df_final = df_final.dropna()
    
    print(f"\nFinal dataset shape: {df_final.shape}")
    print(f"Target distribution:\n{df_final['target'].value_counts()}")
    
    return df_final, features

def main():
    os.makedirs("data/processed", exist_ok=True)
    
    df = load_raw_data()
    df = create_target(df)
    df = clean_phase(df)
    df = clean_enrollment(df)
    df = clean_sponsor(df)
    df = calculate_duration(df)
    df = clean_study_type(df)
    
    df_final, features = get_final_features(df)
    
    output_path = "data/processed/trials_processed.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\nSaved processed data to {output_path}")
    
    return df_final, features

if __name__ == "__main__":
    main()