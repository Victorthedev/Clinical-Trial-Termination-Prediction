import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os

FEATURES = [
    "phase_clean",
    "log_enrollment",
    "is_industry",
    "duration_missing",
    "is_interventional"
]

def load_data(path="data/processed/trials_processed.csv"):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df["target"]
    return X, y

def train_models(X_train_scaled, X_train_raw, y_train):
    models = {
        "Logistic Regression": (LogisticRegression(random_state=42), X_train_scaled),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), X_train_raw),
        "XGBoost": (XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"), X_train_raw)
    }
    
    trained = {}
    for name, (model, X) in models.items():
        print(f"Training {name}...")
        model.fit(X, y_train)
        trained[name] = model
        
    return trained

def evaluate_models(trained_models, X_test_scaled, X_test_raw, y_test):
    test_data = {
        "Logistic Regression": X_test_scaled,
        "Random Forest": X_test_raw,
        "XGBoost": X_test_raw
    }
    
    results = {}
    for name, model in trained_models.items():
        X = test_data[name]
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc_roc": auc
        }
        
        print(f"\n{name}")
        print(f"  Accuracy:  {report['accuracy']:.4f}")
        print(f"  Precision: {report['1']['precision']:.4f}")
        print(f"  Recall:    {report['1']['recall']:.4f}")
        print(f"  F1:        {report['1']['f1-score']:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        
    return results

def save_best_model(trained_models, results):
    best_name = max(results, key=lambda x: results[x]["auc_roc"])
    best_model = trained_models[best_name]
    
    os.makedirs("models", exist_ok=True)
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "name": best_name,
            "features": FEATURES
        }, f)
        
    print(f"\nBest model: {best_name} (AUC-ROC: {results[best_name]['auc_roc']:.4f})")
    print("Saved to models/best_model.pkl")
    return best_name

def main():
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    os.makedirs("models", exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    trained_models = train_models(X_train_scaled, X_train, y_train)
    results = evaluate_models(trained_models, X_test_scaled, X_test, y_test)
    best = save_best_model(trained_models, results)
    
    return results, best

if __name__ == "__main__":
    main()