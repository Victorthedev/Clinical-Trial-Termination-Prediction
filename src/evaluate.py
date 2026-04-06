import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report, roc_auc_score
)
import shap
import pickle
import os

FEATURES = [
    "phase_clean",
    "log_enrollment",
    "is_industry",
    "duration_missing",
    "is_interventional"
]

FEATURE_LABELS = {
    "phase_clean": "Trial Phase",
    "log_enrollment": "Log Enrollment Size",
    "is_industry": "Industry Sponsor",
    "duration_missing": "Duration Not Reported",
    "is_interventional": "Interventional Study"
}

def load_data(path="data/processed/trials_processed.csv"):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df["target"]
    return X, y

def plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Completed", "Terminated"],
        yticklabels=["Completed", "Terminated"]
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    path = f"{output_dir}/confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_roc_curves(models_data, y_test, output_dir):
    plt.figure(figsize=(8, 6))
    
    for name, y_prob in models_data.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")
    
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.tight_layout()
    path = f"{output_dir}/roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_shap(model, X_test, output_dir):
    explainer = shap.LinearExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)
    
    X_display = X_test.copy()
    X_display.columns = [FEATURE_LABELS[f] for f in FEATURES]
    
    plt.figure()
    shap.summary_plot(
        shap_values, X_display,
        plot_type="bar",
        show=False
    )
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    path = f"{output_dir}/shap_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_model_comparison(results, output_dir):
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    df = pd.DataFrame(results).T[metrics]
    
    df.plot(kind="bar", figsize=(10, 6), ylim=(0, 1))
    plt.title("Model Comparison — All Metrics")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = f"{output_dir}/model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def main():
    os.makedirs("outputs", exist_ok=True)
    
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURES
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURES
    )
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    }
    
    train_data = {
        "Logistic Regression": X_train_scaled,
        "Random Forest": X_train,
        "XGBoost": X_train
    }
    
    test_data = {
        "Logistic Regression": X_test_scaled,
        "Random Forest": X_test,
        "XGBoost": X_test
    }
    
    trained = {}
    results = {}
    probs = {}
    
    for name, model in models.items():
        model.fit(train_data[name], y_train)
        trained[name] = model
        
        y_pred = model.predict(test_data[name])
        y_prob = model.predict_proba(test_data[name])[:, 1]
        probs[name] = y_prob
        
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            "accuracy": report["accuracy"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc_roc": roc_auc_score(y_test, y_prob)
        }
        
        plot_confusion_matrix(y_test, y_pred, name, "outputs")
    
    plot_roc_curves(probs, y_test, "outputs")
    plot_model_comparison(results, "outputs")
    plot_shap(trained["Logistic Regression"], X_test_scaled, "outputs")
    
    print("\nFinal Results Table:")
    print(pd.DataFrame(results).T.round(4))

if __name__ == "__main__":
    main()