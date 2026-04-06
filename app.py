import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from openai import OpenAI
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

@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        obj = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return obj["model"], obj["name"], scaler

@st.cache_data
def load_background():
    return pd.read_csv("data/processed/trials_processed.csv")[FEATURES]

def get_openai_client():
    try:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
        return None

def preprocess_input(phase, enrollment, sponsor, duration_known, study_type):
    phase_map = {"Phase 1": 1, "Phase 2": 2, "Phase 3": 3, "Phase 4": 4, "Not Applicable": 0}
    row = {
        "phase_clean": phase_map[phase],
        "log_enrollment": np.log1p(min(enrollment, 5000)),
        "is_industry": 1 if sponsor == "Industry" else 0,
        "duration_missing": 0 if duration_known else 1,
        "is_interventional": 1 if study_type == "Interventional" else 0
    }
    return pd.DataFrame([row])

def plot_shap_waterfall(model, scaler, input_df):
    background = load_background()
    background_scaled = pd.DataFrame(
        scaler.transform(background),
        columns=FEATURES
    )
    input_scaled = pd.DataFrame(
        scaler.transform(input_df),
        columns=FEATURES
    )

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, background_scaled)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_scaled)

    fig, ax = plt.subplots(figsize=(8, 4))
    features = [FEATURE_LABELS[f] for f in FEATURES]
    values = shap_values[0]
    colors = ["#d32f2f" if v > 0 else "#1976d2" for v in values]

    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on termination risk)")
    ax.set_title("Why did the model predict this?")
    plt.tight_layout()
    return fig

def batch_predict(model, scaler, df):
    phase_map = {
        "PHASE1": 1, "PHASE2": 2, "PHASE3": 3,
        "PHASE4": 4, "NA": 0, "EARLY_PHASE1": 0
    }
    df["phase_clean"] = df["phase"].map(phase_map).fillna(0)
    df["log_enrollment"] = np.log1p(
        pd.to_numeric(df["enrollment"], errors="coerce").clip(upper=5000).fillna(0)
    )
    df["is_industry"] = (df["sponsor_class"] == "INDUSTRY").astype(int)
    df["duration_missing"] = df["completion_date"].isnull().astype(int)
    df["is_interventional"] = (df["study_type"] == "INTERVENTIONAL").astype(int)

    X = df[FEATURES]
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    df["termination_risk"] = probs
    df["risk_label"] = pd.cut(
        probs,
        bins=[0, 0.4, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    return df

def generate_explanation(model, scaler, input_df, prob):
    client = get_openai_client()

    background = load_background()
    background_scaled = pd.DataFrame(scaler.transform(background), columns=FEATURES)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURES)

    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, background_scaled)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_scaled)[0]

    shap_summary = "\n".join([
        f"- {FEATURE_LABELS[f]}: SHAP value = {v:.3f} ({'increases' if v > 0 else 'decreases'} termination risk)"
        for f, v in zip(FEATURES, shap_values)
    ])

    direction = "high" if prob >= 0.6 else "medium" if prob >= 0.4 else "low"

    input_context = f"""Trial details entered:
- Trial Phase: {int(input_df['phase_clean'].iloc[0])}
- Enrollment Size: {int(np.expm1(input_df['log_enrollment'].iloc[0])):,}
- Sponsor Type: {'Industry' if input_df['is_industry'].iloc[0] else 'Academic / Other'}
- Completion Date Known: {'No' if input_df['duration_missing'].iloc[0] else 'Yes'}
- Study Type: {'Interventional' if input_df['is_interventional'].iloc[0] else 'Observational'}"""

    prompt = f"""A machine learning model predicted that a clinical trial has a {prob:.1%} probability of early termination, classified as {direction} risk.

{input_context}

The model's SHAP values show which features drove this prediction:
{shap_summary}

Write a clear 2-3 sentence explanation for a non-technical user. Use the actual trial details above, not assumptions. Reference only what is true about this specific trial. Do not mention SHAP or model."""

    if client is None:
        return f"This trial is assessed as **{direction} risk** ({prob:.1%})."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    return response.choices[0].message.content

# --- App Layout ---
st.set_page_config(page_title="Clinical Trial Termination Predictor", layout="wide")

st.title("Clinical Trial Termination Risk Predictor")
st.markdown("Predict the likelihood of a clinical trial being terminated early using machine learning.")

model, model_name, scaler = load_model()

tab1, tab2 = st.tabs(["Single Trial Prediction", "Batch Upload"])

# --- Tab 1: Single Prediction ---
with tab1:
    st.subheader("Enter Trial Details")

    col1, col2 = st.columns(2)

    with col1:
        phase = st.selectbox("Trial Phase", ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Not Applicable"])
        enrollment = st.number_input("Expected Enrollment Size", min_value=1, max_value=100000, value=100)
        sponsor = st.selectbox("Sponsor Type", ["Industry", "Academic / Other"])

    with col2:
        study_type = st.selectbox("Study Type", ["Interventional", "Observational"])
        duration_known = st.radio("Is completion date known?", ["Yes", "No"]) == "Yes"

    if st.button("Predict Termination Risk"):
        input_df = preprocess_input(phase, enrollment, sponsor, duration_known, study_type)
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURES)
        prob = model.predict_proba(input_scaled)[0][1]

        st.divider()

        if prob >= 0.6:
            st.error(f"High Termination Risk: {prob:.1%}")
        elif prob >= 0.4:
            st.warning(f"Medium Termination Risk: {prob:.1%}")
        else:
            st.success(f"Low Termination Risk: {prob:.1%}")

        st.markdown(generate_explanation(model, scaler, input_df, prob))

        st.subheader("Why this prediction?")
        fig = plot_shap_waterfall(model, scaler, input_df)
        st.pyplot(fig)

        st.caption("Red bars increase termination risk. Blue bars decrease it.")

# --- Tab 2: Batch Upload ---
with tab2:
    st.subheader("Upload a CSV of Trials")
    st.markdown("CSV must include columns: `phase`, `enrollment`, `sponsor_class`, `completion_date`, `study_type`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df)} trials")

        try:
            results = batch_predict(model, scaler, df.copy())

            st.subheader("Results")
            display_cols = ["nct_id", "termination_risk", "risk_label"] if "nct_id" in results.columns else ["termination_risk", "risk_label"]
            st.dataframe(
                results[display_cols].sort_values("termination_risk", ascending=False),
                use_container_width=True
            )

            st.subheader("Risk Distribution")
            fig2, ax = plt.subplots()
            risk_counts = results["risk_label"].value_counts()
            risk_counts = risk_counts.reindex(["High", "Medium", "Low"]).fillna(0)
            risk_counts.plot(kind="bar", ax=ax, color=["#d32f2f", "#f57c00", "#1976d2"])
            ax.set_xlabel("Risk Level")
            ax.set_ylabel("Number of Trials")
            ax.set_title("Termination Risk Distribution")
            plt.tight_layout()
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error processing file: {e}")

st.divider()
st.caption(f"Model: {model_name} | AUC-ROC: 0.749 | Data source: ClinicalTrials.gov")