import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import time

load_dotenv()

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

PHASE_MAP = {
    "PHASE1": 1, "PHASE2": 2, "PHASE3": 3,
    "PHASE4": 4, "NA": 0, "EARLY_PHASE1": 0,
    "phase1": 1, "phase2": 2, "phase3": 3,
    "phase4": 4, "na": 0, "early_phase1": 0,
    "1": 1, "2": 2, "3": 3, "4": 4,
    "i": 1, "ii": 2, "iii": 3, "iv": 4
}

PHASE_LABELS = {0: "N/A", 1: "Phase 1", 2: "Phase 2", 3: "Phase 3", 4: "Phase 4"}

PHASE_COLORS = {
    "Phase 1": "#7B68EE",
    "Phase 2": "#4169E1",
    "Phase 3": "#1E90FF",
    "Phase 4": "#00BFFF",
    "N/A": "#888888"
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

def clean_batch_df(df):
    df.columns = df.columns.str.strip().str.lower()
    issues = []

    if "phase" in df.columns:
        df["phase_clean"] = df["phase"].astype(str).str.strip().str.upper().map(PHASE_MAP).fillna(0)
    else:
        df["phase_clean"] = 0
        issues.append("Column 'phase' not found, defaulting to 0")

    if "enrollment" in df.columns:
        df["log_enrollment"] = np.log1p(
            pd.to_numeric(df["enrollment"], errors="coerce").clip(upper=5000).fillna(0)
        )
    else:
        df["log_enrollment"] = 0
        issues.append("Column 'enrollment' not found, defaulting to 0")

    if "sponsor_class" in df.columns:
        df["is_industry"] = df["sponsor_class"].astype(str).str.strip().str.upper().eq("INDUSTRY").astype(int)
    else:
        df["is_industry"] = 0
        issues.append("Column 'sponsor_class' not found, defaulting to 0")

    if "completion_date" in df.columns:
        df["duration_missing"] = pd.to_datetime(df["completion_date"], errors="coerce").isnull().astype(int)
    else:
        df["duration_missing"] = 1
        issues.append("Column 'completion_date' not found, assuming missing")

    if "study_type" in df.columns:
        df["is_interventional"] = df["study_type"].astype(str).str.strip().str.upper().eq("INTERVENTIONAL").astype(int)
    else:
        df["is_interventional"] = 0
        issues.append("Column 'study_type' not found, defaulting to 0")

    return df, issues

def batch_predict(model, scaler, df):
    df, issues = clean_batch_df(df.copy())
    X = df[FEATURES]
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    df["termination_risk"] = probs
    df["termination_risk_pct"] = (probs * 100).round(1).astype(str) + "%"
    df["risk_label"] = pd.cut(
        probs,
        bins=[0, 0.4, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    return df, issues

def plot_shap_waterfall(model, scaler, input_df):
    background = load_background()
    background_scaled = pd.DataFrame(scaler.transform(background), columns=FEATURES)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=FEATURES)

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

def generate_single_explanation(model, scaler, input_df, prob):
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

    input_context = f"""Trial details:
- Trial Phase: {int(input_df['phase_clean'].iloc[0])}
- Enrollment Size: {int(np.expm1(input_df['log_enrollment'].iloc[0])):,}
- Sponsor Type: {'Industry' if input_df['is_industry'].iloc[0] else 'Academic / Other'}
- Completion Date Known: {'No' if input_df['duration_missing'].iloc[0] else 'Yes'}
- Study Type: {'Interventional' if input_df['is_interventional'].iloc[0] else 'Observational'}"""

    prompt = f"""A machine learning model predicted that a clinical trial has a {prob:.1%} probability of early termination, classified as {direction} risk.

{input_context}

The model's SHAP values show which features drove this prediction:
{shap_summary}

Write a 2-3 sentence explanation for a non-technical user. For the most important factors only, explain not just what they do to the risk but WHY they have that effect in the context of clinical trials. Use the actual trial details above. Do not mention SHAP or model. Be specific and educational. Keep it concise."""

    if client is None:
        return f"This trial is assessed as **{direction} risk** ({prob:.1%})."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    return response.choices[0].message.content

def generate_trial_one_liner(row):
    client = get_openai_client()
    if client is None:
        return ""

    phase = PHASE_LABELS.get(int(row.get("phase_clean", 0)), "Unknown phase")
    sponsor = "industry sponsored" if row.get("is_industry", 0) == 1 else "non-industry sponsored"
    duration = "no completion date registered" if row.get("duration_missing", 0) == 1 else "has a registered completion date"
    enrollment = int(np.expm1(row.get("log_enrollment", 0)))
    risk = row["termination_risk"]
    study_type = "interventional" if row.get("is_interventional", 0) == 1 else "observational"
    condition = str(row.get("condition", "")) if pd.notna(row.get("condition", "")) else ""

    prompt = f"""A clinical trial {'studying ' + condition if condition else ''} is {phase}, {sponsor}, {study_type}, targets {enrollment:,} participants and {duration}. It has a predicted {risk:.1%} termination risk.

Write one plain English sentence explaining the most important reason for this risk level. Do not start with 'This trial'. Do not mention models or algorithms. Be specific."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60
    )

    return response.choices[0].message.content

def generate_batch_narrative(results_df, context=""):
    client = get_openai_client()
    if client is None:
        return None

    high = int((results_df["risk_label"] == "High").sum())
    medium = int((results_df["risk_label"] == "Medium").sum())
    low = int((results_df["risk_label"] == "Low").sum())
    total = len(results_df)
    avg_risk = results_df["termination_risk"].mean()

    top3 = results_df.sort_values("termination_risk", ascending=False).head(3)
    top_details = []
    for _, row in top3.iterrows():
        nct = str(row.get("nct_id", "Unknown"))
        risk = row["termination_risk"]
        phase = PHASE_LABELS.get(int(row.get("phase_clean", 0)), "Unknown")
        sponsor = "industry sponsored" if row.get("is_industry", 0) == 1 else "non-industry sponsored"
        duration = "no completion date" if row.get("duration_missing", 0) == 1 else "has completion date"
        enrollment = int(np.expm1(row.get("log_enrollment", 0)))
        condition = str(row.get("condition", "")) if "condition" in row and pd.notna(row.get("condition", "")) else ""
        top_details.append(
            f"{nct}{' (' + condition + ')' if condition else ''}: {risk:.1%} risk, {phase}, {sponsor}, {enrollment:,} participants, {duration}"
        )

    top_str = "\n".join(top_details)

    context_line = f"Search term: {context}\n" if context else ""

    prompt = f"""{context_line}A portfolio of {total} clinical trials was analysed for termination risk.

Summary: {high} high risk, {medium} medium risk, {low} low risk. Average termination probability: {avg_risk:.1%}.

Highest risk trials:
{top_str}

Write a 4-5 sentence plain English briefing. Name the specific trial IDs at highest risk and explain what features make them risky. If there are patterns across the risky trials point them out. Focus on the trials not the reviewer. No em dashes. No we or our. No model or algorithm references. Write like a senior analyst briefing a medical director."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )

    return response.choices[0].message.content

def format_date(val):
    if pd.isna(val) or str(val).strip() in ["", "nan", "None"]:
        return "Not registered"
    try:
        return pd.to_datetime(val).strftime("%b %Y")
    except Exception:
        return str(val)

def risk_badge(label, risk):
    colors = {"High": "#d32f2f", "Medium": "#f57c00", "Low": "#2e7d32"}
    color = colors.get(str(label), "#888")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:bold;">{label} Risk &nbsp; {risk:.1%}</span>'

def phase_badge(phase_label):
    color = PHASE_COLORS.get(phase_label, "#888")
    return f'<span style="background:{color};color:white;padding:3px 10px;border-radius:12px;font-size:0.78rem;">{phase_label}</span>'

def render_trial_cards(results_df, show_one_liners=False):
    results_sorted = results_df.sort_values("termination_risk", ascending=False)

    for _, row in results_sorted.iterrows():
        nct = str(row.get("nct_id", "Unknown"))
        condition = str(row.get("condition", "")) if "condition" in row and pd.notna(row.get("condition", "")) else "Condition not specified"
        phase_label = PHASE_LABELS.get(int(row.get("phase_clean", 0)), "N/A")
        sponsor = "Industry" if row.get("is_industry", 0) == 1 else "Academic / Other"
        study_type = "Interventional" if row.get("is_interventional", 0) == 1 else "Observational"
        enrollment = int(np.expm1(row.get("log_enrollment", 0)))
        duration_missing = row.get("duration_missing", 1)
        risk = row["termination_risk"]
        label = str(row["risk_label"])

        start = format_date(row.get("start_date", ""))
        end = format_date(row.get("completion_date", "")) if "completion_date" in row else ("Not registered" if duration_missing else "")

        border_color = "#d32f2f" if label == "High" else "#f57c00" if label == "Medium" else "#2e7d32"
        nct_url = f"https://clinicaltrials.gov/study/{nct}"

        timeline_html = f"<span style='color:#aaa;font-size:0.85rem;'>Started: {start} &nbsp;|&nbsp; Expected completion: {end}</span>"
        if end == "Not registered":
            timeline_html = f"<span style='color:#aaa;font-size:0.85rem;'>Started: {start} &nbsp;|&nbsp; </span><span style='color:#f57c00;font-size:0.85rem;'>No completion date registered</span>"

        card = f"""
        <div style="border-left: 4px solid {border_color}; padding: 16px 20px; margin-bottom: 16px; border-radius: 6px; background: #1a1a1a;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
                <div>
                    <a href="{nct_url}" target="_blank" style="font-size:1rem; font-weight:bold; color:#58a6ff; text-decoration:none;">{nct}</a>
                    <span style="color:#ccc; font-size:0.95rem; margin-left:10px;">{condition}</span>
                </div>
                <div>{risk_badge(label, risk)}</div>
            </div>
            <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:10px; align-items:center;">
                {phase_badge(phase_label)}
                <span style="color:#ccc; font-size:0.85rem;">Sponsor: <strong>{sponsor}</strong></span>
                <span style="color:#ccc; font-size:0.85rem;">Type: <strong>{study_type}</strong></span>
                <span style="color:#ccc; font-size:0.85rem;">Target enrollment: <strong>{enrollment:,} participants</strong></span>
            </div>
            <div style="margin-top:8px;">{timeline_html}</div>
        </div>
        """
        st.markdown(card, unsafe_allow_html=True)

        if show_one_liners:
            with st.spinner(""):
                one_liner = generate_trial_one_liner(row)
            if one_liner:
                st.markdown(
                    f"<div style='margin:-10px 0 16px 20px; color:#aaa; font-size:0.88rem; font-style:italic;'>{one_liner}</div>",
                    unsafe_allow_html=True
                )

def render_batch_results(results, issues, context_label="", show_one_liners=False):
    if issues:
        with st.expander("Data quality notes"):
            for issue in issues:
                st.warning(issue)

    high = int((results["risk_label"] == "High").sum())
    medium = int((results["risk_label"] == "Medium").sum())
    low = int((results["risk_label"] == "Low").sum())
    avg = results["termination_risk"].mean()
    total = len(results)

    st.markdown(
        f"""
        <div style="display:flex; gap:24px; flex-wrap:wrap; margin-bottom:16px;">
            <div style="background:#1a1a1a; border-radius:8px; padding:16px 24px; min-width:120px; text-align:center;">
                <div style="font-size:2rem; font-weight:bold; color:#d32f2f;">{high}</div>
                <div style="color:#aaa; font-size:0.85rem;">High Risk</div>
            </div>
            <div style="background:#1a1a1a; border-radius:8px; padding:16px 24px; min-width:120px; text-align:center;">
                <div style="font-size:2rem; font-weight:bold; color:#f57c00;">{medium}</div>
                <div style="color:#aaa; font-size:0.85rem;">Medium Risk</div>
            </div>
            <div style="background:#1a1a1a; border-radius:8px; padding:16px 24px; min-width:120px; text-align:center;">
                <div style="font-size:2rem; font-weight:bold; color:#2e7d32;">{low}</div>
                <div style="color:#aaa; font-size:0.85rem;">Low Risk</div>
            </div>
            <div style="background:#1a1a1a; border-radius:8px; padding:16px 24px; min-width:120px; text-align:center;">
                <div style="font-size:2rem; font-weight:bold; color:#58a6ff;">{avg:.1%}</div>
                <div style="color:#aaa; font-size:0.85rem;">Avg Risk</div>
            </div>
            <div style="background:#1a1a1a; border-radius:8px; padding:16px 24px; min-width:120px; text-align:center;">
                <div style="font-size:2rem; font-weight:bold; color:#ccc;">{total}</div>
                <div style="color:#aaa; font-size:0.85rem;">Total Trials</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Generating briefing..."):
        narrative = generate_batch_narrative(results, context=context_label)
    if narrative:
        st.markdown(
            f"""
            <div style="background:#1a1a1a; border-left:4px solid #58a6ff; padding:16px 20px; border-radius:6px; margin-bottom:20px; color:#ccc; line-height:1.7;">
            {narrative}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()
    st.subheader(f"All {total} Trials")
    render_trial_cards(results, show_one_liners=show_one_liners)

def fetch_live_trials(query, max_records=100):
    BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
    trials = []
    next_page_token = None

    while len(trials) < max_records:
        params = {
            "query.cond": query,
            "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,NOT_YET_RECRUITING",
            "pageSize": min(100, max_records - len(trials)),
            "fields": "NCTId,OverallStatus,Phase,EnrollmentCount,LeadSponsorClass,StartDate,CompletionDate,StudyType,Condition"
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(BASE_URL, params=params)
        if response.status_code != 200:
            break

        data = response.json()
        studies = data.get("studies", [])
        if not studies:
            break

        for study in studies:
            proto = study.get("protocolSection", {})
            id_module = proto.get("identificationModule", {})
            status_module = proto.get("statusModule", {})
            design_module = proto.get("designModule", {})
            sponsor_module = proto.get("sponsorCollaboratorsModule", {})
            conditions_module = proto.get("conditionsModule", {})

            trials.append({
                "nct_id": id_module.get("nctId"),
                "status": status_module.get("overallStatus"),
                "phase": design_module.get("phases", [None])[0],
                "enrollment": design_module.get("enrollmentInfo", {}).get("count"),
                "sponsor_class": sponsor_module.get("leadSponsor", {}).get("class"),
                "start_date": status_module.get("startDateStruct", {}).get("date"),
                "completion_date": status_module.get("completionDateStruct", {}).get("date"),
                "study_type": design_module.get("studyType"),
                "condition": conditions_module.get("conditions", [None])[0]
            })

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.3)

    return pd.DataFrame(trials)

# --- App Layout ---
st.set_page_config(page_title="Clinical Trial Termination Predictor", layout="wide")

st.title("Clinical Trial Termination Risk Predictor")
st.markdown("Predict the likelihood of a clinical trial being terminated early using machine learning.")

model, model_name, scaler = load_model()

tab1, tab2, tab3 = st.tabs(["Single Trial Prediction", "Batch Upload", "Live API Search"])

# --- Tab 1 ---
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

        st.markdown(generate_single_explanation(model, scaler, input_df, prob))

        st.subheader("Why this prediction?")
        fig = plot_shap_waterfall(model, scaler, input_df)
        st.pyplot(fig)
        st.caption("Red bars increase termination risk. Blue bars decrease it.")

# --- Tab 2 ---
with tab2:
    st.subheader("Upload a CSV of Trials")
    st.markdown("CSV should include columns: `phase`, `enrollment`, `sponsor_class`, `completion_date`, `study_type`. Extra or missing columns are handled automatically.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df)} trials")
        try:
            results, issues = batch_predict(model, scaler, df.copy())
            render_batch_results(results, issues)
        except Exception as e:
            st.error(f"Error processing file: {e}")

# --- Tab 3 ---
with tab3:
    st.subheader("Search Live Trials from ClinicalTrials.gov")
    st.markdown("Search for active trials by condition or keyword. Up to 100 trials will be fetched and scored.")

    query = st.text_input("Search condition or keyword", placeholder="e.g. diabetes, breast cancer, COVID-19")

    if st.button("Search and Predict"):
        if not query.strip():
            st.warning("Please enter a search term.")
        else:
            with st.spinner(f"Fetching trials for '{query}'..."):
                live_df = fetch_live_trials(query, max_records=100)

            if live_df.empty:
                st.error("No trials found for that search term.")
            else:
                st.write(f"Found {len(live_df)} trials. Scoring...")
                try:
                    results, issues = batch_predict(model, scaler, live_df.copy())
                    render_batch_results(results, issues, context_label=query, show_one_liners=True)
                except Exception as e:
                    st.error(f"Error scoring trials: {e}")

st.divider()
st.caption(f"Model: {model_name} | AUC-ROC: 0.749 | Data source: ClinicalTrials.gov")