# Clinical Trial Termination Prediction

A machine learning system that predicts whether a clinical trial will be terminated early, built on data from ClinicalTrials.gov.

**Live app:** [https://clinical-trial-termination-prediction.streamlit.app](https://clinical-trial-termination-prediction.streamlit.app)

---

## What it does

Clinical trials fail early more often than most people realise. When they do, funding is lost, patients lose access to potential treatments and research teams lose years of work. This system uses machine learning to flag trials at risk of early termination using only information available at the point of registration.

The app has three modes:

- **Single Trial Prediction** — enter trial details manually and get a risk score, a SHAP chart showing which features drove the prediction and a plain English explanation of why
- **Batch Upload** — upload a CSV of trials and get every trial scored with a portfolio-level briefing
- **Live API Search** — search ClinicalTrials.gov by condition or keyword and score up to 100 active trials in real time

---

## How it works

Data was pulled from the ClinicalTrials.gov API v2. 5,000 completed and 5,000 terminated trials were used for training. Features include trial phase, enrollment size, sponsor type, study type and whether a completion date was registered.

Three models were compared: Logistic Regression, Random Forest and XGBoost. Logistic Regression performed best with an AUC-ROC of 0.749 and was selected as the final model.

Predictions are explained using SHAP values which are also passed to GPT-4o-mini to generate plain English explanations grounded in clinical trial context.

---

## Project structure

```
Clinical-Trial-Termination-Prediction/
│
├── data/
│   ├── raw/                  # Raw API data (gitignored)
│   └── processed/            # Cleaned feature matrix
│
├── src/
│   ├── ingestion.py          # Pulls data from ClinicalTrials.gov API
│   ├── preprocessing.py      # Cleaning, feature engineering, target encoding
│   ├── model.py              # Trains and saves all three models
│   └── evaluate.py           # Generates confusion matrices, ROC curves, SHAP plots
│
├── models/
│   ├── best_model.pkl        # Trained Logistic Regression model
│   └── scaler.pkl            # Fitted StandardScaler
│
├── outputs/                  # Evaluation charts and visuals
├── app.py                    # Streamlit application
├── sample_batch.csv          # Sample CSV for testing batch upload
├── requirements.txt
└── README.md
```

---

## Setup

**Clone the repo**
```bash
git clone https://github.com/Victorthedev/Clinical-Trial-Termination-Prediction.git
cd Clinical-Trial-Termination-Prediction
```

**Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Add your OpenAI API key**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-key-here
```

> The natural language explanations require an OpenAI API key. The app still works without one but explanations will fall back to a basic risk label.

---

## Reproducing the results

Run the pipeline in order:

```bash
# 1. Fetch data from ClinicalTrials.gov
python src/ingestion.py

# 2. Clean and engineer features
python src/preprocessing.py

# 3. Train and compare models
python src/model.py

# 4. Generate evaluation visuals
python src/evaluate.py

# 5. Run the app
streamlit run app.py
```

---

## Batch upload format

The batch upload tab accepts a CSV with the following columns:

| Column | Description |
|---|---|
| `nct_id` | Trial identifier (optional but recommended) |
| `phase` | Trial phase (PHASE1, PHASE2, PHASE3, PHASE4, NA) |
| `enrollment` | Expected number of participants |
| `sponsor_class` | INDUSTRY or ACADEMIC |
| `completion_date` | Expected completion date (YYYY-MM-DD) |
| `study_type` | INTERVENTIONAL or OBSERVATIONAL |

Missing or malformed columns are handled automatically. A sample file is included at `sample_batch.csv`.

---

## Model performance

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.686 | 0.680 | 0.702 | 0.691 | **0.749** |
| Random Forest | 0.636 | 0.638 | 0.629 | 0.633 | 0.698 |
| XGBoost | 0.674 | 0.696 | 0.618 | 0.655 | 0.744 |

---

## Stack

- Python 3.13
- scikit-learn, XGBoost, SHAP
- Streamlit
- OpenAI GPT-4o-mini
- ClinicalTrials.gov API v2

---

## Data source

All data is sourced from [ClinicalTrials.gov](https://clinicaltrials.gov), a public registry maintained by the US National Library of Medicine. No proprietary or patient-level data is used.