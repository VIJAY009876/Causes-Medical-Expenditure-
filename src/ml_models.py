"""
ml_models.py
============
Train XGBoost classifiers for three targets:
  - CHE (Catastrophic Health Expenditure)
  - under_poverty_since_OOPE (Poverty Line)
  - distress (Distress Financing)

Trained on both hospitalization and non-hospitalization datasets.
Models saved to models/ with joblib.

Fixes from Colab:
- Imports moved to top (were after use)
- joblib.dump added so models are reusable in Streamlit
- Label mapping applied before training
- StratifiedKFold CV results printed per model
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier

from label_maps import (
    HH_SIZE_MAP, HH_EDUCATION_MAP, RELIGION_MAP, SOCIAL_GROUP_MAP,
    AGE_60_MAP, GENDER_MAP, ECONOMIC_QUINTILE_MAP, SECTOR_MAP,
    INCOME_SOURCE_MAP, NSS_STATE_MAP,
    ALL_FEATURES, CATEGORICAL_FEATURES, TARGET_LABELS
)

MODELS_DIR = "models"
DATA_DIR   = "data/processed_data"

TARGETS = list(TARGET_LABELS.keys())   # ["CHE", "under_poverty_since_OOPE", "distress"]


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply human-readable labels to categorical columns."""
    df = df.copy()
    df["HH_size"]                 = df["HH_size"].map(HH_SIZE_MAP)
    df["HH_education"]            = df["HH_education"].map(HH_EDUCATION_MAP)
    df["Religion"]                = df["Religion"].map(RELIGION_MAP)
    df["social_group"]            = df["social_group"].map(SOCIAL_GROUP_MAP)
    df["age_of_household_base60"] = df["age_of_household_base60"].map(AGE_60_MAP)
    df["Gender"]                  = df["Gender"].map(GENDER_MAP)
    df["economic_quintile"]       = df["economic_quintile"].map(ECONOMIC_QUINTILE_MAP)
    df["sector"]                  = df["sector"].map(SECTOR_MAP)
    df["Mejor_source_of_income"]  = df["Mejor_source_of_income"].map(INCOME_SOURCE_MAP)
    df["state"]                   = df["state"].map(NSS_STATE_MAP)
    return df


def build_pipeline() -> Pipeline:
    """XGBoost pipeline with OneHot encoding for categoricals."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"),
             CATEGORICAL_FEATURES)
        ],
        remainder="passthrough"
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])


def train_and_evaluate(df: pd.DataFrame, target: str, label: str):  
    """Train, evaluate, and return (pipeline, metrics_dict)."""
    df = df.copy().fillna(0)
    df = apply_labels(df)

    # Keep only features that exist in the dataframe
    available = [f for f in ALL_FEATURES if f in df.columns]
    X = df[available]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred)  * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred)    * 100, 2),
        "f1":        round(f1_score(y_test, y_pred)        * 100, 2),
        "roc_auc":   round(roc_auc_score(y_test, y_prob)   * 100, 2),
    }

    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    metrics["cv_roc_auc_mean"] = round(cv_scores.mean() * 100, 2)
    metrics["cv_roc_auc_std"]  = round(cv_scores.std()  * 100, 2)

    print(f"\n{'='*50}")
    print(f"  {label} → {target}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")

    return pipe, metrics


def train_all():
    """Train all 6 models (2 datasets × 3 targets) and save to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    hosp     = pd.read_csv(os.path.join(DATA_DIR, "hospital_model.csv"))
    non_hosp = pd.read_csv(os.path.join(DATA_DIR, "non_hospital_model.csv"))

    datasets = {
        "hosp":     hosp,
        "non_hosp": non_hosp
    }
    dataset_labels = {
        "hosp":     "Hospitalization",
        "non_hosp": "Non-Hospitalization"
    }

    all_metrics = {}

    for ds_key, df in datasets.items():
        for target in TARGETS:
            if target not in df.columns:
                print(f"  Skipping {target} — not in {ds_key}")
                continue

            model_name = f"xgb_{target}_{ds_key}"
            pipe, metrics = train_and_evaluate(df, target, dataset_labels[ds_key])

            # Save model
            path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
            joblib.dump(pipe, path)
            print(f"  Saved → {path}")

            all_metrics[model_name] = metrics

            model = pipe.named_steps["classifier"]

            importance = pd.DataFrame({
                "feature": pipe.named_steps["preprocessor"].get_feature_names_out(),
                "importance": model.feature_importances_
            })

            importance = importance.sort_values(by="importance",ascending=False)
            importance.to_csv(
                os.path.join(MODELS_DIR,f"feature_importance_{model_name}.csv"),index=False)

    # Save metrics summary
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(os.path.join(MODELS_DIR, "metrics_summary.csv"))
    print(f"\nMetrics saved → {MODELS_DIR}/metrics_summary.csv")
    return all_metrics


def load_model(target: str, dataset: str) -> Pipeline:
    """Load a saved model. dataset = 'hosp' or 'non_hosp'."""
    path = os.path.join(MODELS_DIR, f"xgb_{target}_{dataset}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}.\n"
            "Run: python src/ml_models.py  (or train_models.py) first."
        )
    return joblib.load(path)


def predict_single(input_dict: dict, target: str, dataset: str) -> dict:
    """
    Predict for a single household.

    Parameters
    ----------
    input_dict : dict  — feature values (human-readable labels already applied)
    target     : str   — 'CHE' | 'under_poverty_since_OOPE' | 'distress'
    dataset    : str   — 'hosp' | 'non_hosp'

    Returns
    -------
    dict with 'prediction' (0/1) and 'probability' (float)
    """
    pipe = load_model(target, dataset)
    row  = pd.DataFrame([input_dict])
    pred = int(pipe.predict(row)[0])
    prob = float(pipe.predict_proba(row)[0, 1])
    return {"prediction": pred, "probability": round(prob * 100, 1)}


# ─────────────────────────────────────────────────────────────────
# CLI: python src/ml_models.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_all()