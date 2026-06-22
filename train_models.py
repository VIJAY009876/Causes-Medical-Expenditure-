# train_models.py

"""
train_models.py
===============
Run ONCE before launching the Streamlit app.

    python train_models.py

Trains all 6 XGBoost models, saves:
  - models/xgb_<target>_<dataset>.pkl      ← model files
  - models/metrics_summary.csv             ← performance metrics
  - models/feature_importance_<name>.csv   ← feature importances
  - models/stat_results/<key>.csv          ← pre-fitted GLM results
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import numpy as np
import joblib

MODELS_DIR = "models"
DATA_DIR   = os.path.join("data", "processed_data")
STAT_DIR   = os.path.join(MODELS_DIR, "stat_results")


def save_feature_importance(pipe, model_name):
    """Extract and save feature importances from trained XGBoost pipeline."""
    try:
        xgb = pipe.named_steps["classifier"]
        pre = pipe.named_steps["preprocessor"]

        # Feature names after OneHot
        ohe_names = list(pre.named_transformers_["cat"].get_feature_names_out())
        remainder_names = [
            f for f in pre.feature_names_in_
            if f not in pre.transformers_[0][2]   # not in cat list
        ]
        all_names = ohe_names + remainder_names

        importances = xgb.feature_importances_
        n = min(len(importances), len(all_names))
        fi_df = pd.DataFrame({
            "feature":    all_names[:n],
            "importance": importances[:n]
        }).sort_values("importance", ascending=False)

        path = os.path.join(MODELS_DIR, f"feature_importance_{model_name}.csv")
        fi_df.to_csv(path, index=False)
        print(f"  Feature importance saved → {path}")
    except Exception as e:
        print(f"  Could not save feature importance: {e}")


def train_xgb_models():
    from ml_models import train_and_evaluate, TARGETS

    os.makedirs(MODELS_DIR, exist_ok=True)

    hosp     = pd.read_csv(os.path.join(DATA_DIR, "hospital_model.csv"),     index_col=0)
    non_hosp = pd.read_csv(os.path.join(DATA_DIR, "non_hospital_model.csv"), index_col=0)

    datasets = {"hosp": hosp, "non_hosp": non_hosp}
    labels   = {"hosp": "Hospitalization", "non_hosp": "Non-Hospitalization"}
    all_metrics = {}

    for ds_key, df in datasets.items():
        for target in TARGETS:
            if target not in df.columns:
                print(f"  Skipping {target} — not in {ds_key}")
                continue

            model_name = f"xgb_{target}_{ds_key}"
            pipe, metrics = train_and_evaluate(df, target, labels[ds_key])

            path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
            joblib.dump(pipe, path)
            print(f"  Model saved → {path}")

            save_feature_importance(pipe, model_name)
            all_metrics[model_name] = metrics

    pd.DataFrame(all_metrics).T.to_csv(os.path.join(MODELS_DIR, "metrics_summary.csv"))
    print(f"\n✅ Metrics saved → {MODELS_DIR}/metrics_summary.csv")


def fit_glm_cache():
    """Pre-fit and save all GLM results so the Statistics page never refits."""
    os.makedirs(STAT_DIR, exist_ok=True)

    MODEL_KEYS = [
        "oope_hosp", "oope_non_hosp",
        "che_hosp",  "che_non_hosp",
        "pl_hosp",   "pl_non_hosp",
        "distress_hosp", "distress_non_hosp",
    ]
    all_present = all(os.path.exists(os.path.join(STAT_DIR, f"{k}.csv")) for k in MODEL_KEYS)
    if all_present:
        print("\n✅ GLM cache already exists — skipping re-fit.")
        return

    print("\nFitting statistical models (GLM)...")
    from statistics_models import run_all_models
    hosp     = pd.read_csv(os.path.join(DATA_DIR, "hospital_model.csv"),     index_col=0)
    non_hosp = pd.read_csv(os.path.join(DATA_DIR, "non_hospital_model.csv"), index_col=0)
    hosp.fillna(0, inplace=True); non_hosp.fillna(0, inplace=True)

    results = run_all_models(hosp, non_hosp)
    for k, df in results.items():
        df.to_csv(os.path.join(STAT_DIR, f"{k}.csv"), index=False)
        print(f"  GLM result saved → {STAT_DIR}/{k}.csv")
    print("✅ All GLM results cached.")


if __name__ == "__main__":
    print("=" * 55)
    print("  NSSO Health App — Training & Caching Pipeline")
    print("=" * 55)

    print("\n[1/2] Training XGBoost models...")
    train_xgb_models()

    print("\n[2/2] Pre-fitting & caching GLM results...")
    fit_glm_cache()

    print("\n" + "=" * 55)
    print("  All done! Launch the app with:")
    print("  streamlit run app.py")
    print("=" * 55)
