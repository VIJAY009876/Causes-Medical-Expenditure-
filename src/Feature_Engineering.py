"""
feature_engineering.py
=======================
FeatureEngineering — transforms raw NSS datasets into
model-ready datasets with derived features.

Features added:
- CHE (Catastrophic Health Expenditure)
- under_poverty_since_OOPE
- age_of_household_base60
- HH_education / HH_size categories
- Religion grouping
- Mejor_source_of_income
- Categorical dtype assignment

Fixes from Colab:
- 'insurence_cover' → 'insurance_cover' (consistent spelling)
- Added __main__ block for standalone use
"""

import pandas as pd
import numpy as np

# ── Rural / Urban poverty lines (monthly per capita, Rs.) ────────
RURAL_POVERTY_LINE = 1509.6
URBAN_POVERTY_LINE = 1805.0


class FeatureEngineering:

    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Age: elderly head (>60) ───────────────────────────────
        df["age_of_household_base60"] = (df["Age"] > 60).astype(int)

        # ── Education: 0=below secondary, 1=below grad, 2=grad+ ──
        df["HH_education"] = np.where(
            df["H_education"] < 6, 0,
            np.where(df["H_education"] < 11, 1, 2)
        )

        # ── Household size: 0=≤4 members, 1=>4 members ───────────
        df["HH_size"] = (df["h_size"] > 4).astype(int)

        # ── OOPE positive flag ────────────────────────────────────
        df["oope_positive"] = (df["OOPE"] > 0).astype(int)

        # ── CHE: OOPE > 40% of non-food expenditure ──────────────
        pov_line = np.where(
            df["sector"] == 1,
            RURAL_POVERTY_LINE * df["h_size"],
            URBAN_POVERTY_LINE * df["h_size"]
        )
        nfe = df["HUCE"] - pov_line          # non-food expenditure proxy
        # Avoid division by zero
        nfe_safe = np.where(nfe <= 0, np.nan, nfe)
        che_ratio = df["OOPE"] / nfe_safe
        df["CHE"] = np.where(che_ratio > 0.40, 1, 0)

        # ── Poverty after OOPE ────────────────────────────────────
        above_line   = df["HUCE"] >= pov_line
        below_after  = (df["HUCE"] - df["OOPE"]) < pov_line
        df["under_poverty_since_OOPE"] = np.where(
            above_line & below_after, 1, 0
        )

        # ── Religion: 1=Hindu, 2=Islam, 0=Other ──────────────────
        df["Religion"] = np.where(df["religion"].isin([1, 2]), df["religion"], 0)

        # ── Major income source ───────────────────────────────────
        def _income_detect(h_type, sector):
            if sector == 1:                      # Rural
                if h_type in [1, 2]:  return 1  # Self-employed
                if h_type in [3, 4]:  return 2  # Regular wage
                if h_type in [5, 6]:  return 3  # Casual labour
                return 4                          # Other
            return h_type                         # Urban: keep as-is

        df["Mejor_source_of_income"] = df.apply(
            lambda r: _income_detect(r["h_type"], r["sector"]), axis=1
        )

        # ── Categorical dtypes ────────────────────────────────────
        cat_cols = [
            "sector", "social_group", "economic_quintile",
            "HH_education", "HH_size", "Religion",
            "Mejor_source_of_income", "distress",
            "insurance_cover",                   # FIX: correct spelling
            "age_of_household_base60",
            "under_poverty_since_OOPE", "CHE"
        ]
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df


# ─────────────────────────────────────────────────────────────────
# CLI: python src/feature_engineering.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    DATA = "data/processed_data"
    data_h  = pd.read_csv(os.path.join(DATA, "hospital_data.csv"))
    data_nh = pd.read_csv(os.path.join(DATA, "non_hospital_data.csv"))

    hospital_model     = FeatureEngineering.add_features(data_h)
    non_hospital_model = FeatureEngineering.add_features(data_nh)


    hospital_model.to_csv(os.path.join(DATA, "hospital_model.csv"),     index=False)
    non_hospital_model.to_csv(os.path.join(DATA, "non_hospital_model.csv"), index=False)

    print(f"hospital_model    : {hospital_model.shape}")
    print(f"non_hospital_model: {non_hospital_model.shape}")
    print("\nCHE distribution (hospitalization):")
    print(hospital_model["CHE"].value_counts(normalize=True).mul(100).round(1))