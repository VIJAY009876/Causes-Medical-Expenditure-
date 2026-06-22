"""
statistics_models.py
====================
Weighted GLM statistical models for NSS health survey data.

Models:
- Gamma GLM (log link)  → OOPE (continuous, right-skewed)
- Binomial GLM (logit)  → CHE, Poverty Line, Distress (binary)

Fixes from Colab:
- x_features string trailing whitespace/newline stripped before formula use
- 'insurence_cover' → 'insurance_cover'
- Results returned as clean DataFrames (not just printed)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

from label_maps import (
    HH_SIZE_MAP, HH_EDUCATION_MAP, RELIGION_MAP, SOCIAL_GROUP_MAP,
    AGE_60_MAP, GENDER_MAP, ECONOMIC_QUINTILE_MAP, SECTOR_MAP,
    INCOME_SOURCE_MAP, NSS_STATE_MAP
)

# ── RHS formula (strip fixes the trailing-newline bug from Colab) ──
X_FEATURES = """
C(sector) +
C(HH_size) +
C(HH_education) +
C(Religion) +
C(age_of_household_base60) +
C(insurance_cover) +
C(Gender) +
C(economic_quintile) +
C(Mejor_source_of_income) +
C(social_group) +
Charity + Govt + Private +
INFECTION + CANCERS + BLOOD_DISEASES + ENDOCRINE + PSYCHIATRIC +
EYE + EAR + CARDIO_VASCULAR + RESPIRATORY + GASTRO + SKIN +
MUSCULO + GENITO + OBSTETRIC + INJURIES + KIDNEY_FAILURE +
SYMPTOM_UNKNOWN + NO_SYMPTOM_INFO
""".strip()


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
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
    return df


def gamma_glm(data: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Weighted Gamma GLM (log link) for OOPE.
    Returns a DataFrame with coefficients and % change.

    FIX: freq_weights caused MemoryError (expands data to population size).
         Use var_weights with NORMALISED weights instead — same coefficient
         estimates, correct standard errors, no memory explosion.
    """
    df = data[data["OOPE"] > 0].copy()   # Gamma requires positive response

    # Normalise weights to sum to sample size (not population size)
    w = df["mult1"].astype(float)
    df["_w"] = w / w.sum() * len(df)

    formula = f"OOPE ~ {X_FEATURES}"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=df["_w"]          # ← normalised; no row expansion
    ).fit(disp=0)

    ci = model.conf_int()
    result = pd.DataFrame({
        "Variable":       model.params.index,
        "Coefficient":    model.params.values,
        "Std_Error":      model.bse.values,
        "t_value":        model.tvalues.values,
        "p_value":        model.pvalues.values,
        "CI_Lower":       ci[0].values,
        "CI_Upper":       ci[1].values,
    })
    result["Percent_Change"] = (np.exp(result["Coefficient"]) - 1) * 100
    result["Significant"]    = result["p_value"] < 0.05
    result = result.round(4)
    if label:
        result.insert(0, "Dataset", label)
    return result


def binomial_glm(data: pd.DataFrame, target: str, label: str = "") -> pd.DataFrame:
    """
    Weighted Binomial GLM (logit link) for binary outcomes.
    Returns a DataFrame with Odds Ratios and confidence intervals.

    target: 'CHE' | 'under_poverty_since_OOPE' | 'distress'

    FIX: freq_weights caused MemoryError. Use normalised var_weights instead.
    """
    df = data.copy()

    # Normalise weights to sum to sample size (not population size)
    w = df["mult1"].astype(float)
    df["_w"] = w / w.sum() * len(df)

    formula = f"{target} ~ {X_FEATURES}"

    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        var_weights=df["_w"]          # normalised; no row expansion
    ).fit(disp=0)

    ci = model.conf_int()
    result = pd.DataFrame({
        "Variable":    model.params.index,
        "Coefficient": model.params.values,
        "Std_Error":   model.bse.values,
        "z_value":     model.tvalues.values,
        "p_value":     model.pvalues.values,
        "Odds_Ratio":  np.exp(model.params.values),
        "OR_CI_Lower": np.exp(ci[0].values),
        "OR_CI_Upper": np.exp(ci[1].values),
    })
    result["Significant"] = result["p_value"] < 0.05
    result = result.round(4)
    if label:
        result.insert(0, "Dataset", label)
    return result


def run_all_models(hosp: pd.DataFrame, non_hosp: pd.DataFrame) -> dict:
    """
    Run all 8 models and return results as a dict of DataFrames.
    Keys: 'oope_hosp', 'oope_non_hosp',
          'che_hosp', 'che_non_hosp',
          'pl_hosp',  'pl_non_hosp',
          'distress_hosp', 'distress_non_hosp'
    """
    hosp_lbl     = apply_labels(hosp.fillna(0))
    non_hosp_lbl = apply_labels(non_hosp.fillna(0))

    return {
        "oope_hosp":         gamma_glm(hosp_lbl,     "Hospitalization"),
        "oope_non_hosp":     gamma_glm(non_hosp_lbl, "Non-Hospitalization"),
        "che_hosp":          binomial_glm(hosp_lbl,     "CHE",                    "Hospitalization"),
        "che_non_hosp":      binomial_glm(non_hosp_lbl, "CHE",                    "Non-Hospitalization"),
        "pl_hosp":           binomial_glm(hosp_lbl,     "under_poverty_since_OOPE","Hospitalization"),
        "pl_non_hosp":       binomial_glm(non_hosp_lbl, "under_poverty_since_OOPE","Non-Hospitalization"),
        "distress_hosp":     binomial_glm(hosp_lbl,     "distress",               "Hospitalization"),
        "distress_non_hosp": binomial_glm(non_hosp_lbl, "distress",               "Non-Hospitalization"),
    }


# ─────────────────────────────────────────────────────────────────
# CLI: python src/statistics_models.py
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    DATA = "data"
    hosp     = pd.read_csv(os.path.join(DATA, "hospital_model.csv"),     index_col=0)
    non_hosp = pd.read_csv(os.path.join(DATA, "non_hospital_model.csv"), index_col=0)

    results = run_all_models(hosp, non_hosp)
    for key, df in results.items():
        print(f"\n{'='*60}\n  {key}\n{'='*60}")
        print(df[["Variable", "Coefficient", "p_value", "Significant"]].to_string(index=False))