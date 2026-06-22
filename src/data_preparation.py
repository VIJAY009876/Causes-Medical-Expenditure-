
import pandas as pd
import numpy as np
import os

class NSSDataBuilder:

    def __init__(self, DF1, DF2, DF4, DF5):

        # ── Block 1: household-level info ──────────────────────────
        self.df1 = DF1[["unique_no", "sector", "state", "HUCE","social_group", "h_size", "h_type","religion","mult1"]].copy()


        # Economic quintile via weighted cumulative distribution
        self.df1 = self.df1.sort_values("HUCE")
        self.df1["cum_wt"] = self.df1["mult1"].cumsum()
        total_wt = self.df1["mult1"].sum()
        self.df1["pct"] = self.df1["cum_wt"] / total_wt
        self.df1["economic_quintile"] = pd.cut(
            self.df1["pct"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
            labels=[1, 2, 3, 4, 5]
        )

        # ── Block 2: individual-level info ─────────────────────────
        self.df2 = DF2[["unique_no", "relation_with_Head", "Gender", "Age","H_education", "Whether_hosp","Whether_last_15_days_dis_yORn","insurance_covered_yORn", "mult2"]].copy()


        # ── Block 4: hospitalization expenditure ───────────────────
        self.df4 = DF4[["unique_no", "mejor_source_for_finance","Nature_Dis", "type_Hopital","total_expences", "total_reimbursed"]].copy()
            
            
            
        

        # ── Block 5: outpatient expenditure (rename to match df4) ──
        DF5 = DF5.rename(columns={
            "level_of_care_last_15days": "type_Hopital",
            "mejor_source_for_finance_last_15days": "mejor_source_for_finance",
            "Nature_Dis_lst_15_days": "Nature_Dis",
            "total_expences_last_15days_Rs": "total_expences",
            "total_reimbursed_last_15days_Rs": "total_reimbursed"
        })

        self.df5 = DF5[[
            "unique_no", "mejor_source_for_finance", "Nature_Dis",
            "type_Hopital", "total_expences", "total_reimbursed"
        ]].copy()

        # Collapse hospital type: >2 → 3 (Private)
        self.df5["type_Hopital"] = np.where(
            self.df5["type_Hopital"] > 2, 3, self.df5["type_Hopital"]
        )

        # ── Household head rows ────────────────────────────────────
        self.d2 = self.df2[self.df2["relation_with_Head"] == 1].copy()

        # ── Insurance coverage flag ────────────────────────────────
        # FIX: renamed 'insurence_cover' → 'insurance_cover' everywhere
        self.df2["insurance_cover"] = np.where(
            self.df2["insurance_covered_yORn"] < 19, 1, 0
        )
        self.insu_cover_id = self.df2[
            self.df2["insurance_cover"] == 1
        ]["unique_no"].unique()

    # ─────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────

    def _create_disease(self, data):
        """Map disease codes to 19 named disease categories via crosstab."""

        v = data["Nature_Dis"].dropna().unique()
        v.sort()
        v = np.insert(v, 0, 0)

        # Safe slicing — handles datasets with fewer codes
        def safe_slice(arr, start, end=None):
            end = end or start + 1
            return arr[start:end] if len(arr) > start else []

        disease_map = {
            1:  safe_slice(v, 1, 13),
            2:  safe_slice(v, 13, 14),
            3:  safe_slice(v, 14, 17),
            4:  safe_slice(v, 17, 21),
            5:  safe_slice(v, 21, 28),
            6:  safe_slice(v, 28, 33),
            7:  safe_slice(v, 33, 35),
            8:  safe_slice(v, 35, 37),
            9:  safe_slice(v, 37, 40),
            10: safe_slice(v, 40, 44),
            11: safe_slice(v, 44, 45),
            12: safe_slice(v, 45, 47),
            13: safe_slice(v, 47, 50),
            14: safe_slice(v, 50, 53),
            15: safe_slice(v, 53, 60),
            16: safe_slice(v, 60, 61),
            17: safe_slice(v, 61, 62),
            18: safe_slice(v, 62, 63),
            19: safe_slice(v, 63, 66),
        }

        disease_labels = [
            "INFECTION", "CANCERS", "BLOOD_DISEASES", "ENDOCRINE",
            "PSYCHIATRIC", "EYE", "EAR", "CARDIO_VASCULAR",
            "RESPIRATORY", "GASTRO", "SKIN", "MUSCULO", "GENITO",
            "OBSTETRIC", "INJURIES", "KIDNEY_FAILURE",
            "SYMPTOM_UNKNOWN", "NO_SYMPTOM_INFO", "CHILDBIRTH"
        ]

        # FIX: completed the reverse-mapping loop (was truncated in Colab)
        code_map = {}
        for disease_group, codes in disease_map.items():
            for code in codes:
                code_map[code] = disease_group

        data = data.copy()
        data["disease_group"] = data["Nature_Dis"].map(code_map)

        # One-hot via crosstab
        dummies = pd.crosstab(data["unique_no"], data["disease_group"])
        dummies = (dummies > 0).astype(int).reset_index()

        # Rename numeric group columns → label columns
        rename = {i + 1: disease_labels[i] for i in range(19)}
        dummies = dummies.rename(columns=rename)

        # Ensure all 19 columns exist
        for lbl in disease_labels:
            if lbl not in dummies.columns:
                dummies[lbl] = 0

        return dummies[["unique_no"] + disease_labels]

    def _create_hospital_type(self, data):
        """One-hot encode hospital type (Govt / Charity / Private)."""

        data = data.copy()
        data["type_Hopital"] = np.where(
            data["type_Hopital"] > 2, 3, data["type_Hopital"]
        )
        type_map = {1: "Govt", 2: "Charity", 3: "Private"}
        data["type_Hopital"] = data["type_Hopital"].map(type_map)

        hosp_type = pd.crosstab(data["unique_no"], data["type_Hopital"])
        hosp_type = (hosp_type > 0).astype(int).reset_index()
        return hosp_type

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def get_hospitalization_data(self):
        """Return household-level hospitalization dataset."""

        # Households with at least one admitted member
        hosp_ids = (
            self.df2.groupby("unique_no")["Whether_hosp"]
            .value_counts()
            .reset_index()
        )
        ids = hosp_ids[hosp_ids["Whether_hosp"] == 1]["unique_no"]

        hh_head = self.d2[self.d2["unique_no"].isin(ids)].copy()
        hh_head["insurance_cover"] = np.where(
            hh_head["unique_no"].isin(self.insu_cover_id), 1, 0
        )

        left = pd.merge(self.df1, hh_head, on="unique_no", how="inner")

        right = (
            self.df4.groupby("unique_no")[["total_expences", "total_reimbursed"]]
            .sum()
            .reset_index()
        )
        # OOPE annualised (÷12) for hospitalization
        right["OOPE"] = (right["total_expences"] - right["total_reimbursed"]) / 12

        data = pd.merge(left, right, on="unique_no", how="inner")

        distress_ids = self.df4[
            self.df4["mejor_source_for_finance"] != 1
        ]["unique_no"]
        data["distress"] = np.where(data["unique_no"].isin(distress_ids), 1, 0)

        hosp_type = self._create_hospital_type(self.df4)
        disease    = self._create_disease(self.df4)

        data = pd.merge(data, hosp_type, on="unique_no", how="left")
        data = pd.merge(data, disease,    on="unique_no", how="left")

        
        #max_education   &  feamale_ratio
        max_edu = (
                self.df2.groupby("unique_no")["H_education"]
                .max()
                .reset_index(name="max_education"))

        female_ratio = (
                    self.df2.groupby("unique_no")["Gender"]
                    .apply(lambda x: (x == 2).mean())
                    .reset_index(name="female_ratio")
                )
        data = data.merge(max_edu, on="unique_no", how="left")
        data = data.merge(female_ratio, on="unique_no", how="left")

        return data

    def get_non_hospitalization_data(self):
        """Return household-level outpatient (15-day) dataset."""

        nh_ids = (
            self.df2.groupby("unique_no")["Whether_last_15_days_dis_yORn"]
            .value_counts()
            .reset_index()
        )
        ids = nh_ids[
            nh_ids["Whether_last_15_days_dis_yORn"] == 1
        ]["unique_no"]

        hh_head = self.d2[self.d2["unique_no"].isin(ids)].copy()
        hh_head["insurance_cover"] = np.where(
            hh_head["unique_no"].isin(self.insu_cover_id), 1, 0
        )

        left = pd.merge(self.df1, hh_head, on="unique_no", how="inner")

        right = (
            self.df5.groupby("unique_no")[["total_expences", "total_reimbursed"]]
            .sum()
            .reset_index()
        )
        # OOPE annualised (×2 for 15-day → monthly → annual)
        right["OOPE"] = (right["total_expences"] - right["total_reimbursed"]) * 2

        data = pd.merge(left, right, on="unique_no", how="inner")

        distress_ids = self.df5[
            self.df5["mejor_source_for_finance"] != 1
        ]["unique_no"]
        data["distress"] = np.where(data["unique_no"].isin(distress_ids), 1, 0)

        hosp_type = self._create_hospital_type(self.df5)
        disease    = self._create_disease(self.df5)

        data = pd.merge(data, hosp_type, on="unique_no", how="left")
        data = pd.merge(data, disease,    on="unique_no", how="left")

        
        #max_education   &  feamale_ratio
        max_edu = (
                self.df2.groupby("unique_no")["H_education"]
                .max()
                .reset_index(name="max_education"))



        female_ratio = (
                    self.df2.groupby("unique_no")["Gender"]
                    .apply(lambda x: (x == 2).mean())
                    .reset_index(name="female_ratio")
                )
        data = data.merge(max_edu, on="unique_no", how="left")
        data = data.merge(female_ratio, on="unique_no", how="left")

        return data


# ─────────────────────────────────────────────────────────────────
# CLI: python src/data_preparation.py
# ─────────────────────────────────────────────────────────────────
# CLI: python src/data_preparation.py
# ─────────────────────────────────────────────────────────────────

import os
import pandas as pd

if __name__ == "__main__":

    DATA = "data/processed_data"

    DF1 = pd.read_csv(os.path.join(DATA, "DF1.csv"))
    DF2 = pd.read_csv(os.path.join(DATA, "DF2.csv"))
    DF4 = pd.read_csv(os.path.join(DATA, "DF4.csv"))
    DF5 = pd.read_csv(os.path.join(DATA, "DF5.csv"))

    builder = NSSDataBuilder(DF1, DF2, DF4, DF5)

    hosp = builder.get_hospitalization_data()
    non_hosp = builder.get_non_hospitalization_data()


    hosp.to_csv(os.path.join(DATA, "hospital_data.csv"), index=False)
    non_hosp.to_csv(os.path.join(DATA, "non_hospital_data.csv"), index=False)

    print(f"hospital_data    : {hosp.shape}")
    print(f"non_hospital_data: {non_hosp.shape}")