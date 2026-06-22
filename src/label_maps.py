"""
label_maps.py
=============
Single source of truth for all categorical label mappings.
Import from here in every module to keep labels consistent.
"""

HH_SIZE_MAP = {0: "≤4 members", 1: ">4 members"}

HH_EDUCATION_MAP = {
    0: "Below Secondary",
    1: "Below Graduation",
    2: "Graduation & above"
}

RELIGION_MAP = {0: "Other", 1: "Hinduism", 2: "Islam"}

SOCIAL_GROUP_MAP = {1: "ST", 2: "SC", 3: "OBC", 4: "General"}

AGE_60_MAP = {0: "Below 60", 1: "60 & above"}

GENDER_MAP = {1: "Male", 2: "Female", 3: "Transgender"}

ECONOMIC_QUINTILE_MAP = {
    1: "1st Quintile (Poorest)",
    2: "2nd Quintile",
    3: "3rd Quintile",
    4: "4th Quintile",
    5: "5th Quintile (Richest)"
}

SECTOR_MAP = {1: "Rural", 2: "Urban"}

INCOME_SOURCE_MAP = {
    1: "Self-employed",
    2: "Regular Wage",
    3: "Casual Labour",
    4: "Other",
    9: "Other"
}

NSS_STATE_MAP = {
    1: "Jammu & Kashmir", 2: "Himachal Pradesh", 3: "Punjab",
    4: "Chandigarh", 5: "Uttarakhand", 6: "Haryana", 7: "Delhi",
    8: "Rajasthan", 9: "Uttar Pradesh", 10: "Bihar", 11: "Sikkim",
    12: "Arunachal Pradesh", 13: "Nagaland", 14: "Manipur",
    15: "Mizoram", 16: "Tripura", 17: "Meghalaya", 18: "Assam",
    19: "West Bengal", 20: "Jharkhand", 21: "Odisha",
    22: "Chhattisgarh", 23: "Madhya Pradesh", 24: "Gujarat",
    25: "Daman & Diu", 27: "Maharashtra", 28: "Andhra Pradesh",
    29: "Karnataka", 30: "Goa", 31: "Lakshadweep", 32: "Kerala",
    33: "Tamil Nadu", 34: "Puducherry",
    35: "Andaman & Nicobar Islands", 36: "Telangana", 37: "Ladakh",
    99: "Unknown"
}

DISEASE_COLS = [
    "INFECTION", "CANCERS", "BLOOD_DISEASES", "ENDOCRINE",
    "PSYCHIATRIC", "EYE", "EAR", "CARDIO_VASCULAR",
    "RESPIRATORY", "GASTRO", "SKIN", "MUSCULO", "GENITO",
    "OBSTETRIC", "INJURIES", "KIDNEY_FAILURE",
    "SYMPTOM_UNKNOWN", "NO_SYMPTOM_INFO", "CHILDBIRTH"
]

HOSPITAL_TYPE_COLS = ["Govt", "Charity", "Private"]

ALL_FEATURES = [
    "sector", "HH_size", "HH_education", "Religion",
    "age_of_household_base60", "insurance_cover", "Gender",    "female_ratio",
    "max_education","economic_quintile", "Mejor_source_of_income", "social_group",
    "state",
    *HOSPITAL_TYPE_COLS,
    *DISEASE_COLS
]

CATEGORICAL_FEATURES = [
    "sector", "HH_size", "HH_education", "Religion",
    "age_of_household_base60", "insurance_cover", "Gender",
    "economic_quintile", "Mejor_source_of_income", "social_group",
    "state"
]
NUMERICAL_FEATURES = [
    "female_ratio",
    "max_education",
]

TARGET_LABELS = {
    "CHE": "Catastrophic Health Expenditure",
    "under_poverty_since_OOPE": "Pushed into Poverty by OOPE",
    "distress": "Distress Financing"
}

