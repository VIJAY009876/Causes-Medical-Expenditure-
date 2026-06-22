# Medical Expenditure Determinants and Prediction

A Streamlit-based analytics dashboard for studying the determinants of medical expenditure using the NSSO 80th Round Health Consumption Expenditure Survey (2022–23).

The project combines statistical modeling and machine learning techniques to analyze healthcare spending patterns, Catastrophic Health Expenditure (CHE), distress financing, and poverty impacts.

---

## Objectives

- Identify determinants of Out-of-Pocket Expenditure (OOPE)
- Analyze Catastrophic Health Expenditure (CHE)
- Examine distress financing due to healthcare costs
- Study poverty impacts resulting from medical expenditure
- Compare hospitalization and non-hospitalization expenditure patterns
- Build predictive machine learning models for healthcare financial outcomes

---

## Project Structure

```
Causes-Medical-Expenditure-/
├── app.py                        ← Main entry point
├── train_models.py               ← Run ONCE to train & cache everything
│
├── src/
│   ├── data_preparation.py       ← NSSDataBuilder class
│   ├── feature_engineering.py    ← FeatureEngineering class
│   ├── statistics_models.py      ← Weighted GLM models
│   ├── ml_models.py              ← XGBoost training & prediction
│   └── label_maps.py             ← All categorical label mappings
│
├── pages/
│   ├── 1_EDA.py                  ← Exploratory Data Analysis
│   ├── 2_Statistics.py           ← GLM results (cached to disk)
│   └── 3_Prediction.py           ← ML prediction interface
│
├── data/
│   └── processed_data/
│       ├── hospital_model.csv    ← Place your processed data here
│       └── non_hospital_model.csv
│
├── models/                       ← Auto-created by train_models.py
│   ├── xgb_CHE_hosp.pkl
│   ├── xgb_CHE_non_hosp.pkl
│   ├── metrics_summary.csv
│   ├── feature_importance_*.csv
│   └── stat_results/             ← Pre-fitted GLM CSVs
│       ├── oope_hosp.csv
│       ├── che_hosp.csv
│       ├── pl_hosp.csv
│       ├── distress_hosp.csv
│       └── ...
│
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/VIJAY009876/Causes-Medical-Expenditure-.git
cd Causes-Medical-Expenditure-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place your processed NSSO CSV files in `data/processed_data/`:
- `hospital_model.csv`
- `non_hospital_model.csv`

### 4. Train Models (Run Once)

```bash
python train_models.py
```

This script will:
- Train all 6 XGBoost models (CHE, Distress Financing, Poverty for both hospitalization types)
- Pre-fit all GLM statistical models
- Cache results to disk for fast page loads
- Generate feature importance files and performance metrics

### 5. Launch the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501/`

---

## Dashboard Features

### 📊 Exploratory Data Analysis (EDA)
- Visualize OOPE, CHE, distress financing, and poverty across:
  - States and regions
  - Income quintiles
  - Hospital types (public/private)
  - Disease categories
  - Rural vs. urban sectors
  - Social groups and insurance coverage

### 📐 Statistical Analysis
- **Weighted GLM models** with population-representative estimates
- **OOPE models**: Gamma GLM for out-of-pocket expenditures
- **CHE models**: Logistic regression for catastrophic health expenditure
- **Poverty Line**: Impact models for poverty after OOPE
- **Distress Financing**: Models with odds ratios and confidence intervals
- Pre-fitted results cached on disk—no re-computation on page load

### 🤖 ML Prediction Engine
- Enter household characteristics to predict:
  - Probability of catastrophic health expenditure (CHE)
  - Risk of poverty after OOPE
  - Likelihood of distress financing
- Feature importance visualization
- Separate models for hospitalization and non-hospitalization cases

---

## Statistical Methodology

### Two-Part Model

#### Part 1: Weighted Logistic Regression (GLM Binomial)
Estimates the probability of incurring healthcare expenditure.

#### Part 2: Weighted Least Squares (WLS)
Models the logarithm of Out-of-Pocket Expenditure (OOPE) conditional on positive expenditure.

### Explanatory Variables

- Disease Category
- Hospital Type
- Economic Quintile
- Rural/Urban Sector
- Social Group
- Insurance Coverage
- Household Size
- Age and Gender

---

## Machine Learning Models

### XGBoost Models (6 Total)

**Hospitalization Cases:**
- CHE (Catastrophic Health Expenditure)
- Distress Financing
- Poverty Impact

**Non-Hospitalization Cases:**
- CHE (Catastrophic Health Expenditure)
- Distress Financing
- Poverty Impact

---

## Data Source

**National Sample Survey Office (NSSO) 80th Round**
- Health Consumption Expenditure Survey (2022–23)
- National Statistical Office (NSO), Government of India
- Covers hospitalization (Blocks 6, 7) and outpatient care (Blocks 8, 9)
- Population-representative estimates using multiplier weights (`mult1`)
- Includes all Indian states and union territories

---

## Tools Used

- **Python** — Core language
- **Streamlit** — Web application framework
- **Pandas & NumPy** — Data manipulation
- **Statsmodels** — Statistical modeling (GLM)
- **Scikit-Learn** — Preprocessing and utilities
- **XGBoost** — Machine learning models
- **Matplotlib, Seaborn** — Data visualization
- **Joblib** — Model caching and serialization

---

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit main page |
| `train_models.py` | One-time setup: trains models and pre-fits GLMs |
| `pages/1_EDA.py` | Interactive exploratory data analysis |
| `pages/2_Statistics.py` | Pre-fitted GLM results and hypothesis testing |
| `pages/3_Prediction.py` | ML model prediction interface |
| `src/data_preparation.py` | Data loading and preprocessing |
| `src/feature_engineering.py` | Feature creation and transformation |
| `src/statistics_models.py` | GLM model definitions and fitting |
| `src/ml_models.py` | XGBoost pipeline and training |
| `src/label_maps.py` | Categorical variable mappings |

---

## Author

**Vijay B**  
M.Sc. Mathematics and Statistics  
Indian Institute of Technology Tirupati

---


## Reference

This project was inspired by and builds upon the methodology presented in:

Nanda, M., & Sharma, R. (2023). *A Comprehensive Examination of the Economic Impact of Out-of-Pocket Health Expenditures in India*. Health Policy and Planning, 38(8), 926–938.

Key concepts adapted from the study include:

- Catastrophic Health Expenditure (CHE)
- Poverty Headcount Ratio
- Distressed Financing
- Disease-wise Financial Burden Analysis



## Acknowledgement

This project was developed during an internship under the Ministry of Statistics and Programme Implementation (MoSPI), Government of India, using NSSO 80th Round Health Consumption Expenditure Survey data.


For questions or issues, please open a GitHub issue.
