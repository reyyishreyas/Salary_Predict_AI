<div align="center">

# 🎯 SalaryPredict AI

**An end-to-end machine learning platform for accurate, fair, and explainable salary predictions**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-black?logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-blue)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Overview

SalaryPredict AI is a full-stack ML project that trains and compares multiple regression models to predict a candidate's expected CTC (Cost to Company) based on their profile — experience, education, skills, location, and more. It ships with a dark-themed Flask web application for single predictions, bulk CSV uploads, analytics dashboards, and fairness auditing.

> **Best model achieved: 99.98% accuracy (R² = 0.9998) using a Stacking Ensemble of XGBoost + Gradient Boosting + Random Forest.**

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔮 **Single Prediction** | Enter candidate details and get an instant salary estimate with ±10% confidence range |
| 📤 **Batch Prediction** | Upload a CSV of candidates and download bulk predictions |
| 📊 **Analytics Dashboard** | Salary distributions, department trends, experience-vs-salary scatter plots |
| ⚖️ **Fairness Analysis** | Salary disparity auditing across location, department, and education level |
| 🧠 **Model Comparison** | Side-by-side performance metrics across 4 algorithms + ensemble |
| 📈 **Feature Importance** | Understand which candidate attributes drive the prediction |

---

## 🗂️ Repository Structure

```
SalaryPredict_AI/
│
├── 📁 cleaning_splitting/          # Data preparation pipeline
│   ├── data_cleaning.py            # Feature selection & stratified split
│   └── data_splitting.py           # Simple 80/20 split utility
│
├── 📁 training/                    # Ensemble training & visualisation
│   ├── train_ensemble.py           # Voting + Stacking ensemble trainer
│   └── visualise.py                # Prediction evaluation plots
│
├── 📁 flask_app/                   # Web application
│   ├── app.py                      # Flask routes & prediction logic
│   ├── static/logo.png             # Brand logo
│   └── templates/
│       ├── base.html               # Shared navbar & dark theme layout
│       ├── dashboard.html          # Landing page with stat cards & charts
│       ├── predict.html            # Single-candidate prediction form
│       ├── batch.html              # CSV upload & bulk prediction
│       ├── analytics.html          # Feature importance & salary trends
│       └── fairness.html           # Bias / fairness audit charts
│
├── 📁 processed/                   # ⚠️ Generated — not tracked in git
│   ├── fair_salary_prediction_dataset.csv   # Master cleaned dataset*
│   ├── X_train.csv                 # Training features
│   ├── X_test.csv                  # Test features
│   ├── y_train.csv                 # Training labels
│   └── y_test.csv                  # Test labels
│
├── 📁 models/                      # ⚠️ .pkl files not tracked in git
│   └── model.pkl                   # Best ensemble model (see Drive link)
│
├── 📁 gradient_boosting/           # Gradient Boosting artefacts
│   ├── model.pkl                   # ⚠️ Not tracked — see Drive link
│   ├── model_metadata.json         # Hyperparams & metrics
│   ├── feature_importance.csv      # Per-feature importances
│   ├── learning_curves.csv         # Train/test R² vs n_estimators
│   ├── test_predictions.csv        # Actual vs predicted on test set
│   └── visualisation/             # 6 evaluation plots
│
├── 📁 xgboost/                     # XGBoost artefacts
│   ├── model.pkl                   # ⚠️ Not tracked — see Drive link
│   ├── model_metadata.json
│   ├── feature_importance.csv
│   ├── test_predictions.csv
│   └── visualisation/             # 4 evaluation plots
│
├── 📁 random_forest/               # Random Forest artefacts
│   ├── model.pkl                   # ⚠️ Not tracked — see Drive link
│   ├── model_metadata.json
│   ├── feature_importance.csv
│   ├── test_predictions.csv
│   ├── performance_report.txt
│   └── visualisation/             # 5 evaluation plots
│
├── 📁 linear_regression/           # ElasticNet + Polynomial artefacts
│   ├── model_pipeline.pkl          # ⚠️ Not tracked — see Drive link
│   ├── model_metadata.json
│   ├── feature_coefficients.csv
│   ├── test_predictions.csv
│   └── visualisation/             # 6 evaluation plots
│
├── 📁 comparison/                  # Cross-model comparison reports
│   ├── data/                       # CSVs: master comparison, ranking, error analysis
│   ├── reports/                    # Comprehensive & quick-reference text reports
│   └── visualizations/            # 5 comparison charts
│
├── 📁 results/                     # Ensemble model output
│   ├── model_metadata.json
│   ├── feature_importance.csv
│   ├── test_predictions.csv
│   └── *.png                       # Evaluation visualisations
│
├── train_gradient_boosting.py      # Train GB model with GridSearchCV
├── train_linear_regression.py      # Train ElasticNet pipeline
├── train_random_forest.py          # Train Random Forest
├── train_xgboost.py               # Train XGBoost with tuning
└── requirements.txt               # Python dependencies
```

> **\*** The master dataset `fair_salary_prediction_dataset.csv` is also available via the Drive link below if not present locally.

---

## 📊 Model Performance Summary

| Model | R² Score | Accuracy | MAE | RMSE | MAPE |
|---|---|---|---|---|---|
| 🥇 **Stacking Ensemble** | **0.9998** | **99.98%** | ₹5,791 | ₹13,884 | **0.50%** |
| 🥈 Gradient Boosting | 0.9998 | 99.98% | ₹6,172 | ₹14,717 | 0.53% |
| 🥉 XGBoost | 0.9998 | 99.98% | ₹7,908 | ₹16,034 | 0.59% |
| Linear Regression | 0.9974 | 99.74% | ₹41,512 | ₹56,367 | 2.45% |
| Random Forest | 0.9888 | 98.88% | ₹82,207 | ₹117,963 | 4.24% |

> Trained on **15,573 samples** | Tested on **3,894 samples** | **29 features**

**Stacking Ensemble breakdown:**
- **93.6%** of predictions within **1%** error
- **98.3%** of predictions within **5%** error
- **99.1%** of predictions within **10%** error

---

## 🔗 Trained Models & Dataset (Google Drive)

> `.pkl` model files are excluded from this repository due to file size constraints.  
> Download them from the link below and place them in the corresponding folders.

| File | Destination Path | Drive Link |
|---|---|---|
| Stacking Ensemble | `models/model.pkl` | 🔗 [Insert Drive Link Here] |
| Gradient Boosting | `gradient_boosting/model.pkl` | 🔗 [Insert Drive Link Here] |
| XGBoost | `xgboost/model.pkl` | 🔗 [Insert Drive Link Here] |
| Random Forest | `random_forest/model.pkl` | 🔗 [Insert Drive Link Here] |
| Linear Regression Pipeline | `linear_regression/model_pipeline.pkl` | 🔗 [Insert Drive Link Here] |
| Dataset | `processed/fair_salary_prediction_dataset.csv` | 🔗 [Insert Drive Link Here] |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/reyyishreyas/SalaryPredict_AI.git
cd SalaryPredict_AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download model files & dataset

Download the files from the Drive links in the table above and place them in the correct paths.

### 5. (Optional) Re-train models from scratch

```bash
# Step 1 – Clean and split the dataset
python cleaning_splitting/data_cleaning.py

# Step 2 – Train individual models
python train_gradient_boosting.py
python train_xgboost.py
python train_random_forest.py
python train_linear_regression.py

# Step 3 – Train the ensemble (uses processed/ splits)
python training/train_ensemble.py

# Step 4 – Generate visualisations
python training/visualise.py
```

### 6. Launch the Flask web app

```bash
cd flask_app
python app.py
```

Then open [http://localhost:50011](http://localhost:50011) in your browser.

---

## 🌐 Web Application Pages

| Route | Page | Description |
|---|---|---|
| `/` | Dashboard | Overview stats, salary distribution, department breakdown |
| `/predict` | Single Prediction | Form-based salary prediction for one candidate |
| `/batch` | Batch Prediction | Upload CSV → download bulk results |
| `/analytics` | Analytics | Feature importance, department salary, experience scatter |
| `/fairness` | Fairness Audit | Salary disparity by location, department, education |

---

## 🧪 Feature Engineering

The model uses **29 predictive features** derived from the raw dataset:

| Category | Features |
|---|---|
| **Experience** | `Total_Experience`, `Total_Experience_in_field_applied` |
| **Compensation** | `Current_CTC`, `Inhand_Offer` |
| **Education** | `Has_PG`, `Has_PHD`, `International_degree_any`, `Passing_Year_Of_Graduation/PG/PHD` |
| **Career** | `No_Of_Companies_worked`, `Certifications`, `Number_of_Publications` |
| **Performance** | `Performance_Score`, `Field_Specialization_Ratio` |
| **Encoded Categoricals** | `Department_encoded`, `Role_encoded`, `Industry_encoded`, `Location_encoded`, etc. |

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| **ML Models** | scikit-learn, XGBoost |
| **Data Processing** | pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Web Framework** | Flask 2.3+ |
| **Frontend** | Bootstrap 5.3, Font Awesome 6 |
| **Model Persistence** | joblib |
| **Geo / Location** | pycountry, geonamescache |

---

## 📁 Data Notes

- Raw dataset contains salary information for **~19,000 applicants** across industries in India.
- Target variable: `Expected_CTC` (annual Cost to Company in ₹)
- Stratified 80/20 train-test split ensures salary range coverage across splits.
- All sensitive personally identifiable information (names, contact details) is excluded from features.

---

## 👤 Author

**Reyyi Shreyas**  
Built for accurate, fair, and actionable salary predictions to help HR and recruitment teams make data-driven decisions.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
