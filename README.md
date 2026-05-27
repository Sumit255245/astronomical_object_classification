# 🔭 Astronomical Object Classification

## Overview

This project builds a machine learning pipeline to automatically classify astronomical objects observed by the **Sloan Digital Sky Survey (SDSS DR17)** into three categories:

- 🌟 **Star** — a single luminous ball of plasma like our Sun
- 🌌 **Galaxy** — a massive system of billions of stars, like the Milky Way
- ⚡ **Quasar (QSO)** — an extremely luminous galaxy core powered by a supermassive black hole

The project covers the full ML lifecycle — from raw data loading and EDA, through feature engineering, model training and comparison, hyperparameter tuning, final evaluation, and a live Streamlit web app for real-time predictions.

## Dataset and Features

The dataset is the [Stellar Classification Dataset — SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) from Kaggle, containing **100,000 observations** and **17 features**.

| Feature Type | Features | Description |
|---|---|---|
| Photometric Bands | `u`, `g`, `r`, `i`, `z` | Brightness measured through 5 colour filters (ultraviolet → infrared) |
| Spectral | `redshift` | How much light is shifted red — indicates distance and velocity |
| Engineered | `u_g`, `g_r`, `r_i`, `i_z` | Colour indices (difference between adjacent bands) |
| Target | `class` | STAR, GALAXY, or QSO |

> **Key insight:** Redshift alone almost perfectly separates the three classes. Stars have redshift ≈ 0, galaxies ≈ 0.1–0.8, and quasars ≈ 0.5–5.0.

## Data Analysis Workflow

- **Data Loading & Cleaning:** Loaded 100,000 SDSS observations, dropped non-predictive metadata columns (telescope IDs, sky coordinates), encoded class labels
- **Exploratory Data Analysis:** Visualised class distribution, redshift distributions, photometric band boxplots, feature correlation heatmap, and pairplots
- **Feature Engineering:** Created 4 colour index features (`u-g`, `g-r`, `r-i`, `i-z`) from raw band magnitudes, capturing spectral slope for better class separation
- **Preprocessing:** Stratified 70/15/15 train/validation/test split, StandardScaler fitted only on training data to prevent data leakage
- **Model Training & Comparison:** Trained and compared 8 ML algorithms using 5-fold cross-validation
- **Hyperparameter Tuning:** RandomizedSearchCV on Random Forest across 6 hyperparameters
- **Final Evaluation:** Confusion matrix, ROC-AUC curves, and feature importance analysis on held-out test set

## Model Results

| Model | Validation Accuracy |
|---|---|
| **Random Forest** ✅ | **97.83%** |
| Gradient Boosting | 97.81% |
| XGBoost | 97.67% |
| Decision Tree | 96.63% |
| K-Nearest Neighbors | 93.47% |
| Logistic Regression | 95.54% |
| Naive Bayes | 71.58% |

> Best model after hyperparameter tuning with RandomizedSearchCV: **Random Forest — 97.8%+ accuracy**

## Key Findings

- **Redshift** is the most dominant feature by a large margin — the model relies on it more than all band features combined
- **Colour indices** (engineered features) ranked higher than raw band values, confirming the value of feature engineering
- **Class imbalance** (GALAXY 59%, STAR 22%, QSO 19%) was handled with stratified splitting
- **Sky coordinates** (RA/Dec) had zero predictive power and were dropped

## Steps to Run

1. Clone the repository:
```
git clone https://github.com/Sumit255245/astronomical-object-classification
cd astronomical-object-classification
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run app.py
```
