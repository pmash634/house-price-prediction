# 🏠 House Price Prediction — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-Score_0.14768-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

---

## 📌 Project Overview

An end-to-end regression pipeline predicting residential property sale prices using the
[Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset (1,460 training samples, 80 features).

The focus was on **principled feature engineering** and **rigorous model evaluation** rather than
defaulting to a black-box model. All preprocessing decisions are documented and justified.

---

## 📊 Results

| Model | 5-Fold CV RMSE (log scale) | Kaggle Public Score |
|---|---|---|
| Random Forest (baseline) | 0.1467 | — |
| **XGBoost (tuned)** ✅ | **0.1454 ± 0.0045** | **0.14768** |

> The close alignment between CV RMSE (0.1454) and Kaggle score (0.14768) confirms the model
> generalises well and is not overfitting to the training data.

---

## 🛠️ Approach

### 1. Feature Engineering
The most impactful step. Rather than using raw columns, I engineered domain-informed features:

| Feature | Description | Rationale |
|---|---|---|
| `HouseAge` | Year sold − Year built | Newer homes command higher prices |
| `RemodelAge` | Year sold − Year remodelled | Recent renovations signal quality |
| `TotalSF` | Above-ground area + basement area | Total usable space is a primary price driver |
| `TotalBath` | Full baths + 0.5×half baths (above + below) | Combined bathroom count |
| `TotalPorchSF` | Sum of all porch/deck areas | Captures outdoor living space |
| `ExterQualScore` | Exterior quality mapped to 1–5 scale | Ordinal encoding preserving order |
| `KitchenQualScore` | Kitchen quality mapped to 1–5 scale | Strong predictor of interior finish |
| `TotalSF_x_Quality` | TotalSF × ExterQualScore | Interaction: large + high quality = premium |
| `Bathrooms_x_Quality` | TotalBath × KitchenQualScore | Interaction: bathrooms × finish level |
| `Bathrooms_x_OverallQual` | TotalBath × OverallQual | Overall quality amplifies bathroom value |
| `MedNhbdPrice` | Median sale price per neighbourhood | Captures location premium |

**Key design decision:** `MedNhbdPrice` was computed from the training set only and then
mapped to the test set — avoiding data leakage from the target variable.

### 2. Target Transformation
Applied `np.log1p()` to `SalePrice` to correct right-skew and stabilise variance.
Predictions are reversed with `np.expm1()` before submission.

### 3. Model Selection
Benchmarked **Random Forest** vs **XGBoost** using 5-fold cross-validation on the
log-transformed target. XGBoost won on RMSE and was selected for submission.

**XGBoost hyperparameters:**
```python
XGBRegressor(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
```

### 4. Validation Strategy
- 5-fold cross-validation on full training set before any test-set prediction
- CV RMSE: **0.1454 ± 0.0045** (low variance = stable model)
- Final model fit on 100% of training data before generating submission

---

## 📁 Project Structure

```
house-price-prediction/
│
├── notebook/
│   └── house_price_corrected.ipynb   # Full pipeline notebook
│
├── submission/
│   └── submission_xgb_corrected.csv  # Kaggle submission (score: 0.14768)
│
└── README.md
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/pmash634/house-price-prediction.git
cd house-price-prediction

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# 3. Download the dataset from Kaggle
# kaggle.com/c/house-prices-advanced-regression-techniques → Data → Download

# 4. Run the notebook
jupyter notebook notebook/house_price_corrected.ipynb
```

---

## 🔑 Key Learnings

- **Feature engineering > model complexity.** Adding `MedNhbdPrice` and interaction terms
  contributed more to RMSE reduction than hyperparameter tuning alone.
- **Log-transforming the target is essential** for right-skewed regression problems —
  it stabilises variance and makes RMSE a more meaningful metric.
- **CV score and Kaggle score alignment** (0.1454 vs 0.14768) validates that 5-fold CV
  is a reliable proxy for generalisation on this dataset.
- **Consistent feature engineering across train and test** is critical — a mismatch
  in even one feature silently corrupts all predictions.

---

## 👩‍💻 Author

**Purity Wanjiku Macharia**
🔗 [LinkedIn](https://linkedin.com/in/purity-wanjiku-11581b200) · 💻 [GitHub](https://github.com/pmash634) · 📩 puritymacharia111@gmail.com
