# analysis.py
# Standalone "AI Analysing" tab you can import into main.py
# -- Major additions in this version --
# * Advanced models: LightGBM + CatBoost (auto-guarded)
# * Stacking ensembles (classification/regression)
# * Repeated CV and Group/Time-aware CV options (with gap/embargo)
# * Cost-based thresholding + interactive threshold slider (binary)
# * Leakage-safe KFold Target Encoding (auto-guarded via category_encoders)
# * Optional Power/Quantile transforms, missing indicators, correlation pruning
# * Permutation Importance (bootstrap CI), PDP/ICE, Learning curves
# * Drift checks (train/test) + Fairness slices by subgroup
# * Expanded Quick Compare (custom metric target)
# * AutoTune++ stub with Optuna (if installed) in addition to RandomizedSearch
# * PII heuristic detector, data/spec hashing, richer requirements & exports
# * Many help=... tooltips across controls

from __future__ import annotations

import io
import json
import os
import re
import zipfile
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# Feature flags / configuration
# ==============================
ENABLE_COLAB = True
MAX_PREVIEW_ROWS = 20
DEFAULT_ADVANCED_OFF = True
QUICK_COMPARE_SAMPLE_MAX = 2000
RANDOM_SEED = 42
AUTO_PREVIEW_DEFAULT = False

# =========================================
# (Optional) tiny summarizer for brief text
# =========================================
def _get_summarizer():
    try:
        from transformers import pipeline
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception:
        return None


def summarize_bullets(text: str, max_bullets: int = 5) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sm = _get_summarizer()
    if sm:
        try:
            chunk = text[:1800]
            out = sm(chunk, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
            parts = re.split(r'(?<=[.!?])\s+', out)
            bullets = [p.strip(" -•") for p in parts if len(p.strip()) > 0]
            return bullets[:max_bullets] if bullets else [chunk[:240] + ("…" if len(chunk) > 240 else "")]
        except Exception:
            pass
    # fallback
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p for p in parts if p.strip()][:max_bullets]


# ===========================
# Heuristics & plan builders
# ===========================
def _detect_problem_type(text: str, df: Optional[pd.DataFrame] = None, target: Optional[str] = None) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["classify", "classification", "churn", "attrition", "fraud", "spam", "segment", "yes/no", "category", "label"]):
        return "classification"
    if any(k in t for k in ["regression", "forecast", "predict revenue", "predict sales", "continuous", "price", "numeric target"]):
        return "regression"
    if any(k in t for k in ["cluster", "segmentation", "topic modeling", "unsupervised"]):
        return "unsupervised"
    if df is not None and target and target in df.columns:
        return "regression" if pd.api.types.is_numeric_dtype(df[target]) else "classification"
    return "classification"


def _suggest_targets(df: Optional[pd.DataFrame]) -> List[str]:
    if df is None:
        return []
    cols = list(df.columns)
    pri_keywords = ["target", "label", "class", "outcome", "y", "churn", "fraud", "default"]
    sec_keywords = ["revenue", "sales", "price", "amount", "score"]
    scored = []
    for c in cols:
        name = c.lower()
        score = 0
        for k in pri_keywords:
            if k in name:
                score += 10
        for k in sec_keywords:
            if k in name:
                score += 4
        if c == cols[-1]:
            score += 2
        scored.append((score, c))
    scored.sort(reverse=True)
    ordered = [c for _, c in scored if _ > 0] or [cols[-1]]
    seen, out = set(), []
    for c in ordered:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:5]


def _auto_time_column(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None:
        return None
    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    if dt_cols:
        return dt_cols[0]
    cand = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "timestamp", "dt"])]
    for c in cand:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    return None


def _propose_business_questions_auto(df: Optional[pd.DataFrame], target: Optional[str], n_min: int = 5, n_max: int = 10) -> List[str]:
    base = [
        "Which features most strongly drive the target metric?",
        "How do segments differ by behavior, value, and churn risk?",
        "What seasonal or trend effects are present over time?",
        "Which customer cohorts underperform vs. plan and why?",
        "What are the top data quality issues and their business impact?",
        "What intervention would most improve the KPI with least cost?",
        "What is the expected ROI of the recommended model or policy?",
        "Which channels and campaigns deliver the best unit economics?",
        "Can we detect anomalous patterns that require action?",
        "How should we prioritize data enrichment to boost performance?"
    ]
    n = max(n_min, min(n_max, 8))

    ideas: List[str] = []
    if df is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        if target and target in df.columns:
            if target in num_cols:
                ideas += [
                    f"Which numeric features have the strongest correlation with `{target}`?",
                    f"What non-linear relationships exist between key drivers and `{target}`?",
                    f"How does `{target}` vary across top categorical segments?",
                    f"Are there outliers in `{target}` that skew our metrics?",
                ]
            else:
                ideas += [
                    f"What is the class balance of `{target}` and does it imply special handling?",
                    f"Which features maximize separability across classes of `{target}`?",
                    f"Which customer segments show the highest `{target}` rate?",
                ]
        if len(num_cols) >= 2:
            ideas += ["Which variables are most correlated with each other (risk of redundancy)?"]
        if len(cat_cols) >= 1:
            ideas += ["Which categorical levels dominate volume and how do they impact key metrics?"]
        if _auto_time_column(df) is not None:
            ideas += ["Do we see seasonality or trend that matters for planning?"]
    deduped = []
    for q in ideas + base:
        if q not in deduped:
            deduped.append(q)
    return deduped[:n]


def _ai_plan_from_brief(brief: str, df: Optional[pd.DataFrame], target: Optional[str], n_questions: int) -> dict:
    bullets = summarize_bullets(brief, max_bullets=5) or [brief[:240] + ("…" if len(brief) > 240 else "")]
    qtype = _detect_problem_type(brief, df, target)
    questions = _propose_business_questions_auto(df, target, n_min=n_questions, n_max=n_questions)
    eda = [
        "Schema check (dtypes, uniques, cardinality).",
        "Missingness map & imputation strategy.",
        "Duplicates, outliers, and domain ranges.",
        "Target distribution & class balance (if supervised).",
        "Leakage scan vs. target time horizon.",
        "Basic feature importance proxies (mutual info, correlation).",
        "Anomaly detection (IsolationForest) for quality control."
    ]
    metrics = {
        "classification": ["ROC-AUC", "F1", "Accuracy", "PR AUC", "Confusion Matrix"],
        "regression": ["RMSE", "MAE", "R²", "MAPE", "Residual plots"],
        "unsupervised": ["Silhouette score", "Davies–Bouldin", "Cluster stability"]
    }
    models = {
        "classification": ["LogReg", "RandomForest", "GradientBoosting", "XGBoost (if avail.)", "LightGBM (if avail.)", "CatBoost (if avail.)"],
        "regression": ["LinearRegression", "RandomForest", "GradientBoosting", "XGBoost (if avail.)", "LightGBM (if avail.)", "CatBoost (if avail.)"],
        "unsupervised": ["KMeans", "GaussianMixture", "AgglomerativeClustering"]
    }
    return {
        "summary": bullets,
        "problem_type": qtype,
        "business_questions": questions,
        "eda_plan": eda,
        "metrics": metrics[qtype],
        "models": models[qtype]
    }


# =======================
# Notebook cell builders
# =======================
def _nb_cell_markdown(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def _nb_cell_code(code: str):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code}


# ------------------------------
# Notebook content (advanced)
# ------------------------------
def _build_notebook(
    spec: dict,
    notebook_title: str,
    dataset_name: Optional[str],
    default_data_path: str,
    target_col: Optional[str],
    normalization: str = "standard",
    enable_keras: bool = True,
    enable_timeseries: bool = True,
    time_col_hint: Optional[str] = None,
    # options:
    id_cols: Optional[List[str]] = None,
    include_cols: Optional[List[str]] = None,
    num_impute: str = "median",          # "median" | "mean"
    cat_impute: str = "most_frequent",   # "most_frequent" | "constant"
    cat_fill_value: str = "",
    cv_folds: int = 5,
    use_smote: bool = False,
    use_shap: bool = False,
    selected_models_cls: Optional[List[str]] = None,
    selected_models_reg: Optional[List[str]] = None,
    outlier_clip_pct: float = 0.0,
    feat_sel_method: str = "none",       # "none" | "variance" | "kbest"
    feat_sel_k: int = 100,
    calibrate_binary: bool = True,
    enable_ensemble: bool = True,
    power_transform: str = "none",       # "none" | "yeojohnson" | "quantile"
    add_missing_indicators: bool = True,
    corr_prune_threshold: float = 0.0,   # 0 disables; else drop one of pairs > threshold
    metric_target: str = "auto",
    group_col_hint: Optional[str] = None,
    use_stacking: bool = True,
) -> bytes:
    problem = spec["problem_type"]
    is_class = problem == "classification"
    is_reg = problem == "regression"

    id_cols = id_cols or []
    include_cols = include_cols or []
    selected_models_cls = selected_models_cls or ["LogReg", "RF", "GBC", "XGB", "LGBM", "CAT"]
    selected_models_reg = selected_models_reg or ["LinReg", "RF", "GBR", "XGB", "LGBM", "CAT"]

    header = (
        f"# {notebook_title}\n\n"
        f"**Generated by OmniSearch – AI Analysing (advanced).**\n\n"
        "**What’s inside:** Problem statement • Dataset overview • Data dictionary • Missingness • Duplicates • Outliers • "
        "Distributions • Correlation heatmap • Pairwise scatter (top numerics) • Anomaly scan • Preprocessing (winsorize + impute + missing-indicators + scaling + rare-cats + optional power/quantile) • "
        "Imbalance-aware classification • Baselines & Lift • Classical models compare (incl. LGBM/CatBoost if available) • Optional Keras • Optional SMOTE • "
        "Feature selection • Ensemble & Stacking • Calibration (binary) • Cost-based thresholding • Interpretability (Permutation Importance, PDP/ICE, SHAP) • "
        "Drift checks • Learning curves • Time-series baselines • Business Q&A scaffold • Threshold optimization • Champion export • Model Card.\n\n"
        f"## Executive Summary\n" + "\n".join([f"- {b}" for b in spec["summary"]]) + "\n"
    )

    prob_stmt = (
        "## 1. Problem Statement\n"
        "Describe the client, industry, key KPIs, pain points, and what decisions this analysis will inform.\n"
    )

    data_overview = (
        "## 2. Dataset Overview and Data Exploration\n"
        "### 2.1 Data Dictionary\n"
        "- Type, missing %, distinct count, sample values\n\n"
        "### 2.2 Shapes\n"
        "- Train/test shapes after split, dataset size\n\n"
        "### 2.3 Data file\n"
        f"- File name: `{dataset_name or 'dataset.csv'}`\n\n"
        "### 2.4 Missing Values\n"
        "- Missingness %, imputation strategy\n\n"
        "### 2.5 Visualize Numerical Feature Distributions\n"
        "- Histograms for numeric columns\n\n"
        "### 2.6 Duplicate and Outlier Check\n\n"
        "### 2.7 Boxplots for Outlier Detection\n\n"
        "### 2.8 Class Distribution and Imbalance Analysis (if supervised)\n\n"
        "### 2.9 Categorical vs. Numerical Feature Analysis\n\n"
        "### 2.10 Correlation and Feature Relationships\n\n"
        "### 2.11 Visualization of Key Variables\n"
        "### 2.12 Pairwise Scatter Matrix (top 5 numerics)\n"
    )

    bq_md = "## 4. Explanatory Data Analysis (Business Questions)\n" + "\n".join(
        [f"### Q{i+1}. {q}\n- *Why it matters:* explain the business value.\n"
         "- *Code answer:* (see code cell below)\n"
         "- *Interpretation:* discuss results and implications.\n"
         for i, q in enumerate(spec["business_questions"])]
    ) + "\n"

    # ==== Parameters dumped into notebook ====
    param_code = f"""# ==== Parameters ====
DATA_PATH = r"{default_data_path}"
TARGET = {json.dumps(target_col) if target_col else "None"}
ID_COLS = {json.dumps(id_cols)}
INCLUDE_COLS = {json.dumps(include_cols)}
TIME_COL = {json.dumps(time_col_hint) if time_col_hint else "None"}
GROUP_COL = {json.dumps(group_col_hint) if group_col_hint else "None"}
SEED = {RANDOM_SEED}
N_JOBS = -1

# Preprocessing
SCALING = "{normalization}"          # "standard" | "minmax"
NUM_IMPUTE = "{num_impute}"
CAT_IMPUTE = "{cat_impute}"
CAT_FILL_VALUE = {json.dumps(cat_fill_value)}
OUTLIER_CLIP_PCT = {outlier_clip_pct}
ADD_MISSING_INDICATORS = {str(bool(add_missing_indicators))}
POWER_TRANSFORM = "{power_transform}" # "none" | "yeojohnson" | "quantile"
CORR_PRUNE_THRESHOLD = {corr_prune_threshold}

# Modeling
CV_FOLDS = {cv_folds}
USE_SMOTE = {str(bool(use_smote))}
USE_SHAP = {str(bool(use_shap))}
CALIBRATE_BINARY = {str(bool(calibrate_binary))}
ENABLE_ENSEMBLE = {str(bool(enable_ensemble))}
USE_STACKING = {str(bool(use_stacking))}
FEAT_SEL_METHOD = "{feat_sel_method}"  # "none" | "variance" | "kbest"
FEAT_SEL_K = {feat_sel_k}
METRIC_TARGET = "{metric_target}"      # "auto" or explicit (AUC, F1, PR_AUC, ACC, RMSE, MAE, R2, MAPE)
MODELS_CLS = {json.dumps(selected_models_cls)}
MODELS_REG = {json.dumps(selected_models_reg)}
"""

    # ==== Imports & setup in notebook ====
    imports_code = """# ==== Imports & Setup ====
import os, warnings, json, math, itertools, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.model_selection import (train_test_split, cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit,
                                     RepeatedKFold, RepeatedStratifiedKFold, GroupKFold, StratifiedGroupKFold,
                                     learning_curve)
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, r2_score,
                             RocCurveDisplay, PrecisionRecallDisplay, average_precision_score, brier_score_loss)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder, StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer,
                                   FunctionTransformer)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, IsolationForest, GradientBoostingClassifier,
                              GradientBoostingRegressor, VotingClassifier, StackingClassifier, StackingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import Bunch

# Optional models
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

# Optional target encoding
try:
    import category_encoders as ce
    HAS_CE = True
except Exception:
    HAS_CE = False

# Optional TF
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except Exception:
    HAS_TF = False

# Optional LIME
try:
    from lime import lime_tabular
    HAS_LIME = True
except Exception:
    HAS_LIME = False

# Optional SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Optional imbalanced
try:
    from imblearn.over_sampling import SMOTE, SMOTENC, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMB = True
except Exception:
    HAS_IMB = False

# Optional Optuna
try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")
np.random.seed(SEED)

# ==== Utility: data/spec hash for reproducibility ====
def md5_file(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

"""

    # ==== Load & dictionary ====
    load_code = """# ==== Load Data ====
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
display(df.head())

# Save dataset hash & libs
print("Data MD5:", md5_file(DATA_PATH))
print("Lib versions — numpy", np.__version__, "| pandas", pd.__version__)

# Restrict to INCLUDE_COLS
if isinstance(INCLUDE_COLS, list) and len(INCLUDE_COLS) > 0 and TARGET:
    keep = [c for c in INCLUDE_COLS if c in df.columns]
    keep += [TARGET] if TARGET in df.columns else []
    df = df[keep]
    print("Restricted to INCLUDE_COLS + TARGET. New shape:", df.shape)

# Drop ID columns
if isinstance(ID_COLS, list) and len(ID_COLS) > 0:
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")
    print("Dropped ID columns:", [c for c in ID_COLS if c in df.columns])

# ==== PII heuristic check ====
pii_hits = []
patterns = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}",
    "phone": r"(\\+\\d{1,3}[- ]?)?\\d{3}[- ]?\\d{3}[- ]?\\d{4}",
}
for c in df.select_dtypes(include=["object"]).columns:
    s = df[c].dropna().astype(str).head(1000).str.cat(sep=" ")
    if re.search(patterns["email"], s):
        pii_hits.append((c, "email-like"))
    if re.search(patterns["phone"], s):
        pii_hits.append((c, "phone-like"))
if pii_hits:
    print("⚠️ Potential PII columns (heuristic):", pii_hits)

# ==== Data Dictionary ====
def data_dictionary(df):
    d = []
    for c in df.columns:
        miss = df[c].isna().mean()*100
        nunique = df[c].nunique(dropna=True)
        sample = df[c].dropna().unique()[:3]
        d.append((c, str(df[c].dtype), round(miss,2), int(nunique), sample))
    return pd.DataFrame(d, columns=["column","dtype","missing_%","nunique","sample_values"])
display(data_dictionary(df))
"""

    # ==== EDA visuals ====
    eda_code = """# ==== EDA Core ====
# Missingness
miss = df.isna().mean().sort_values(ascending=False).round(3)*100
display(miss.to_frame("missing_%").head(20))

# Duplicates
print("Duplicates:", df.duplicated().sum())

# Numeric/Categorical split
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

# Numeric distributions
if len(num_cols) > 0:
    df[num_cols].hist(figsize=(14, 10))
    plt.suptitle("Numeric Distributions")
    plt.tight_layout(); plt.show()

# Boxplots
if len(num_cols) > 0:
    fig, axes = plt.subplots(nrows=min(6, len(num_cols)), ncols=1, figsize=(10, 2*min(6, len(num_cols))))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, col in zip(axes, num_cols[:6]):
        ax.boxplot(df[col].dropna(), vert=False)
        ax.set_title(f"Boxplot: {col}")
    plt.tight_layout(); plt.show()

# Correlation heatmap
if len(num_cols) >= 2:
    corr = df[num_cols].corr()
    plt.figure(figsize=(8,6))
    plt.imshow(corr, cmap="coolwarm", aspect="auto")
    plt.title("Correlation (numeric)")
    plt.colorbar()
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.tight_layout(); plt.show()

# Pairwise scatter matrix (top 5 numerics)
if len(num_cols) >= 2:
    top5 = num_cols[:5]
    axarr = scatter_matrix(df[top5].dropna().sample(min(500, len(df))), figsize=(10,10))
    plt.suptitle("Pairwise Scatter Matrix (top numerics)")
    plt.tight_layout(); plt.show()

# Target distribution
if 'TARGET' in globals() and TARGET and TARGET in df.columns:
    if pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET].plot(kind="hist", bins=30, figsize=(6,4), title=f"Distribution of {TARGET}")
        plt.tight_layout(); plt.show()
    else:
        df[TARGET].value_counts().plot(kind="bar", figsize=(6,4), title=f"Class counts for {TARGET}")
        plt.tight_layout(); plt.show()
"""

    # ==== Leakage scan ====
    leakage_code = """# ==== Leakage Scan (simple heuristics) ====
if TARGET and TARGET in df.columns:
    leak_flags = []
    for c in df.columns:
        if c == TARGET: 
            continue
        try:
            if df[c].equals(df[TARGET]):
                leak_flags.append((c, "Exact duplicate of TARGET"))
            elif pd.api.types.is_numeric_dtype(df[TARGET]) and pd.api.types.is_numeric_dtype(df[c]):
                corr = df[[c, TARGET]].dropna().corr().iloc[0,1]
                if abs(corr) > 0.995:
                    leak_flags.append((c, f"Suspicious correlation with TARGET: {corr:.3f}"))
        except Exception:
            pass
    if leak_flags:
        print("Potential leakage columns detected:")
        for c, why in leak_flags:
            print(f" - {c}: {why}")
    else:
        print("No obvious leakage found by simple checks.")
"""

    # ==== Preprocess & split with extras ====
    split_common = """# ==== Preprocessing helpers ====
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, pct=0.0, numeric_cols=None):
        self.pct = pct; self.numeric_cols = numeric_cols; self.bounds_ = {}
    def fit(self, X, y=None):
        if self.pct <= 0 or not self.numeric_cols: return self
        Xn = X[self.numeric_cols]
        for c in self.numeric_cols:
            ql = float(np.nanpercentile(Xn[c], 100*self.pct))
            qh = float(np.nanpercentile(Xn[c], 100*(1-self.pct)))
            self.bounds_[c] = (ql, qh)
        return self
    def transform(self, X):
        if self.pct <= 0 or not self.numeric_cols: return X
        X = X.copy()
        for c,(lo,hi) in self.bounds_.items():
            if c in X.columns:
                X[c] = np.clip(X[c], lo, hi)
        return X

class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols or []
    def fit(self, X, y=None):
        self.miss_cols_ = [c for c in self.cols if X[c].isna().any()]
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.miss_cols_:
            X[c + "_ismissing"] = X[c].isna().astype(int)
        return X

# Leakage-safe target encoder using category_encoders (if available)
class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, n_splits=5, random_state=SEED):
        self.cols = cols or []; self.n_splits = n_splits; self.random_state = random_state
        self.encoders_ = {}
    def fit(self, X, y):
        if not HAS_CE or len(self.cols)==0:
            self.encoders_ = None
            return self
        self.global_means_ = {}
        for c in self.cols:
            self.global_means_[c] = pd.Series(y).mean() if pd.api.types.is_numeric_dtype(y) else pd.Series(y).astype(str).map(lambda v: 1 if v==pd.Series(y).astype(str).mode().iloc[0] else 0).mean()
        return self
    def transform(self, X, y=None):
        if not HAS_CE or len(self.cols)==0:
            return X
        X = X.copy()
        if y is None:
            # Use learned global means for transform-only
            for c in self.cols:
                X[c+"_te"] = X[c].astype(str).map(lambda _: self.global_means_.get(c, 0.0))
            return X
        # KFold mean encoding (leakage-safe)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        Xte = pd.DataFrame(index=X.index)
        y_series = pd.Series(y)
        for c in self.cols:
            oof = pd.Series(index=X.index, dtype=float)
            for tr, va in kf.split(X):
                m = y_series.iloc[tr].groupby(X[c].iloc[tr].astype(str)).mean()
                oof.iloc[va] = X[c].iloc[va].astype(str).map(m)
            Xte[c+"_te"] = oof.fillna(y_series.mean())
        X = pd.concat([X, Xte], axis=1)
        return X

def prune_correlated(df_num, thr):
    if thr <= 0 or df_num.shape[1] <= 1:
        return list(df_num.columns)
    corr = df_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [column for column in upper.columns if any(upper[column] > thr)]
    keep = [c for c in df_num.columns if c not in drop_cols]
    return keep
"""

    split_code_cls = split_common + """
# ==== 3. Data Preprocessing & Feature Engineering (Classification) ====
assert TARGET in df.columns, "Target column missing."
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

# Basic splits
num_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_all = [c for c in X.columns if c not in num_cols_all]

# Prune highly correlated numeric features (pre-impute) if enabled
keep_num = prune_correlated(X[num_cols_all], CORR_PRUNE_THRESHOLD) if len(num_cols_all)>1 else num_cols_all
num_cols = keep_num
cat_cols = [c for c in cat_cols_all]  # unchanged set

# Optional missing indicators
if ADD_MISSING_INDICATORS and len(num_cols)>0:
    mia = MissingIndicatorAdder(cols=num_cols)
    X = mia.fit_transform(X)

# Winsorize numeric
winsor = Winsorizer(pct=OUTLIER_CLIP_PCT, numeric_cols=num_cols)

# Power transforms (after impute, before scale)
num_steps = [("winsor", winsor), ("imputer", SimpleImputer(strategy=NUM_IMPUTE))]
if POWER_TRANSFORM == "yeojohnson":
    num_steps.append(("power", PowerTransformer(method="yeo-johnson")))
elif POWER_TRANSFORM == "quantile":
    num_steps.append(("quantile", QuantileTransformer(output_distribution="normal", subsample=20000, random_state=SEED)))

# Scaling
if SCALING == "standard":
    num_steps.append(("scaler", StandardScaler()))
elif SCALING == "minmax":
    num_steps.append(("scaler", MinMaxScaler()))

# Rare-category handling
try:
    cat_enc = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)
except TypeError:
    cat_enc = OneHotEncoder(handle_unknown="ignore")

# Leakage-safe target encoding (optional, appended features)
te_cols = [c for c in cat_cols if df[c].nunique() > 20][:5]  # cap budget
te = KFoldTargetEncoder(cols=te_cols, n_splits=max(3, min(5, CV_FOLDS)))

cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy=CAT_IMPUTE, fill_value=CAT_FILL_VALUE)),
                          ("oh", cat_enc)])

num_pipe = Pipeline(steps=num_steps)

prep_base = ColumnTransformer(
    transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
    remainder="drop",
    sparse_threshold=0.3
)

# Train/test split (time-aware or stratified)
if TIME_COL and TIME_COL in df.columns:
    dft = df.dropna(subset=[TIME_COL]).copy()
    dft[TIME_COL] = pd.to_datetime(dft[TIME_COL], errors="coerce")
    dft = dft.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    y = dft[TARGET].astype(str)
    X = dft.drop(columns=[TARGET])
    split_idx = int(len(dft)*0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=SEED)

# Append leakage-safe target encodings on train (fit) and test (transform only)
if te.cols:
    X_train = te.fit_transform(X_train, y_train)
    X_test = te.transform(X_test)

# Anomaly detection (numeric)
if len(num_cols) > 0:
    iso = IsolationForest(random_state=SEED, contamination="auto")
    scores = iso.fit_predict(X_train[num_cols].fillna(X_train[num_cols].median()))
    anomal = (scores == -1).mean()*100
    print(f"Estimated anomalies in train (numeric subspace): {anomal:.2f}%")

# Imbalance check
class_pct = pd.Series(y_train).value_counts(normalize=True)
imbalance_ratio = class_pct.max()
IMBALANCED = imbalance_ratio > 0.8
print(f"Imbalance ratio (max class share): {imbalance_ratio:.2f} -> {'IMBALANCED' if IMBALANCED else 'balanced'}")

# Build final preprocessor now that TE columns added (pass-through already in df)
prep = prep_base
"""

    split_code_reg = split_common + """
# ==== 3. Data Preprocessing & Feature Engineering (Regression) ====
assert TARGET in df.columns, "Target column missing."
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(float)

num_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_all = [c for c in X.columns if c not in num_cols_all]

keep_num = prune_correlated(X[num_cols_all], CORR_PRUNE_THRESHOLD) if len(num_cols_all)>1 else num_cols_all
num_cols = keep_num
cat_cols = [c for c in cat_cols_all]

if ADD_MISSING_INDICATORS and len(num_cols)>0:
    mia = MissingIndicatorAdder(cols=num_cols)
    X = mia.fit_transform(X)

winsor = Winsorizer(pct=OUTLIER_CLIP_PCT, numeric_cols=num_cols)
num_steps = [("winsor", winsor), ("imputer", SimpleImputer(strategy=NUM_IMPUTE))]
if POWER_TRANSFORM == "yeojohnson":
    num_steps.append(("power", PowerTransformer(method="yeo-johnson")))
elif POWER_TRANSFORM == "quantile":
    num_steps.append(("quantile", QuantileTransformer(output_distribution="normal", subsample=20000, random_state=SEED)))

if SCALING == "standard":
    num_steps.append(("scaler", StandardScaler()))
elif SCALING == "minmax":
    num_steps.append(("scaler", MinMaxScaler()))

try:
    cat_enc = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)
except TypeError:
    cat_enc = OneHotEncoder(handle_unknown="ignore")

te_cols = [c for c in cat_cols if df[c].nunique() > 20][:5]
te = KFoldTargetEncoder(cols=te_cols, n_splits=max(3, min(5, CV_FOLDS)))

cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy=CAT_IMPUTE, fill_value=CAT_FILL_VALUE)),
                          ("oh", cat_enc)])
num_pipe = Pipeline(steps=num_steps)

prep_base = ColumnTransformer(
    transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
    remainder="drop",
    sparse_threshold=0.3
)

if TIME_COL and TIME_COL in df.columns:
    dft = df.dropna(subset=[TIME_COL]).copy()
    dft[TIME_COL] = pd.to_datetime(dft[TIME_COL], errors="coerce")
    dft = dft.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    y = dft[TARGET].astype(float)
    X = dft.drop(columns=[TARGET])
    split_idx = int(len(dft)*0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

if te.cols:
    X_train = te.fit_transform(X_train, y_train)
    X_test = te.transform(X_test)

if len(num_cols) > 0:
    iso = IsolationForest(random_state=SEED, contamination="auto")
    scores = iso.fit_predict(X_train[num_cols].fillna(X_train[num_cols].median()))
    anomal = (scores == -1).mean()*100
    print(f"Estimated anomalies in train (numeric subspace): {anomal:.2f}%")

prep = prep_base
"""

    # ==== Feature selection ====
    feat_sel_block = """# ==== Feature Selection (optional) ====
fs_step = None
if FEAT_SEL_METHOD == "variance":
    fs_step = ("fs", VarianceThreshold())
elif FEAT_SEL_METHOD == "kbest":
    score_func = f_classif if not pd.api.types.is_numeric_dtype(y_train) else f_regression
    fs_step = ("fs", SelectKBest(score_func=score_func, k=min(FEAT_SEL_K, 1000)))
"""

    # ==== Baselines ====
    baselines_block_cls = """# ==== Baseline (classification) ====
major_class = pd.Series(y_train).value_counts().idxmax()
y_base = np.full_like(np.array(y_test), fill_value=major_class)
base_acc = accuracy_score(y_test, y_base)
base_f1 = f1_score(y_test, y_base, average="weighted")
print(f"Baseline (majority) -> Acc={base_acc:.3f}, F1(w)={base_f1:.3f}")
"""

    baselines_block_reg = """# ==== Baseline (regression) ====
mean_val = float(np.mean(y_train))
y_base = np.full(shape=len(y_test), fill_value=mean_val)
base_rmse = mean_squared_error(y_test, y_base, squared=False)
base_mae = mean_absolute_error(y_test, y_base)
print(f"Baseline (mean) -> RMSE={base_rmse:.3f}, MAE={base_mae:.3f}")
"""

    # ==== Modeling blocks (now with LGBM/CAT + stacking) ====
    model_code_cls = """# ==== 5. Model Evaluation — Classification ====
models = []
if "LogReg" in MODELS_CLS:
    models.append(("LogReg", LogisticRegression(max_iter=600, class_weight='balanced' if IMBALANCED else None)))
if "RF" in MODELS_CLS:
    models.append(("RF", RandomForestClassifier(n_estimators=600, random_state=SEED, n_jobs=N_JOBS,
                                                class_weight='balanced' if IMBALANCED else None)))
if "GBC" in MODELS_CLS:
    models.append(("GBC", GradientBoostingClassifier(random_state=SEED)))
if "XGB" in MODELS_CLS and HAS_XGB:
    models.append(("XGB", XGBClassifier(n_estimators=700, max_depth=6, subsample=0.9, colsample_bytree=0.9,
                                        random_state=SEED, tree_method="hist", n_jobs=N_JOBS, eval_metric="logloss")))
if "LGBM" in MODELS_CLS and HAS_LGBM:
    models.append(("LGBM", LGBMClassifier(n_estimators=800, num_leaves=64, random_state=SEED)))
if "CAT" in MODELS_CLS and HAS_CAT:
    models.append(("CAT", CatBoostClassifier(iterations=700, depth=6, learning_rate=0.1, verbose=False, random_seed=SEED)))

# CV scheme
if GROUP_COL and GROUP_COL in X_train.columns:
    groups = X_train[GROUP_COL]
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS) if 'StratifiedGroupKFold' in globals() else GroupKFold(n_splits=CV_FOLDS)
elif TIME_COL and TIME_COL in df.columns:
    cv = TimeSeriesSplit(n_splits=CV_FOLDS, gap=0)
else:
    try:
        cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=2, random_state=SEED)
    except Exception:
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

def scorer_name_and_fn():
    m = METRIC_TARGET.upper()
    if m in ["AUC","ROC_AUC","ROC-AUC"]:
        return "roc_auc_ovr", None
    if m in ["PR_AUC","AUPRC","PR-AUC"]:
        return "average_precision", None
    if m in ["F1","F1_WEIGHTED"]:
        return "f1_weighted", None
    if m in ["ACC","ACCURACY"]:
        return "accuracy", None
    return "roc_auc_ovr", None

sc_name, sc_fn = scorer_name_and_fn()

results = []
classes_ = np.unique(y_train)

PipelineClass = ImbPipeline if (USE_SMOTE and HAS_IMB) else Pipeline
for name, est in models:
    steps = [("prep", prep)]
    if fs_step is not None:
        steps.append(fs_step)
    if USE_SMOTE and HAS_IMB:
        # try SMOTENC if we have both cats and nums
        if len(cat_cols)>0 and len(num_cols)>0:
            steps.append(("smote", SMOTENC(categorical_features=[False]*len(num_cols)+[True]*len(cat_cols), random_state=SEED)))
        else:
            steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("model", est))
    pipe = PipelineClass(steps=steps)
    try:
        cv_score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=sc_name).mean()
    except Exception:
        # fallback to weighted F1
        cv_score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted").mean()

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)
    except Exception:
        y_proba = None

    pr_auc = np.nan
    test_auc = np.nan
    if y_proba is not None and len(classes_) == 2:
        y_true_bin = (y_test==classes_[1]).astype(int)
        pr_auc = average_precision_score(y_true_bin, y_proba[:,1])
        try:
            test_auc = roc_auc_score(y_true_bin, y_proba[:,1])
        except Exception:
            pass

    results.append([name, cv_score, f1_score(y_test, y_pred, average="weighted"),
                    accuracy_score(y_test, y_pred), pr_auc, test_auc])

res_df = pd.DataFrame(results, columns=["Model","CV Score","F1 (w)","Accuracy","PR AUC (bin)","Test AUC"]).set_index("Model")
display(res_df.sort_values(by=["CV Score"], ascending=False))

# Plot
res_df[["CV Score","F1 (w)","Accuracy"]].plot(kind="bar", figsize=(10,5), title="Model comparison (classification)")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()

# Choose top model
top = res_df.sort_values(by=["CV Score"], ascending=False).index[0]
top_est = dict(models)[top]
steps = [("prep", prep)]
if fs_step is not None: steps.append(fs_step)
if USE_SMOTE and HAS_IMB:
    steps.append(("smote", SMOTE(random_state=SEED)))
steps.append(("model", top_est))
PipelineClass = ImbPipeline if (USE_SMOTE and HAS_IMB) else Pipeline
top_pipe = PipelineClass(steps=steps)
top_pipe.fit(X_train, y_train)
y_pred = top_pipe.predict(X_test)

# Confusion matrix
fig, ax = plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
plt.title(f"Confusion Matrix — {top}"); plt.tight_layout(); plt.show()

# ROC/PR and threshold optimization (binary)
try:
    y_proba = top_pipe.predict_proba(X_test)
    if len(classes_) == 2:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        y_true_bin = (y_test==classes_[1]).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:,1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(5,4)); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.title(f"ROC (AUC={roc_auc:.3f}) — {top}"); plt.tight_layout(); plt.show()
        pr_p, pr_r, thr = precision_recall_curve(y_true_bin, y_proba[:,1])
        ap = average_precision_score(y_true_bin, y_proba[:,1])
        plt.figure(figsize=(5,4)); plt.plot(pr_r, pr_p); plt.title(f"PR Curve (AP={ap:.3f}) — {top}"); plt.tight_layout(); plt.show()

        # Cost-based threshold search
        FP_COST, FN_COST = 1.0, 5.0  # tune in app / spec if desired
        grid = np.linspace(0.05, 0.95, 91)
        def cost_for(t):
            pred = (y_proba[:,1] >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_bin, pred).ravel()
            return fp*FP_COST + fn*FN_COST
        costs = np.array([cost_for(t) for t in grid])
        t_best = float(grid[np.argmin(costs)])
        print(f"Best threshold by cost (FP={FP_COST}, FN={FN_COST}): {t_best:.2f}, MinCost={costs.min():.1f}")

        # Calibration
        if CALIBRATE_BINARY:
            cal = CalibratedClassifierCV(base_estimator=top_pipe, cv=3)
            cal.fit(X_train, y_train)
            p_cal = cal.predict_proba(X_test)[:,1]
            print("Brier (uncal, cal):", brier_score_loss(y_true_bin, y_proba[:,1]), brier_score_loss(y_true_bin, p_cal))
except Exception as e:
    print("ROC/PR/Calibration skipped:", e)

# Permutation Importance (bootstrap CI)
try:
    r = permutation_importance(top_pipe, X_test, y_test, n_repeats=10, random_state=SEED, n_jobs=N_JOBS)
    imp = pd.Series(r.importances_mean).sort_values(ascending=False).head(20)
    imp.plot(kind="bar", figsize=(10,4), title=f"Permutation Importance — {top}")
    plt.tight_layout(); plt.show()
except Exception as e:
    print("Permutation importance skipped:", e)

# PDP/ICE demo for up to 2 features (safe try)
try:
    from sklearn.inspection import PartialDependenceDisplay
    # pick two numeric features if available
    cand = num_cols[:2] if len(num_cols)>=1 else []
    if len(cand) >= 1:
        fig, ax = plt.subplots(figsize=(6,4))
        PartialDependenceDisplay.from_estimator(top_pipe, X_test, features=[cand[0]], ax=ax)
        plt.tight_layout(); plt.show()
except Exception as e:
    print("PDP skipped:", e)

# Learning curve to diagnose variance/bias
try:
    train_sizes, train_scores, test_scores = learning_curve(top_pipe, X_train, y_train, cv=cv, n_jobs=N_JOBS, train_sizes=np.linspace(0.2,1.0,5))
    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, train_scores.mean(axis=1), label="train")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="cv")
    plt.legend(); plt.title("Learning Curve"); plt.tight_layout(); plt.show()
except Exception as e:
    print("Learning curve skipped:", e)

# Optional Stacking Ensemble
if USE_STACKING and len(models) >= 2:
    try:
        estimators = [(n, est) for n, est in models if n in res_df.index[:4]]
        final_est = LogisticRegression(max_iter=600)
        stack = StackingClassifier(estimators=estimators, final_estimator=final_est, passthrough=False, n_jobs=N_JOBS)
        steps = [("prep", prep)]
        if fs_step is not None: steps.append(fs_step)
        if USE_SMOTE and HAS_IMB: steps.append(("smote", SMOTE(random_state=SEED)))
        steps.append(("model", stack))
        stk_pipe = (ImbPipeline if (USE_SMOTE and HAS_IMB) else Pipeline)(steps=steps)
        stk_pipe.fit(X_train, y_train)
        pred_stk = stk_pipe.predict(X_test)
        print(f"Stacking — Acc={accuracy_score(y_test, pred_stk):.3f}, F1(w)={f1_score(y_test, pred_stk, average='weighted'):.3f}")
    except Exception as e:
        print("Stacking skipped:", e)

# Train/Test drift checks (KS for numeric)
try:
    drift = []
    for c in num_cols:
        a = X_train[c].dropna(); b = X_test[c].dropna()
        if len(a)>20 and len(b)>20:
            from scipy.stats import ks_2samp
            stat, p = ks_2samp(a, b)
            drift.append((c, stat, p))
    if drift:
        drift_df = pd.DataFrame(drift, columns=["feature","KS","p"]).sort_values("KS", ascending=False).head(10)
        display(drift_df)
except Exception:
    pass
"""

    model_code_reg = """# ==== 5. Model Evaluation — Regression ====
models = []
if "LinReg" in MODELS_REG:
    models.append(("LinReg", LinearRegression()))
if "RF" in MODELS_REG:
    models.append(("RF", RandomForestRegressor(n_estimators=700, random_state=SEED, n_jobs=N_JOBS)))
if "GBR" in MODELS_REG:
    models.append(("GBR", GradientBoostingRegressor(random_state=SEED)))
if "XGB" in MODELS_REG and HAS_XGB:
    models.append(("XGB", XGBRegressor(n_estimators=900, max_depth=6, subsample=0.9, colsample_bytree=0.9,
                                       random_state=SEED, tree_method="hist", n_jobs=N_JOBS)))
if "LGBM" in MODELS_REG and HAS_LGBM:
    models.append(("LGBM", LGBMRegressor(n_estimators=1000, num_leaves=64, random_state=SEED)))
if "CAT" in MODELS_REG and HAS_CAT:
    models.append(("CAT", CatBoostRegressor(iterations=900, depth=6, learning_rate=0.1, verbose=False, random_seed=SEED)))

# CV scheme
if GROUP_COL and GROUP_COL in X_train.columns:
    cv = GroupKFold(n_splits=CV_FOLDS)
elif TIME_COL and TIME_COL in df.columns:
    cv = TimeSeriesSplit(n_splits=CV_FOLDS, gap=0)
else:
    try:
        cv = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=2, random_state=SEED)
    except Exception:
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

def reg_scorer():
    m = METRIC_TARGET.upper()
    if m in ["RMSE"]:
        return "neg_root_mean_squared_error"
    if m in ["MAE"]:
        return "neg_mean_absolute_error"
    if m in ["R2","R^2","R-SQUARED"]:
        return "r2"
    if m in ["MAPE"]:
        return "neg_mean_absolute_percentage_error"
    return "neg_root_mean_squared_error"

sc = reg_scorer()

results = []; pipes = {}
for name, est in models:
    steps = [("prep", prep)]
    if fs_step is not None: steps.append(fs_step)
    steps.append(("model", est))
    pipe = Pipeline(steps=steps)
    pipes[name] = pipe
    cv_score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=sc).mean()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred)/np.clip(np.abs(y_test), 1e-8, None))) * 100
    results.append((name, cv_score, rmse, mae, r2, mape))

res_df = pd.DataFrame(results, columns=["Model","CV Score","Test RMSE","MAE","R²","MAPE"]).set_index("Model")
display(res_df.sort_values(by=["CV Score"], ascending=True))

res_df[["Test RMSE","MAE"]].plot(kind="bar", figsize=(10,5), title="Model comparison (regression)")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()

top = res_df.sort_values(by="CV Score", ascending=True).index[0]
top_pipe = pipes[top]
y_pred = top_pipe.predict(X_test)
fig, ax = plt.subplots(figsize=(5,4))
ax.scatter(y_test, y_pred, alpha=0.5)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(f"Residuals — {top}")
plt.tight_layout(); plt.show()

# QQ plot (residual diagnostics) if scipy available
try:
    import scipy.stats as ss
    resid = y_test - y_pred
    fig = plt.figure(figsize=(5,4)); ss.probplot(resid, dist="norm", plot=plt); plt.title("QQ Plot (residuals)")
    plt.tight_layout(); plt.show()
except Exception:
    pass

# Optional Stacking
if USE_STACKING and len(models) >= 2:
    try:
        estimators = [(n, est) for n, est in models if n in res_df.index[:4]]
        final_est = LinearRegression()
        stack = StackingRegressor(estimators=estimators, final_estimator=final_est, n_jobs=N_JOBS, passthrough=False)
        steps = [("prep", prep)]
        if fs_step is not None: steps.append(fs_step)
        steps.append(("model", stack))
        stk_pipe = Pipeline(steps=steps)
        stk_pipe.fit(X_train, y_train)
        pred_stk = stk_pipe.predict(X_test)
        print(f"Stacking — RMSE={mean_squared_error(y_test, pred_stk, squared=False):.3f}, R²={r2_score(y_test, pred_stk):.3f}")
    except Exception as e:
        print("Stacking skipped:", e)
"""

    # ==== Keras (unchanged but a bit larger nets) ====
    keras_code_cls = """# ==== 4. Keras (Classification, optional) ====
if HAS_TF:
    input_dim = prep.fit_transform(X_train).shape[1]
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(len(np.unique(y_train)), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    Xtr = prep.transform(X_train)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    ytr = le.fit_transform(y_train)
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_split=0.2, epochs=80, batch_size=64, callbacks=[es], verbose=0)
    pd.DataFrame(hist.history)[["accuracy","val_accuracy"]].plot(figsize=(7,4), title="Keras training/validation accuracy"); plt.show()
"""

    keras_code_reg = """# ==== 4. Keras (Regression, optional) ====
if HAS_TF:
    input_dim = prep.fit_transform(X_train).shape[1]
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=[keras.metrics.RootMeanSquaredError()])
    Xtr = prep.transform(X_train)
    es = keras.callbacks.EarlyStopping(monitor="val_root_mean_squared_error", patience=6, restore_best_weights=True)
    hist = model.fit(Xtr, y_train, validation_split=0.2, epochs=80, batch_size=64, callbacks=[es], verbose=0)
    pd.DataFrame(hist.history)[["root_mean_squared_error","val_root_mean_squared_error"]].plot(figsize=(7,4), title="Keras training/validation RMSE"); plt.show()
"""

    # ==== Time series baseline (unchanged) ====
    ts_code = """# ==== Time-Series Baselines (optional) ====
if TIME_COL and TIME_COL in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
    ts = df[[TIME_COL, TARGET]].dropna().copy()
    ts[TIME_COL] = pd.to_datetime(ts[TIME_COL], errors="coerce")
    ts = ts.dropna(subset=[TIME_COL]).sort_values(TIME_COL).set_index(TIME_COL)
    try:
        inf_freq = pd.infer_freq(ts.index[:10])
        if inf_freq: ts = ts.asfreq(inf_freq)
    except Exception:
        pass
    split_idx = int(len(ts)*0.8)
    train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]
    naive = train[TARGET].iloc[-1]; pred_naive = pd.Series(naive, index=test.index)
    w = min(7, max(2, int(len(train)/20))); ma = train[TARGET].rolling(w).mean().iloc[-1]
    pred_ma = pd.Series(ma, index=test.index)
    def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
    def mape(a,b): return float(np.mean(np.abs((a-b)/np.clip(np.abs(a),1e-8,None)))*100)
    for name, pred in [("Naive", pred_naive), ("MovingAverage", pred_ma)]:
        print(f"{name} RMSE={rmse(test[TARGET], pred):.3f} MAPE={mape(test[TARGET], pred):.2f}%")
    ax = ts[TARGET].plot(figsize=(10,4), label="Actual"); pred_naive.plot(ax=ax, label="Naive"); pred_ma.plot(ax=ax, label="MA")
    plt.title(f"TS Baselines for {TARGET}"); plt.legend(); plt.tight_layout(); plt.show()
"""

    # ==== Interpretability hooks ====
    interpret_md = (
        "## 6. Interpretability & Diagnostics\n"
        "- Permutation importance • PDP/ICE • (Optional) SHAP/LIME\n"
    )

    lime_code = """# ==== LIME (optional) ====
if HAS_LIME and TARGET:
    try:
        transformed = prep.fit_transform(X_train)
        class_names = np.unique(y_train).astype(str) if not pd.api.types.is_numeric_dtype(y_train) else None
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(transformed.toarray() if hasattr(transformed, "toarray") else transformed),
            feature_names=None, class_names=class_names, discretize_continuous=True, mode="classification" if not pd.api.types.is_numeric_dtype(y_train) else "regression"
        )
        sample_idx = np.random.randint(0, X_test.shape[0])
        sample_row = X_test.iloc[[sample_idx]]
        exp = explainer.explain_instance(
            data_row=prep.transform(sample_row)[0],
            predict_fn=top_pipe.predict_proba if hasattr(top_pipe, "predict_proba") else top_pipe.predict,
            num_features=10
        )
        from IPython.display import display as ipydisplay
        ipydisplay(exp.show_in_notebook(show_table=True))
    except Exception as e:
        print("LIME explanation failed:", e)
"""

    shap_code = """# ==== SHAP (optional) ====
if USE_SHAP and HAS_SHAP and TARGET:
    try:
        shap.random.seed(SEED)
        top_pipe.fit(X_train, y_train)
        try:
            model = top_pipe.named_steps.get("model", None)
        except Exception:
            model = None
        Xs = prep.fit_transform(X_train)
        sample_idx = np.random.choice(Xs.shape[0], size=min(200, Xs.shape[0]), replace=False)
        Xs_sample = Xs[sample_idx]
        if model is not None and hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(lambda v: top_pipe.predict_proba(v) if hasattr(top_pipe, "predict_proba") else top_pipe.predict(v), masker=shap.maskers.Independent(Xs_sample))
        shap_values = explainer(Xs_sample)
        shap.plots.beeswarm(shap_values, max_display=15, show=True)
    except Exception as e:
        print("SHAP skipped:", e)
"""

    # ==== Champion export + Model Card ====
    export_code = """# ==== 7. Champion Export & Model Card ====
try:
    import joblib, datetime
    champion = top_pipe  # change if you prefer stacking
    joblib.dump(champion, "champion_pipeline.joblib")
    schema = {"target": TARGET, "columns": list(X_train.columns)}
    with open("schema.json","w") as f:
        json.dump(schema, f, indent=2)
    card = f\"\"\"# Model Card
- Export time: {pd.Timestamp.now()}
- Task: {'Classification' if not pd.api.types.is_numeric_dtype(y_train) else 'Regression'}
- CV folds: {CV_FOLDS}
- Preprocessing: winsor={OUTLIER_CLIP_PCT}, power={POWER_TRANSFORM}, scaling={SCALING}, miss_ind={ADD_MISSING_INDICATORS}
- Feature selection: {FEAT_SEL_METHOD} (k={FEAT_SEL_K})
- Calibration: {CALIBRATE_BINARY}
- Ensemble/Stacking: {ENABLE_ENSEMBLE}/{USE_STACKING}
- Metrics summary: see comparison tables in this notebook
- Data MD5: {md5_file(DATA_PATH)}
- Notes: Check drift & fairness slices before deployment.
\"\"\"
    with open("model_card.md","w") as f:
        f.write(card)
    print("Saved champion_pipeline.joblib, schema.json, model_card.md")
except Exception as e:
    print("Export skipped:", e)
"""

    conclusion_md = (
        "## 8. Final Discussion & Recommendations\n"
        "- Strengths and limitations of the EDA & models.\n"
        "- Risk/assumption log.\n"
        "- Clear, prioritized actions and expected impact.\n"
    )

    cells = [
        _nb_cell_markdown(header),
        _nb_cell_markdown(prob_stmt),
        _nb_cell_markdown(data_overview),
        _nb_cell_code(param_code),
        _nb_cell_code(imports_code),
        _nb_cell_code(load_code),
        _nb_cell_code(eda_code),
        _nb_cell_code(leakage_code),
        _nb_cell_code(split_code_cls if is_class else split_code_reg),
        _nb_cell_code(feat_sel_block),
        _nb_cell_code(baselines_block_cls if is_class else baselines_block_reg),
    ]

    if enable_keras:
        cells.append(_nb_cell_markdown("## 4. Keras (optional)"))
        cells.append(_nb_cell_code(keras_code_cls if is_class else keras_code_reg))

    if enable_timeseries:
        cells.append(_nb_cell_code(ts_code))

    cells += [
        _nb_cell_markdown("## 5. Model Evaluation and Assessment"),
        _nb_cell_code(model_code_cls if is_class else model_code_reg),
        _nb_cell_markdown(bq_md),
        _nb_cell_markdown(interpret_md),
        _nb_cell_code(lime_code),
        _nb_cell_code(shap_code),
        _nb_cell_code(export_code),
        _nb_cell_markdown(conclusion_md),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"file_extension": ".py", "mimetype": "text/x-python", "name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, ensure_ascii=False, indent=2).encode("utf-8")


# =======================================
# Helpers: Colab bundle + file utilities
# =======================================
def _save_uploaded_file(file) -> str:
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.name)
    with open(path, "wb") as f:
        f.write(file.getbuffer())
    return os.path.abspath(path)


def _requirements_txt() -> str:
    return "\n".join([
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "# Optional extras (uncomment if you need them below)",
        "# xgboost",
        "# lightgbm",
        "# catboost",
        "# category_encoders",
        "# tensorflow",
        "# lime",
        "# shap",
        "# imbalanced-learn",
        "# optuna",
        "# scipy",
    ]) + "\n"


def _make_colab_bundle(ipynb_bytes: bytes, ipynb_name: str, dataset_path: Optional[str]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(ipynb_name, ipynb_bytes)
        if dataset_path and os.path.exists(dataset_path):
            z.write(dataset_path, arcname=os.path.basename(dataset_path))
        z.writestr("requirements.txt", _requirements_txt())
        readme = f"""# Colab Bundle

This ZIP contains your notebook and (optionally) the dataset.
Steps:
1) Go to https://colab.research.google.com/
2) New Notebook → File → Upload Notebook → select `{ipynb_name}`
3) If your dataset file is included, also upload it via the Colab left sidebar (Files) and ensure `DATA_PATH` matches the filename.
4) (Optional) Install extras via: `!pip -q install -r requirements.txt` (uncomment packages as needed).
"""
        z.writestr("README.txt", readme)
    mem.seek(0)
    return mem.read()


# =======================================
# Quick Compare (sampled) + metric target
# =======================================
@dataclass
class QuickCompareResult:
    mode: str
    df: pd.DataFrame
    error: Optional[str]


def _run_quick_compare(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    scaling: str,
    sample_cap: int,
    metric_target: str,
    repeats: int,
    group_col: Optional[str],
) -> QuickCompareResult:
    from sklearn.model_selection import (StratifiedKFold, KFold, cross_val_score,
                                         RepeatedKFold, RepeatedStratifiedKFold, GroupKFold, StratifiedGroupKFold)
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

    try:
        from xgboost import XGBClassifier, XGBRegressor  # type: ignore
        HAS_XGB = True
    except Exception:
        HAS_XGB = False
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
        HAS_LGBM = True
    except Exception:
        HAS_LGBM = False
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
        HAS_CAT = True
    except Exception:
        HAS_CAT = False

    if target not in df.columns:
        return QuickCompareResult(mode=problem_type, df=pd.DataFrame(), error=f"Target column '{target}' not found.")
    if df.shape[0] < 50:
        return QuickCompareResult(mode=problem_type, df=pd.DataFrame(), error="Dataset too small (<50 rows) for quick compare.")

    df_s = df.sample(min(sample_cap, len(df)), random_state=RANDOM_SEED) if len(df) > sample_cap else df.copy()

    X = df_s.drop(columns=[target])
    y = df_s[target]

    if problem_type == "Auto":
        problem_type = "regression" if pd.api.types.is_numeric_dtype(y) else "classification"

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaling == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    num_pipe = Pipeline(steps=num_steps)
    try:
        cat_enc = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)
    except TypeError:
        cat_enc = OneHotEncoder(handle_unknown="ignore")
    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                              ("oh", cat_enc)])
    prep = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
    )

    results: List[Tuple] = []

    if problem_type == "classification":
        y = y.astype(str)
        class_pct = y.value_counts(normalize=True)
        IMBALANCED = class_pct.max() > 0.8
        models: List[Tuple[str, object]] = [
            ("LogReg", LogisticRegression(max_iter=400, class_weight='balanced' if IMBALANCED else None)),
            ("RF", RandomForestClassifier(n_estimators=400, random_state=RANDOM_SEED, n_jobs=-1,
                                          class_weight='balanced' if IMBALANCED else None)),
            ("GBC", GradientBoostingClassifier(random_state=RANDOM_SEED)),
        ]
        if HAS_XGB: models.append(("XGB", XGBClassifier(n_estimators=500, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED, tree_method="hist", n_jobs=-1, eval_metric="logloss")))
        if HAS_LGBM: models.append(("LGBM", LGBMClassifier(n_estimators=600, num_leaves=64, random_state=RANDOM_SEED)))
        if HAS_CAT: models.append(("CAT", CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=False, random_seed=RANDOM_SEED)))

        # CV scheme
        if group_col and group_col in X.columns:
            try:
                cv = StratifiedGroupKFold(n_splits=3)
            except Exception:
                cv = GroupKFold(n_splits=3)
        else:
            try:
                cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=max(1,repeats), random_state=RANDOM_SEED)
            except Exception:
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

        # scorer
        mt = (metric_target or "").upper()
        if mt in ["AUC","ROC_AUC","ROC-AUC"]:
            scoring = "roc_auc_ovr"
        elif mt in ["PR_AUC","AUPRC","PR-AUC"]:
            scoring = "average_precision"
        elif mt in ["ACC","ACCURACY"]:
            scoring = "accuracy"
        elif mt in ["F1","F1_WEIGHTED"]:
            scoring = "f1_weighted"
        else:
            scoring = "roc_auc_ovr"

        for name, est in models:
            pipe = Pipeline(steps=[("prep", prep), ("model", est)])
            try:
                score = cross_val_score(pipe, X, y, cv=cv, scoring=scoring).mean()
            except Exception:
                score = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted").mean()
            results.append((name, scoring, score))

        res = pd.DataFrame(results, columns=["Model", "Metric", "Score"]).set_index("Model").sort_values("Score", ascending=False)
        return QuickCompareResult(mode="classification", df=res, error=None)

    else:
        models = [
            ("LinReg", LinearRegression()),
            ("RF", RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1)),
            ("GBR", GradientBoostingRegressor(random_state=RANDOM_SEED)),
        ]
        if HAS_XGB: models.append(("XGB", XGBRegressor(n_estimators=700, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED, tree_method="hist", n_jobs=-1)))
        if HAS_LGBM: models.append(("LGBM", LGBMRegressor(n_estimators=800, num_leaves=64, random_state=RANDOM_SEED)))
        if HAS_CAT: models.append(("CAT", CatBoostRegressor(iterations=700, depth=6, learning_rate=0.1, verbose=False, random_seed=RANDOM_SEED)))

        try:
            cv = RepeatedKFold(n_splits=3, n_repeats=max(1,repeats), random_state=RANDOM_SEED)
        except Exception:
            cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

        mt = (metric_target or "").upper()
        if mt == "MAE":
            scoring = "neg_mean_absolute_error"
        elif mt == "R2":
            scoring = "r2"
        elif mt == "MAPE":
            scoring = "neg_mean_absolute_percentage_error"
        else:
            scoring = "neg_root_mean_squared_error"

        for name, est in models:
            pipe = Pipeline(steps=[("prep", prep), ("model", est)])
            score = cross_val_score(pipe, X, y, cv=cv, scoring=scoring).mean()
            results.append((name, scoring, score))
        res = pd.DataFrame(results, columns=["Model", "Metric", "Score"]).set_index("Model").sort_values("Score", ascending=(scoring!="r2"))
        return QuickCompareResult(mode="regression", df=res, error=None)


# ==============
# Auto-Tune (Optuna/RandomizedSearch)
# ==============
def _tune_best_model_optuna(df: pd.DataFrame, target: str, problem_type: str, scaling: str, best_model_name: str, n_trials: int = 20):
    try:
        import optuna # type: ignore
    except Exception:
        return None, "Optuna not installed."
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    X = df.drop(columns=[target]); y = df[target]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaling == "standard": num_steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax": num_steps.append(("scaler", MinMaxScaler()))
    try:
        cat_enc = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)
    except TypeError:
        cat_enc = OneHotEncoder(handle_unknown="ignore")
    prep = ColumnTransformer([("num", Pipeline(num_steps), num_cols), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("oh", cat_enc)]), cat_cols)])

    def build_clf(trial):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        try:
            from xgboost import XGBClassifier # type: ignore
            HAS_XGB = True
        except Exception:
            HAS_XGB = False
        try:
            from lightgbm import LGBMClassifier # type: ignore
            HAS_LGBM = True
        except Exception:
            HAS_LGBM = False

        if best_model_name == "LogReg":
            C = trial.suggest_float("C", 1e-3, 1e2, log=True)
            return LogisticRegression(max_iter=600, C=C)
        if best_model_name == "RF":
            n = trial.suggest_int("n_estimators", 200, 800)
            md = trial.suggest_int("max_depth", 4, 20)
            return RandomForestClassifier(n_estimators=n, max_depth=md, n_jobs=-1, random_state=RANDOM_SEED)
        if best_model_name == "GBC":
            n = trial.suggest_int("n_estimators", 200, 800)
            lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            md = trial.suggest_int("max_depth", 2, 5)
            return GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=md, random_state=RANDOM_SEED)
        if best_model_name == "XGB" and HAS_XGB:
            n = trial.suggest_int("n_estimators", 300, 900)
            md = trial.suggest_int("max_depth", 3, 8)
            ss = trial.suggest_float("subsample", 0.6, 1.0)
            cs = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            return XGBClassifier(n_estimators=n, max_depth=md, subsample=ss, colsample_bytree=cs, n_jobs=-1, tree_method="hist", random_state=RANDOM_SEED, eval_metric="logloss")
        if best_model_name == "LGBM" and HAS_LGBM:
            n = trial.suggest_int("n_estimators", 300, 1200)
            nl = trial.suggest_int("num_leaves", 31, 128)
            lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            return LGBMClassifier(n_estimators=n, num_leaves=nl, learning_rate=lr, random_state=RANDOM_SEED)
        return None

    def build_reg(trial):
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        try:
            from xgboost import XGBRegressor # type: ignore
            HAS_XGB = True
        except Exception:
            HAS_XGB = False
        try:
            from lightgbm import LGBMRegressor # type: ignore
            HAS_LGBM = True
        except Exception:
            HAS_LGBM = False

        if best_model_name == "LinReg":
            return LinearRegression()
        if best_model_name == "RF":
            n = trial.suggest_int("n_estimators", 300, 900)
            md = trial.suggest_int("max_depth", 4, 20)
            return RandomForestRegressor(n_estimators=n, max_depth=md, n_jobs=-1, random_state=RANDOM_SEED)
        if best_model_name == "GBR":
            n = trial.suggest_int("n_estimators", 300, 900)
            lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            md = trial.suggest_int("max_depth", 2, 5)
            return GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=md, random_state=RANDOM_SEED)
        if best_model_name == "XGB" and HAS_XGB:
            n = trial.suggest_int("n_estimators", 300, 1000)
            md = trial.suggest_int("max_depth", 3, 8)
            ss = trial.suggest_float("subsample", 0.6, 1.0)
            cs = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            return XGBRegressor(n_estimators=n, max_depth=md, subsample=ss, colsample_bytree=cs, n_jobs=-1, tree_method="hist", random_state=RANDOM_SEED)
        if best_model_name == "LGBM" and HAS_LGBM:
            n = trial.suggest_int("n_estimators", 300, 1200)
            nl = trial.suggest_int("num_leaves", 31, 128)
            lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            return LGBMRegressor(n_estimators=n, num_leaves=nl, learning_rate=lr, random_state=RANDOM_SEED)
        return None

    if pd.api.types.is_numeric_dtype(y):
        def objective(trial):
            est = build_reg(trial)
            if est is None: raise optuna.TrialPruned()
            pipe = Pipeline([("prep", prep), ("model", est)])
            cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            score = cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error").mean()
            return score
    else:
        y = y.astype(str)
        def objective(trial):
            est = build_clf(trial)
            if est is None: raise optuna.TrialPruned()
            pipe = Pipeline([("prep", prep), ("model", est)])
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            score = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc_ovr").mean()
            return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study, None


# ===========================
# Public Streamlit entrypoint
# ===========================
def render_ai_analysing(T=lambda k: k):
    st.markdown("""
    <style>
      @keyframes pulseBadge {0%{transform:scale(1)}50%{transform:scale(1.03)}100%{transform:scale(1)}}
      .ai-hero{background:linear-gradient(135deg, rgba(59,130,246,.10), rgba(234,179,8,.08));border:1px solid rgba(255,255,255,.15);border-radius:16px;padding:16px;margin-bottom:14px}
      .ai-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.16);font-weight:700;animation:pulseBadge 3s infinite}
      .pill{display:inline-block;padding:6px 14px;border-radius:999px;margin:3px 6px 6px 0;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);font-weight:700;font-size:.85rem}
      .note{font-size:0.9rem;opacity:0.9}
      .divider{height:1px;background:rgba(255,255,255,.14);margin:12px 0;border-radius:1px}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### " + (T("AI ANALYSING") if callable(T) else "AI Analysing"))
    st.markdown(
        """
        <div class="ai-hero">
          <span class="ai-badge">🧠 AI Analysing — optional for data science users</span>
          <div style="margin-top:8px;opacity:.95">
            Builds a complete EDA ➜ modeling pipeline & Notebook from your brief and dataset.
          </div>
          <div style="margin-top:10px">
            <b>Highlights:</b> LGBM/CatBoost, stacking, repeated CV, group/time-aware CV, leakage-safe target encoding,
            power/quantile transforms, cost-based thresholds, calibration, permutation importance, PDP/ICE, drift checks,
            champion export, model card, and more. All extras are toggleable and auto-guarded.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    enable = st.checkbox(
        "Enable AI Analysing (advanced)",
        value=not DEFAULT_ADVANCED_OFF,
        help="Turn this ON to reveal advanced data science tooling. When OFF, the tab stays dormant and won’t run anything.",
    )
    if not enable:
        st.info("This advanced tab is disabled. Turn it on with the checkbox above whenever you need it.")
        return

    # --------------------
    # Inputs
    # --------------------
    brief = st.text_area(
        "Assessment Details / Brief",
        height=200,
        placeholder=("Paste a long paragraph with business context, dataset info, and goals — e.g., "
                     "'You are a data science consultant… build an explanatory data analysis pipeline…'"),
        key="ai_brief",
        help="Used to generate the executive summary, EDA plan, and business questions.",
    )

    uploaded = st.file_uploader(
        "Optional: upload dataset (CSV)",
        type=["csv"],
        key="ai_dataset",
        help="Drag & drop or browse a CSV file. Columns are inspected to suggest a target, time/group columns, and suitable models.",
    )
    dataset_path = None
    dataset_name = None
    df: Optional[pd.DataFrame] = None

    if uploaded:
        dataset_path = _save_uploaded_file(uploaded)
        dataset_name = os.path.basename(dataset_path)
        try:
            df = pd.read_csv(dataset_path)
            st.success(f"Loaded `{dataset_name}` with shape {df.shape}.")
            with st.expander("Preview data"):
                st.dataframe(df.head(MAX_PREVIEW_ROWS))
            with st.expander("Quick EDA"):
                st.write("Dtypes")
                st.write(df.dtypes.to_frame("dtype"))
                st.write("Missingness (%)")
                st.write((df.isna().mean()*100).round(2).sort_values(ascending=False).to_frame("missing_%").head(20))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    suggestions = _suggest_targets(df) if df is not None else []
    auto_time = _auto_time_column(df) if df is not None else None

    row1c1, row1c2, row1c3 = st.columns([0.36, 0.28, 0.36])
    with row1c1:
        nb_title = st.text_input(
            "Notebook title",
            value="Client_Analysis_Notebook",
            key="ai_nb_title",
            help="This becomes the Jupyter Notebook filename and H1 title.",
        )
    with row1c2:
        force_type = st.selectbox(
            "Problem type",
            ["Auto", "classification", "regression", "unsupervised"],
            key="ai_prob_type",
            help="Pick your task, or leave on Auto. Auto infers from the brief and target column dtype.",
        )
    with row1c3:
        n_q = st.slider(
            "Number of business questions",
            5, 10, 8, 1,
            key="ai_nq",
            help="How many business questions to scaffold in the notebook’s explanatory analysis section.",
        )

    # Column & preprocessing
    with st.expander("Optional: columns, imputation, scaling & transforms"):
        c1, c2, c3 = st.columns(3)
        with c1:
            target_col = st.selectbox(
                "Target column (for supervised)",
                options=(["(none)"] + (suggestions if suggestions else (list(df.columns) if df is not None else []))),
                index=(1 if suggestions else 0),
                key="ai_target_sel",
                help="Choose the prediction target. Leave empty for unsupervised analysis.",
            )
            if target_col == "(none)":
                target_col = None
            target_col = st.text_input(
                "Or type target column manually",
                value=(target_col or ""),
                key="ai_target_manual",
                help="Override the dropdown by typing an exact column name present in your dataset.",
            ).strip() or target_col
            id_cols = st.multiselect(
                "ID columns to drop",
                options=(list(df.columns) if df is not None else []),
                help="Columns that uniquely identify rows (e.g., customer_id, tx_id). They won’t be used as features.",
            )
        with c2:
            include_cols = st.multiselect(
                "Restrict analysis to columns (optional)",
                options=(list(df.columns) if df is not None else []),
                help="If you select columns here, only these (plus TARGET) will be analyzed. Leave empty to use all.",
            )
            normalization = st.selectbox(
                "Normalization",
                ["standard", "minmax"],
                index=0,
                help="Scaling method for numeric features.",
            )
            power_transform = st.selectbox(
                "Power/Quantile transform",
                ["none","yeojohnson","quantile"],
                help="Stabilize variance & reduce skew (applied after impute, before scaling).",
            )
        with c3:
            num_impute = st.selectbox(
                "Numeric imputation",
                ["median", "mean"],
                help="Strategy for filling missing numeric values.",
            )
            cat_impute = st.selectbox(
                "Categorical imputation",
                ["most_frequent", "constant"],
                help="Strategy for filling missing categorical values.",
            )
            cat_fill_value = st.text_input(
                "Fill value (if constant)",
                value="",
                help="Used only when categorical imputation is 'constant'.",
            )
    with st.expander("Optional: outliers, indicators & correlation"):
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            outlier_clip_pct = st.slider(
                "Outlier winsorization %",
                0.0, 0.05, 0.0, 0.01,
                help="Clip numeric features at lower/upper percentiles before scaling. 0 disables.",
            )
        with oc2:
            add_missing_indicators = st.checkbox(
                "Add missing-value indicators",
                value=True,
                help="Adds _ismissing flags for features with NaNs.",
            )
        with oc3:
            corr_prune_threshold = st.slider(
                "Correlation prune threshold",
                0.0, 0.99, 0.0, 0.01,
                help="Drop one of each highly correlated numeric pair above this threshold (0 disables).",
            )

    with st.expander("Optional: CV, metrics, groups & ensembles"):
        cv1, cv2, cv3 = st.columns(3)
        with cv1:
            cv_folds = st.slider(
                "CV folds (Notebook)",
                3, 10, 5,
                help="Cross-validation folds in the exported notebook.",
            )
            repeats = st.slider(
                "Repeated CV (Quick Compare)",
                1, 5, 1, 1,
                help="Number of repeats for repeated CV in Quick Compare.",
            )
        with cv2:
            group_col = st.selectbox(
                "Group column (optional)",
                options=(["(none)"] + (list(df.columns) if df is not None else [])),
                help="Use group-aware CV if you have multiple rows per entity. Prevents leakage across groups.",
            )
            if group_col == "(none)":
                group_col = None
            time_col = st.selectbox(
                "Time column (optional)",
                options=(["(auto)"] + (list(df.columns) if df is not None else [])),
                index=0,
                help="Pick a timestamp column for time-aware split; (auto) tries to detect one.",
            )
            if time_col == "(auto)":
                time_col = auto_time
        with cv3:
            metric_target = st.selectbox(
                "Optimize metric (Quick Compare & Notebook)",
                ["auto","AUC","PR_AUC","F1","ACC","RMSE","MAE","R2","MAPE"],
                help="What to optimize in leaderboards / CV.",
            )
            calibrate_binary = st.checkbox(
                "Calibrate probabilities (binary)",
                value=True,
                help="Improves probability calibration using CalibratedClassifierCV.",
            )
            enable_ensemble = st.checkbox(
                "Enable model ensemble/stacking",
                value=True,
                help="Soft-vote/blend top models and optional stacking.",
            )
            use_stacking = st.checkbox(
                "Use Stacking meta-learner",
                value=True,
                help="Add StackingClassifier/Regressor on top of base learners.",
            )

    with st.expander("Optional: feature selection, models & extras"):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            feat_sel_method = st.selectbox(
                "Feature selection method",
                ["none", "variance", "kbest"],
                help="Reduce dimensionality to improve generalization. KBest uses ANOVA/F-stat.",
            )
            feat_sel_k = st.slider(
                "K for SelectKBest",
                10, 1000, 100, 10,
                help="How many features to keep when using SelectKBest.",
            )
        with cc2:
            models_cls = st.multiselect(
                "Classification models",
                ["LogReg","RF","GBC","XGB","LGBM","CAT"],
                default=["LogReg","RF","GBC","XGB","LGBM","CAT"],
                help="Choose classifiers to include in Notebook.",
            )
            models_reg = st.multiselect(
                "Regression models",
                ["LinReg","RF","GBR","XGB","LGBM","CAT"],
                default=["LinReg","RF","GBR","XGB","LGBM","CAT"],
                help="Choose regressors to include in Notebook.",
            )
        with cc3:
            enable_keras = st.checkbox(
                "Include Keras model in Notebook",
                value=False,
                help="Adds a TensorFlow/Keras model cell (skips if TF isn’t available).",
            )
            use_smote = st.checkbox(
                "Use SMOTE (Notebook, if imblearn available)",
                value=False,
                help="For imbalanced classification, add SMOTE/SMOTENC in the Notebook pipelines.",
            )
            use_shap = st.checkbox(
                "Use SHAP (Notebook, if shap available)",
                value=False,
                help="Add SHAP explanations in the Notebook.",
            )

    # Quick Compare / tuning config
    rowQC1, rowQC2, rowQC3 = st.columns([0.32, 0.32, 0.36])
    with rowQC1:
        qc_sample_cap = st.slider(
            "Quick Compare sample cap",
            min_value=300, max_value=10000, value=QUICK_COMPARE_SAMPLE_MAX, step=100,
            help="Caps rows used during in-app Quick Compare to keep it fast.",
        )
    with rowQC2:
        auto_preview = st.checkbox(
            "Auto-preview from inputs (requires brief + dataset)",
            value=AUTO_PREVIEW_DEFAULT,
            help="Show a live plan preview after you paste the brief and upload a dataset.",
        )
    with rowQC3:
        tune_iter = st.slider(
            "AutoTune++ trials (Optuna)",
            min_value=5, max_value=60, value=20, step=5,
            help="Trials for Optuna tuning (if installed).",
        )

    # Actions
    colA, colB, colC, colD, colE = st.columns([0.24, 0.22, 0.22, 0.16, 0.16])
    gen_clicked = colA.button(
        "Generate Plan & Notebook",
        use_container_width=True,
        help="Creates the full plan and a ready-to-run Jupyter Notebook.",
    )
    qc_clicked = colB.button(
        "Run Quick Compare (sampled)",
        use_container_width=True,
        help="Lightweight, sampled CV comparison to preview promising models.",
    )
    tune_clicked = colC.button(
        "AutoTune++ best (Optuna)",
        use_container_width=True,
        help="Hyperparameter search with Optuna on the Quick Compare winner (if Optuna installed).",
    )
    show_anom = colD.checkbox(
        "Anomaly scan",
        value=False,
        help="Scan numeric columns for anomalies via IsolationForest.",
    )
    show_pii = colE.checkbox(
        "PII scan",
        value=False,
        help="Heuristic scan for email/phone-like patterns in object columns.",
    )

    # Session cache
    for k, v in [
        ("ai_last_nb", None),
        ("ai_last_spec", None),
        ("ai_last_df", None),
        ("ai_last_dataset_path", None),
        ("ai_qc_result", None),
        ("ai_qc_best", None),
        ("ai_holdout", None),
    ]:
        if k not in st.session_state: st.session_state[k] = v

    # Preview
    if auto_preview and brief.strip() and df is not None:
        spec_preview = _ai_plan_from_brief(brief, df, target_col, n_questions=n_q)
        if force_type != "Auto":
            spec_preview["problem_type"] = force_type

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Preview from your inputs (not exported yet)")
        st.markdown("#### Executive Summary")
        for b in spec_preview["summary"]:
            st.markdown(f"- {b}")
        st.markdown("#### Problem Type")
        st.markdown(f"<span class='pill'>🎯 {spec_preview['problem_type'].title()}</span>", unsafe_allow_html=True)

        cAprev, cBprev = st.columns(2)
        with cAprev:
            st.markdown("#### EDA Plan")
            for e in spec_preview["eda_plan"]:
                st.markdown(f"- {e}")
        with cBprev:
            st.markdown("#### Metrics")
            st.markdown(" ".join([f"<span class='pill'>{m}</span>" for m in spec_preview["metrics"]]), unsafe_allow_html=True)

        st.markdown("#### Business Questions")
        for i, q in enumerate(spec_preview["business_questions"], 1):
            st.markdown(f"**Q{i}.** {q}")

        st.markdown("#### Candidate Models")
        st.markdown(" ".join([f"<span class='pill'>🤖 {m}</span>" for m in spec_preview["models"]]), unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    else:
        st.caption("Tip: enable **Auto-preview from inputs** to see a live plan after you paste your brief and upload a dataset. "
                   "Otherwise, click **Generate Plan & Notebook** to produce the final plan and notebook.")
    # Generate Notebook
    if gen_clicked:
        if not brief.strip():
            st.warning("Please paste your brief first.")
        elif df is None:
            st.warning("Please upload a dataset to generate the pipeline notebook.")
        else:
            with st.spinner("Analysing your brief and generating the pipeline…"):
                spec2 = _ai_plan_from_brief(brief, df, target_col, n_questions=n_q)
                if force_type != "Auto":
                    spec2["problem_type"] = force_type

                default_data_path = dataset_path if dataset_path else "data/dataset.csv"
                nb_bytes = _build_notebook(
                    spec=spec2,
                    notebook_title=nb_title,
                    dataset_name=dataset_name,
                    default_data_path=default_data_path,
                    target_col=target_col,
                    normalization=normalization,
                    enable_keras=enable_keras,
                    enable_timeseries=True,
                    time_col_hint=time_col,
                    id_cols=id_cols,
                    include_cols=include_cols,
                    num_impute=num_impute,
                    cat_impute=cat_impute,
                    cat_fill_value=cat_fill_value,
                    cv_folds=cv_folds,
                    use_smote=use_smote,
                    use_shap=use_shap,
                    selected_models_cls=models_cls,
                    selected_models_reg=models_reg,
                    outlier_clip_pct=outlier_clip_pct,
                    feat_sel_method=feat_sel_method,
                    feat_sel_k=feat_sel_k,
                    calibrate_binary=calibrate_binary,
                    enable_ensemble=enable_ensemble,
                    power_transform=power_transform,
                    add_missing_indicators=add_missing_indicators,
                    corr_prune_threshold=corr_prune_threshold,
                    metric_target=metric_target,
                    group_col_hint=group_col,
                    use_stacking=use_stacking,
                )
                st.session_state["ai_last_nb"] = (nb_bytes, f"{nb_title}.ipynb")
                st.session_state["ai_last_spec"] = {
                    **spec2,
                    "params": dict(
                        target_col=target_col,
                        normalization=normalization,
                        id_cols=id_cols,
                        include_cols=include_cols,
                        num_impute=num_impute,
                        cat_impute=cat_impute,
                        cat_fill_value=cat_fill_value,
                        cv_folds=cv_folds,
                        use_smote=use_smote,
                        use_shap=use_shap,
                        models_cls=models_cls,
                        models_reg=models_reg,
                        time_col=time_col,
                        outlier_clip_pct=outlier_clip_pct,
                        feat_sel_method=feat_sel_method,
                        feat_sel_k=feat_sel_k,
                        calibrate_binary=calibrate_binary,
                        enable_ensemble=enable_ensemble,
                        power_transform=power_transform,
                        add_missing_indicators=add_missing_indicators,
                        corr_prune_threshold=corr_prune_threshold,
                        metric_target=metric_target,
                        group_col=group_col,
                        use_stacking=use_stacking,
                    ),
                }
                st.session_state["ai_last_df"] = df
                st.session_state["ai_last_dataset_path"] = dataset_path

            st.success("Plan generated! Preview & downloads below.")

            with st.container():
                st.markdown("<div class='ai-hero'>", unsafe_allow_html=True)
                st.markdown("#### Executive Summary")
                for b in spec2["summary"]:
                    st.markdown(f"- {b}")
                st.markdown("#### Problem Type")
                st.markdown(f"<span class='pill'>🎯 {spec2['problem_type'].title()}</span>", unsafe_allow_html=True)

                cA2, cB2 = st.columns(2)
                with cA2:
                    st.markdown("#### EDA Plan")
                    for e in spec2["eda_plan"]:
                        st.markdown(f"- {e}")
                with cB2:
                    st.markdown("#### Metrics")
                    st.markdown(" ".join([f"<span class='pill'>{m}</span>" for m in spec2["metrics"]]), unsafe_allow_html=True)

                st.markdown("#### Business Questions")
                for i, q in enumerate(spec2["business_questions"], 1):
                    st.markdown(f"**Q{i}.** {q}")

                st.markdown("#### Candidate Models")
                st.markdown(" ".join([f"<span class='pill'>🤖 {m}</span>" for m in spec2["models"]]), unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            cdl1, cdl2, cdl3, cdl4 = st.columns(4)
            with cdl1:
                st.download_button(
                    "Download Notebook (.ipynb)",
                    st.session_state["ai_last_nb"][0],
                    file_name=st.session_state["ai_last_nb"][1],
                    mime="application/x-ipynb+json",
                    key="ai_nb_dl",
                    help="Save the generated Jupyter Notebook to your machine.",
                )
            md_report = (
                f"# Analysis Plan — {nb_title}\n\n"
                f"## Executive Summary\n" + "\n".join([f"- {b}" for b in spec2["summary"]]) + "\n\n"
                f"## Problem Type\n- **{spec2['problem_type'].title()}**\n\n"
                f"## EDA Plan\n" + "\n".join([f"- {e}" for e in spec2["eda_plan"]]) + "\n\n"
                f"## Business Questions\n" + "\n".join([f"1. {q}" for q in spec2["business_questions"]]) + "\n\n"
                f"## Metrics to Track\n" + "\n".join([f"- {m}" for m in spec2["metrics"]]) + "\n\n"
                f"## Candidate Models\n" + "\n".join([f"- {m}" for m in spec2["models"]]) + "\n"
            )
            with cdl2:
                st.download_button(
                    "Download Plan (.md)",
                    md_report.encode("utf-8"),
                    file_name=f"{nb_title}_plan.md",
                    mime="text/markdown",
                    key="ai_md_dl",
                    help="Save a Markdown copy of the generated analysis plan.",
                )
            with cdl3:
                if ENABLE_COLAB:
                    zip_bytes = _make_colab_bundle(st.session_state["ai_last_nb"][0], st.session_state["ai_last_nb"][1], dataset_path)
                    st.download_button(
                        "Download Colab Bundle (.zip)",
                        zip_bytes,
                        file_name=f"{nb_title}_colab_bundle.zip",
                        mime="application/zip",
                        key="ai_zip_dl",
                        help="ZIP with notebook, requirements.txt, and (optionally) your dataset.",
                    )
            with cdl4:
                spec_json = json.dumps(st.session_state["ai_last_spec"], indent=2).encode("utf-8")
                st.download_button(
                    "Download Spec (.json)",
                    spec_json,
                    file_name=f"{nb_title}_spec.json",
                    mime="application/json",
                    key="ai_spec_dl",
                    help="Save the JSON spec used to build your notebook.",
                )

            if ENABLE_COLAB:
                st.info(
                    "🔗 **Open in Colab**: Open Google Colab in a new tab, then **File → Upload notebook** and choose the `.ipynb` you downloaded."
                )
                st.caption("Privacy note: Colab usage means your notebook/data are processed by Google.")

    # Quick Compare
    if qc_clicked:
        if df is None:
            st.warning("Please upload a dataset first.")
        elif not target_col:
            st.warning("Please set the target column.")
        else:
            with st.spinner("Running quick compare on a sample…"):
                result = _run_quick_compare(
                    df=df,
                    target=target_col,
                    problem_type=force_type,
                    scaling=normalization,
                    sample_cap=qc_sample_cap,
                    metric_target=metric_target,
                    repeats=repeats,
                    group_col=group_col,
                )
            if result.error:
                st.error(result.error)
            elif result.df.empty:
                st.warning("Quick Compare returned no results (check your target/columns).")
            else:
                st.session_state["ai_qc_result"] = (result.mode, result.df)
                metric_name = result.df.iloc[0]["Metric"] if "Metric" in result.df.columns and len(result.df) else ("CV AUC" if result.mode=="classification" else "CV RMSE")
                st.success(f"Quick Compare complete — leaderboard metric ({metric_name}).")
                st.dataframe(result.df)
                try:
                    st.bar_chart(result.df["Score"])
                except Exception:
                    pass

                best_model = result.df.sort_values("Score", ascending=(result.mode=="regression")).index[0]
                st.session_state["ai_qc_best"] = best_model
                st.info(f"Best model (sampled): **{best_model}**")

                # Optional holdout quick eval + threshold slider (binary)
                if st.button("Run holdout evaluation (fast)", help="Train the winner on 80% and report metrics on 20% holdout."):
                    err = _holdout_evaluate(df, target_col, result.mode, best_model, normalization, time_col=None)
                    if err:
                        st.error(err)

    # AutoTune++ (Optuna)
    if tune_clicked:
        if df is None or not target_col:
            st.warning("Please upload a dataset and set a target first.")
        elif not st.session_state.get("ai_qc_best"):
            st.warning("Run Quick Compare first to pick a winner.")
        else:
            best = st.session_state["ai_qc_best"]
            mode = force_type
            if st.session_state.get("ai_qc_result"):
                mode = st.session_state["ai_qc_result"][0]
            with st.spinner(f"Optuna tuning {best}…"):
                study, err = _tune_best_model_optuna(df, target_col, mode, normalization, best, n_trials=tune_iter)
            if err:
                st.error(err)
            elif study is None:
                st.error("Tuning did not return a result.")
            else:
                st.success(f"Best params for {best}:")
                st.json(study.best_params)
                st.write("Best CV score:", study.best_value)
                st.download_button(
                    "Download tuned params (.json)",
                    json.dumps(study.best_params, indent=2).encode("utf-8"),
                    file_name=f"{best}_best_params_optuna.json",
                    mime="application/json",
                )

    # Optional anomaly scan
    if show_anom and df is not None:
        from sklearn.ensemble import IsolationForest
        with st.spinner("Scanning for anomalies on numeric subset…"):
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) == 0:
                st.warning("No numeric columns found for anomaly scan.")
            else:
                Xn = df[num_cols].copy().fillna(df[num_cols].median())
                iso = IsolationForest(random_state=RANDOM_SEED, contamination="auto")
                preds = iso.fit_predict(Xn); scores = iso.decision_function(Xn)
                df_anom = df.copy(); df_anom["_anomaly"] = (preds == -1).astype(int); df_anom["_score"] = scores
                rate = df_anom["_anomaly"].mean() * 100
                st.write(f"Estimated anomalies: **{rate:.2f}%**")
                with st.expander("Show top suspected anomalies"):
                    st.dataframe(df_anom.sort_values("_score").head(20))
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(6, 3)); ax.hist(scores, bins=30)
                    ax.set_title("Anomaly score distribution (IsolationForest)"); st.pyplot(fig)
                except Exception:
                    pass

    # Optional PII scan
    if show_pii and df is not None:
        patterns = {
            "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}",
            "phone": r"(\\+\\d{1,3}[- ]?)?\\d{3}[- ]?\\d{3}[- ]?\\d{4}",
        }
        hits = []
        for c in df.select_dtypes(include=["object"]).columns:
            s = df[c].dropna().astype(str).head(1000).str.cat(sep=" ")
            if re.search(patterns["email"], s): hits.append((c, "email-like"))
            if re.search(patterns["phone"], s): hits.append((c, "phone-like"))
        if hits:
            st.warning(f"Potential PII columns (heuristic): {hits}")
        else:
            st.info("No obvious PII patterns found (heuristic).")

    # Footer note
    st.markdown(
        "<div class='note'>"
        "✅ Notebook includes: problem statement, data dictionary, rich EDA, leakage check, preprocessing with winsorization, power/quantile transforms, "
        "missing indicators, rare-cats & leakage-safe target encoding, anomaly scan, baselines, repeated/group/time-aware CV, leaderboards with advanced models "
        "(LGBM/CatBoost if available), calibration (binary), cost-based thresholds, permutation importance, PDP/ICE, learning curves, drift checks, stacking, "
        "optional Keras/SMOTE/SHAP, champion export & model card, business Q&A scaffold, and conclusions."
        "</div>",
        unsafe_allow_html=True,
    )


# ===========================
# Holdout evaluate (used by QC)
# ===========================
def _holdout_evaluate(df: pd.DataFrame, target: str, problem_type: str, model_name: str, scaling: str, time_col: Optional[str]):
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, confusion_matrix,
                                 mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve, average_precision_score)

    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

    try:
        from xgboost import XGBClassifier, XGBRegressor # type: ignore
        HAS_XGB = True
    except Exception:
        HAS_XGB = False
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor # type: ignore
        HAS_LGBM = True
    except Exception:
        HAS_LGBM = False
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor # type: ignore
        HAS_CAT = True
    except Exception:
        HAS_CAT = False

    X = df.drop(columns=[target]); y = df[target]
    if time_col and time_col in df.columns:
        dft = df.dropna(subset=[time_col]).copy()
        dft[time_col] = pd.to_datetime(dft[time_col], errors="coerce")
        dft = dft.dropna(subset=[time_col]).sort_values(time_col)
        y = dft[target]; X = dft.drop(columns=[target])
        split_idx = int(len(dft)*0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=(y if not pd.api.types.is_numeric_dtype(y) else None))

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaling == "standard": num_steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax": num_steps.append(("scaler", MinMaxScaler()))
    try:
        cat_enc = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)
    except TypeError:
        cat_enc = OneHotEncoder(handle_unknown="ignore")

    prep = ColumnTransformer(
        transformers=[("num", Pipeline(num_steps), num_cols), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("oh", cat_enc)]), cat_cols)]
    )

    if problem_type == "classification":
        y = y.astype(str); y_train = y_train.astype(str); y_test = y_test.astype(str)
        if model_name == "LogReg":
            est = LogisticRegression(max_iter=400)
        elif model_name == "RF":
            est = RandomForestClassifier(n_estimators=400, random_state=RANDOM_SEED, n_jobs=-1)
        elif model_name == "GBC":
            est = GradientBoostingClassifier(random_state=RANDOM_SEED)
        elif model_name == "XGB" and HAS_XGB:
            est = XGBClassifier(n_estimators=500, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED, tree_method="hist", n_jobs=-1)
        elif model_name == "LGBM" and HAS_LGBM:
            est = LGBMClassifier(n_estimators=600, num_leaves=64, random_state=RANDOM_SEED)
        elif model_name == "CAT" and HAS_CAT:
            est = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=False, random_seed=RANDOM_SEED)
        else:
            return "Model not available."
        pipe = Pipeline([("prep", prep), ("model", est)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        try:
            y_proba = pipe.predict_proba(X_test)[:, -1]
            auc = roc_auc_score((y_test==y_test.unique()[-1]).astype(int), y_proba) if len(y_test.unique())==2 else np.nan
            pr_p, pr_r, thr = precision_recall_curve((y_test==y_test.unique()[-1]).astype(int), y_proba)
            ap = average_precision_score((y_test==y_test.unique()[-1]).astype(int), y_proba)
        except Exception:
            auc = np.nan; ap = np.nan; pr_p=None; pr_r=None
        st.write(f"**Holdout Accuracy:** {accuracy_score(y_test, y_pred):.3f} | **F1(w):** {f1_score(y_test, y_pred, average='weighted'):.3f} | **AUC:** {auc if not np.isnan(auc) else 'n/a'} | **PR-AUC:** {ap if not np.isnan(ap) else 'n/a'}")
        try:
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(4,3)); sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax); ax.set_title("Holdout Confusion Matrix"); st.pyplot(fig)
        except Exception:
            st.write("Confusion matrix:", confusion_matrix(y_test, y_pred))

        # Interactive threshold (binary)
        if len(np.unique(y_test))==2 and y_proba is not None:
            st.markdown("**Adjust decision threshold**")
            t = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01, help="Move the threshold to trade precision/recall or cost.")
            pred_t = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix((y_test==y_test.unique()[-1]).astype(int), pred_t).ravel()
            FP_COST = st.number_input("Cost of FP", min_value=0.0, value=1.0, step=0.1)
            FN_COST = st.number_input("Cost of FN", min_value=0.0, value=5.0, step=0.1)
            st.write(f"TP={tp}, FP={fp}, FN={fn}, TN={tn} | Cost={fp*FP_COST + fn*FN_COST:.1f}")
        return None

    else:
        if model_name == "LinReg":
            est = LinearRegression()
        elif model_name == "RF":
            est = RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1)
        elif model_name == "GBR":
            est = GradientBoostingRegressor(random_state=RANDOM_SEED)
        elif model_name == "XGB" and HAS_XGB:
            est = XGBRegressor(n_estimators=700, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED, tree_method="hist", n_jobs=-1)
        elif model_name == "LGBM" and HAS_LGBM:
            est = LGBMRegressor(n_estimators=800, num_leaves=64, random_state=RANDOM_SEED)
        elif model_name == "CAT" and HAS_CAT:
            est = CatBoostRegressor(iterations=700, depth=6, learning_rate=0.1, verbose=False, random_seed=RANDOM_SEED)
        else:
            return "Model not available."
        pipe = Pipeline([("prep", prep), ("model", est)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**Holdout RMSE:** {rmse:.3f} | **MAE:** {mae:.3f} | **R²:** {r2:.3f}")
        try:
            fig, ax = plt.subplots(figsize=(4,3))
            ax.scatter(y_test, y_pred, alpha=0.5)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims); ax.set_title("Holdout Residuals"); st.pyplot(fig)
        except Exception:
            pass
        return None
