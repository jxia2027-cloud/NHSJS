"""
nhsjs.py - NHSJS CKD Paper (Revised per Reviewer Comments)
==========================================================
Produces all analyses for:
  When Perfect Is Wrong: Exposing the AUC = 1.0 Illusion in
  Chronic Kidney Disease Machine Learning Benchmarks

REVISION (per reviewer comments):
  - Stratified subsampling in all learning curves (#7)
  - UCI uses one-hot for nominals + ordered for sg/al (#8)
  - 20 repetitions everywhere (#6)
  - Cross-dataset evaluation: UCI-trained models tested on NHANES/Tawam
    via common-feature subset (#2)
  - NHANES with 3 feature sets: all / no eGFR-ACR / no creatinine (#3)
  - UCI ablation: drop top-3 features and retrain (#24)
  - Power-law: full fit output, bootstrap CIs, sensitivity to early n (#5,#15)
  - Calibration: Brier score on NHANES and Tawam (#14)
  - Tawam train/test gap analysis for overfitting (#19)
  - Single-feature: logistic regression instead of HGB (#11)
  - KDIGO stage mapping table data (#20)

Datasets:
  UCI_kidney_disease400.csv
  CKD_NHANES_2021_2023_11933.csv
  tawam491.csv

Install: pip install pandas numpy scikit-learn scipy matplotlib

Configure N_REPS at the top of the script. Default is 20 per reviewer #6.
Use N_REPS = 5 for quick smoke tests.
"""
import warnings, json, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
    HistGradientBoostingClassifier, VotingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────
def resolve_path(*candidates):
    for rel in candidates:
        p = BASE_DIR / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not resolve any of: {candidates}")

UCI_PATH    = resolve_path("datasetsPythonScripts/UCI_kidney_disease400.csv",
                           "UCI_kidney_disease400.csv")
NHANES_PATH = resolve_path("datasetsPythonScripts/CKD_NHANES_2021_2023_11933.csv",
                           "CKD_NHANES_2021_2023_11933.csv")
TAWAM_PATH  = resolve_path("datasetsPythonScripts/tawam491.csv", "tawam491.csv")
OUT         = BASE_DIR / "outputs"
LATEX_DATA  = OUT / "latex_data"
OUT.mkdir(exist_ok=True)
LATEX_DATA.mkdir(exist_ok=True)

N_REPS    = int(os.getenv("N_REPS", "20"))     # Reviewer #6: increased from 5 to 20
N_REPS_NH = int(os.getenv("N_REPS_NH", "20"))  # Reviewer #6: increased from 3 to 20
N_REPS_TW = int(os.getenv("N_REPS_TW", "20"))
N_BOOT    = int(os.getenv("N_BOOT", "500"))    # Bootstrap iterations for power-law CIs

# ── Colour system ─────────────────────────────────────────────────
PAL = {
    "LogReg":  {"line":"#E74C3C","fill":"#FADBD8","marker":"o","label":"Logistic Regression"},
    "RF":      {"line":"#2980B9","fill":"#D6EAF8","marker":"s","label":"Random Forest"},
    "HGB":     {"line":"#27AE60","fill":"#D5F5E3","marker":"^","label":"Gradient Boosting"},
    "MLP":     {"line":"#8E44AD","fill":"#E8DAEF","marker":"D","label":"Neural Network (balanced)"},
    "Ensemble":{"line":"#D35400","fill":"#FAE5D3","marker":"P","label":"Ensemble"},
}
plt.rcParams.update({
    "font.family":"DejaVu Sans",
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.grid":True, "grid.alpha":0.25,
    "axes.labelsize":13, "axes.titlesize":13,
    "xtick.labelsize":11, "ytick.labelsize":11,
    "legend.fontsize":11, "figure.dpi":150,
})

# ══════════════════════════════════════════════════════════════════
# 1. DATA LOADING (REVISED per reviewer #8)
# ══════════════════════════════════════════════════════════════════

# UCI variable classifications (reviewer #8)
UCI_NUMERIC = ["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]
UCI_ORDINAL = ["sg","al","su"]    # Clinically ordered (dipstick scales, urine specific gravity)
UCI_NOMINAL = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]  # Nominal -> one-hot

# Ordinal mappings for UCI clinically-ordered variables
SG_MAP = {1.005:1, 1.010:2, 1.015:3, 1.020:4, 1.025:5}
AL_MAP = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
SU_MAP = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}

def clean_string(series):
    return (series.astype(str)
            .str.strip()
            .str.lower()
            .replace({"?": np.nan, "": np.nan, "nan": np.nan}))

def map_float_code(val, mapping):
    if pd.isna(val):
        return np.nan
    try:
        return mapping.get(round(float(val), 3), np.nan)
    except Exception:
        return np.nan

def map_int_code(val, mapping):
    if pd.isna(val):
        return np.nan
    try:
        return mapping.get(int(round(float(val))), np.nan)
    except Exception:
        return np.nan

def load_uci(path):
    """Loads UCI CKD with proper encoding per reviewer #8:
    one-hot for nominal variables; explicit ordered codes for sg/al/su."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    df["label"] = df["classification"].astype(str).str.strip().str.lower().apply(
        lambda x: 1 if x == "ckd" else 0)

    # Clean strings
    for c in UCI_ORDINAL + UCI_NOMINAL:
        if c in df.columns:
            df[c] = clean_string(df[c])

    # Ordered encoding for clinically ordered variables
    if "sg" in df.columns:
        df["sg"] = df["sg"].apply(lambda v: map_float_code(v, SG_MAP))
    if "al" in df.columns:
        df["al"] = df["al"].apply(lambda v: map_int_code(v, AL_MAP))
    if "su" in df.columns:
        df["su"] = df["su"].apply(lambda v: map_int_code(v, SU_MAP))

    # Numeric coercion only. Imputation now happens after the train/test split.
    for c in UCI_NUMERIC + UCI_ORDINAL:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    nominal_cols = [c for c in UCI_NOMINAL if c in df.columns]
    numeric_cols = [c for c in UCI_NUMERIC + UCI_ORDINAL if c in df.columns]
    keep_cols = numeric_cols + nominal_cols + ["label"]
    return df[keep_cols].copy(), {
        "numeric": numeric_cols,
        "nominal": nominal_cols,
        "all": numeric_cols + nominal_cols,
    }

def load_nhanes(path, feature_set="all"):
    """Loads NHANES with optional feature set restriction per reviewer #3.

    feature_set: 'all'        - all available predictors
                 'no_egfr_acr' - removes eGFR and ACR (direct label inputs)
                 'no_lab_surrogates' - also removes serum_creatinine (eGFR input)
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns={"bp_systolic":"bp_sys","bp_diastolic":"bp_dia",
        "blood_urea_nitrogen":"bun","albumin_serum":"albumin",
        "egfr":"gfr","diabetes_diagnosed":"diabetes",
        "albumin_creatinine_ratio":"acr","ckd_present":"label"})
    if "hypertension" not in df.columns:
        df["hypertension"] = (df.get("bp_sys", 130) >= 130).astype(float)

    full_feats = ["age","bp_sys","bp_dia","serum_creatinine","bun",
                  "albumin","bmi","gfr","acr","diabetes","hypertension"]
    if feature_set == "no_egfr_acr":
        # Remove direct label-defining variables
        full_feats = [f for f in full_feats if f not in ("gfr","acr")]
    elif feature_set == "no_lab_surrogates":
        # Also remove serum_creatinine (used to compute eGFR)
        full_feats = [f for f in full_feats
                      if f not in ("gfr","acr","serum_creatinine","bun")]

    avail = [f for f in full_feats if f in df.columns]
    df2 = df[avail + ["label"]].copy()
    for c in avail:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2.dropna(subset=["label"]).assign(
        label=lambda d: d["label"].astype(int)), {
            "numeric": avail, "nominal": [], "all": avail
        }

def load_tawam(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"AgeBaseline":"age","sBPBaseline":"bp_sys",
        "dBPBaseline":"bp_dia","CreatnineBaseline":"creatinine",
        "eGFRBaseline":"gfr","BMIBaseline":"bmi",
        "HistoryDiabetes":"diabetes","HistoryHTN ":"hypertension",
        "CholesterolBaseline":"cholesterol",
        "TriglyceridesBaseline":"triglycerides",
        "HgbA1C":"hba1c","EventCKD35":"label"})
    feats = [f for f in ["age","bp_sys","bp_dia","creatinine","gfr","bmi",
                         "diabetes","hypertension","cholesterol",
                         "triglycerides","hba1c"]
             if f in df.columns]
    df2 = df[feats + ["label"]].copy()
    for c in feats:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2.assign(label=lambda d: d["label"].astype(int)), {
        "numeric": feats, "nominal": [], "all": feats
    }

def make_preprocessor(feature_spec):
    numeric_cols = feature_spec["numeric"]
    nominal_cols = feature_spec["nominal"]
    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if nominal_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            nominal_cols,
        ))
    return ColumnTransformer(transformers)

def get_feature_names(preprocessor, feature_spec):
    names = []
    if feature_spec["numeric"]:
        names.extend(feature_spec["numeric"])
    if feature_spec["nominal"]:
        oh = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        names.extend(list(oh.get_feature_names_out(feature_spec["nominal"])))
    return names

print("Loading data...")
df_uci, uci_spec = load_uci(UCI_PATH)
df_nh,  nh_spec  = load_nhanes(NHANES_PATH, feature_set="all")
df_tw,  tw_spec  = load_tawam(TAWAM_PATH)
uci_raw_feats = uci_spec["all"]
nh_feats = nh_spec["all"]
tw_feats = tw_spec["all"]
print(f"  UCI    n={len(df_uci)} CKD={df_uci['label'].mean():.1%} feats={len(uci_raw_feats)}")
print(f"  NHANES n={len(df_nh)}  CKD={df_nh['label'].mean():.1%} feats={len(nh_feats)}")
print(f"  Tawam  n={len(df_tw)}  event={df_tw['label'].mean():.1%} feats={len(tw_feats)}")

# ══════════════════════════════════════════════════════════════════
# 2. PREPROCESSING (stratified per reviewer #7)
# ══════════════════════════════════════════════════════════════════
def split_scale(df, feature_spec, test_size=0.25, seed=42):
    """Stratified train/test split with train-only preprocessing."""
    feats = feature_spec["all"]
    X = df[feats].copy()
    y = df["label"].values.astype(int)
    Xtr_df, Xte_df, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    pre = make_preprocessor(feature_spec)
    Xtr = pre.fit_transform(Xtr_df)
    Xte = pre.transform(Xte_df)
    return Xtr, Xte, ytr, yte, pre, get_feature_names(pre, feature_spec), Xtr_df, Xte_df

Xuci_tr, Xuci_te, yuci_tr, yuci_te, uci_pre, uci_feats, _, _ = split_scale(df_uci, uci_spec)
Xnh_tr,  Xnh_te,  ynh_tr,  ynh_te,  nh_pre, _, _, _ = split_scale(df_nh,  nh_spec, test_size=0.15)
Xtw_tr,  Xtw_te,  ytw_tr,  ytw_te,  tw_pre, _, _, _ = split_scale(df_tw,  tw_spec)

def audit_dataset(name, df, feature_spec, y_train, y_test, duplicate_ids=0):
    y_all = df["label"].values.astype(int)
    predictors = feature_spec["all"]
    return {
        "dataset": name,
        "n": int(len(df)),
        "positive_n": int(np.sum(y_all == 1)),
        "positive_rate": float(np.mean(y_all)),
        "predictor_n": int(len(predictors)),
        "missing_predictor_cells": int(df[predictors].isna().sum().sum()),
        "train_n": int(len(y_train)),
        "train_positive_rate": float(np.mean(y_train)),
        "test_n": int(len(y_test)),
        "test_positive_rate": float(np.mean(y_test)),
        "duplicate_ids": int(duplicate_ids),
    }

tw_raw = pd.read_csv(TAWAM_PATH)
tawam_duplicate_ids = int(tw_raw["StudyID"].duplicated().sum()) if "StudyID" in tw_raw else 0
dataset_audit = [
    audit_dataset("UCI CKD", df_uci, uci_spec, yuci_tr, yuci_te),
    audit_dataset("NHANES 2021-2023", df_nh, nh_spec, ynh_tr, ynh_te),
    audit_dataset("Tawam UAE", df_tw, tw_spec, ytw_tr, ytw_te, tawam_duplicate_ids),
]

# ══════════════════════════════════════════════════════════════════
# 3. MODEL FACTORIES
# ══════════════════════════════════════════════════════════════════
MAKERS = {
    "LogReg":   lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "RF":       lambda: RandomForestClassifier(200, max_depth=8, min_samples_leaf=5,
                                               random_state=42, n_jobs=1),
    "HGB":      lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                       max_depth=5, random_state=42),
    "MLP":      lambda: MLPClassifier((128,64,32), activation="relu", solver="adam",
                                      max_iter=500, early_stopping=True,
                                      validation_fraction=0.1, random_state=42),
    "Ensemble": lambda: VotingClassifier([
        ("hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                               max_depth=5, random_state=42)),
        ("rf",  RandomForestClassifier(200, max_depth=8, min_samples_leaf=5,
                                       random_state=42, n_jobs=1)),
        ("mlp", MLPClassifier((128,64,32), activation="relu", solver="adam",
                              max_iter=500, early_stopping=True,
                              validation_fraction=0.1, random_state=42))],
        voting="soft", n_jobs=1),
}
MODEL_NAMES = list(MAKERS.keys())

def rebalance_binary_training_data(X, y, seed=42):
    """Simple random oversampling for minority class when sample_weight is unavailable."""
    y = np.asarray(y)
    cls, counts = np.unique(y, return_counts=True)
    if len(cls) < 2 or counts[0] == counts[1]:
        return X, y
    maj = cls[np.argmax(counts)]
    min_ = cls[np.argmin(counts)]
    maj_idx = np.where(y == maj)[0]
    min_idx = np.where(y == min_)[0]
    rng = np.random.RandomState(seed)
    extra_idx = rng.choice(min_idx, size=len(maj_idx) - len(min_idx), replace=True)
    all_idx = np.concatenate([maj_idx, min_idx, extra_idx])
    rng.shuffle(all_idx)
    return X[all_idx], y[all_idx]

def fit_model(model, Xtr, ytr, use_balanced_mlp=False):
    if use_balanced_mlp and isinstance(model, MLPClassifier):
        try:
            sw = compute_sample_weight("balanced", ytr)
            model.fit(Xtr, ytr, sample_weight=sw)
            return model
        except TypeError:
            X_bal, y_bal = rebalance_binary_training_data(Xtr, ytr)
            model.fit(X_bal, y_bal)
            return model
    model.fit(Xtr, ytr)
    return model

def fit_auc(maker, Xtr, ytr, Xte, yte, use_balanced_mlp=False):
    m = maker()
    m = fit_model(m, Xtr, ytr, use_balanced_mlp=use_balanced_mlp)
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), m, p

def fit_auc_train(maker, Xtr, ytr, Xte, yte, use_balanced_mlp=False):
    m = maker()
    m = fit_model(m, Xtr, ytr, use_balanced_mlp=use_balanced_mlp)
    return (roc_auc_score(ytr, m.predict_proba(Xtr)[:,1]),
            roc_auc_score(yte, m.predict_proba(Xte)[:,1]), m)

def threshold_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "threshold": threshold,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def stratified_subsample(y, size, seed):
    """Stratified subsample respecting class balance (reviewer #7)."""
    rng = np.random.RandomState(seed)
    cls0 = np.where(y == 0)[0]
    cls1 = np.where(y == 1)[0]
    p1 = y.mean()
    n1 = max(1, int(round(size * p1)))
    n0 = max(1, size - n1)
    n1 = min(n1, len(cls1))
    n0 = min(n0, len(cls0))
    idx = np.concatenate([rng.choice(cls0, n0, replace=False),
                          rng.choice(cls1, n1, replace=False)])
    rng.shuffle(idx)
    return idx

# ══════════════════════════════════════════════════════════════════
# 4. POWER-LAW FITTING (per reviewer #5, #15)
# ══════════════════════════════════════════════════════════════════
def power_law(n, ceil, a, b):
    """AUC(n) = ceiling - a * n^(-b), with ceiling bounded <= 1.0"""
    return ceil - a * np.array(n, dtype=float) ** (-b)

def fit_power(sizes, aucs):
    """Returns dict with parameters, CIs (bootstrap), R^2, predictions."""
    sizes = np.array(sizes); aucs = np.array(aucs)
    mask = ~np.isnan(aucs)
    sizes, aucs = sizes[mask], aucs[mask]
    if len(sizes) < 3:
        return None
    try:
        max_auc = float(np.max(aucs))
        lower_ceiling = min(max_auc, 0.999999)
        popt, pcov = curve_fit(
            power_law, sizes, aucs,
            p0=[min(1.0, max_auc + 0.01), 2.0, 0.5],
            bounds=([lower_ceiling, 0, 0], [1.0, 200, 5]),
            maxfev=8000)
        pred = power_law(sizes, *popt)
        ss_res = np.sum((aucs - pred) ** 2)
        ss_tot = np.sum((aucs - np.mean(aucs)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Bootstrap CIs (reviewer #5)
        boot = []
        rng = np.random.RandomState(0)
        for _ in range(N_BOOT):
            idx = rng.choice(len(sizes), len(sizes), replace=True)
            try:
                p, _ = curve_fit(power_law, sizes[idx], aucs[idx],
                                 p0=popt,
                                 bounds=([lower_ceiling,0,0],[1.0,200,5]),
                                 maxfev=2000)
                boot.append(p)
            except Exception:
                continue
        boot = np.array(boot) if boot else np.zeros((0,3))
        ci = np.percentile(boot, [2.5, 97.5], axis=0) if len(boot) > 10 else None

        # Sensitivity: refit dropping smallest 2 points
        sens = None
        if len(sizes) >= 5:
            try:
                idx = np.argsort(sizes)[2:]
                p2, _ = curve_fit(power_law, sizes[idx], aucs[idx],
                                  p0=popt,
                                  bounds=([lower_ceiling,0,0],[1.0,200,5]),
                                  maxfev=2000)
                sens = {"ceiling": float(p2[0]), "a": float(p2[1]), "b": float(p2[2])}
            except Exception:
                pass

        return {"ceiling": float(popt[0]), "a": float(popt[1]), "b": float(popt[2]),
                "ci": ci.tolist() if ci is not None else None,
                "r2": float(r2),
                "sensitivity_drop2_lowest": sens,
                "n_points": int(len(sizes))}
    except Exception as e:
        return None

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1 - Full model comparison (all datasets)
# ══════════════════════════════════════════════════════════════════
print("\n[Exp1] Full model x dataset comparison...")
full_results = {}
calib_results = {}
threshold_results = {}
for name, maker in MAKERS.items():
    a_uci, _, _    = fit_auc(maker, Xuci_tr, yuci_tr, Xuci_te, yuci_te)
    a_nh,  _, p_nh = fit_auc(maker, Xnh_tr,  ynh_tr,  Xnh_te,  ynh_te)
    a_tw,  _, p_tw = fit_auc(maker, Xtw_tr,  ytw_tr,  Xtw_te,  ytw_te,
                             use_balanced_mlp=True)
    full_results[name] = {"UCI": a_uci, "NHANES": a_nh, "Tawam": a_tw}
    # Reviewer #14: calibration via Brier score on real-difficulty datasets
    calib_results[name] = {
        "NHANES_brier": float(brier_score_loss(ynh_te, p_nh)),
        "Tawam_brier":  float(brier_score_loss(ytw_te, p_tw)),
    }
    threshold_results[name] = {
        "NHANES": threshold_metrics(ynh_te, p_nh),
        "Tawam": threshold_metrics(ytw_te, p_tw),
    }
    print(f"  {name}: UCI={a_uci:.4f} NH={a_nh:.4f} TW={a_tw:.4f} "
          f"BrierNH={calib_results[name]['NHANES_brier']:.4f} "
          f"BrierTW={calib_results[name]['Tawam_brier']:.4f}")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2 - UCI learning curves (stratified, all algorithms)
# ══════════════════════════════════════════════════════════════════
print(f"\n[Exp2] UCI learning curves (N_REPS={N_REPS}, stratified)...")
UCI_SIZES = [10,20,40,50,75,100,150,200,len(yuci_tr)]
uci_curves = {}
for name, maker in MAKERS.items():
    test_means, test_lo, test_hi, train_means = [], [], [], []
    for size in UCI_SIZES:
        t_aucs, tr_aucs = [], []
        for seed in range(N_REPS):
            idx = stratified_subsample(yuci_tr, size, seed + 100)
            if len(np.unique(yuci_tr[idx])) < 2:
                continue
            try:
                tr_a, te_a, _ = fit_auc_train(
                    maker, Xuci_tr[idx], yuci_tr[idx], Xuci_te, yuci_te)
                t_aucs.append(te_a); tr_aucs.append(tr_a)
            except Exception:
                pass
        m  = float(np.nanmean(t_aucs)) if t_aucs else float("nan")
        sd = float(np.nanstd(t_aucs))  if t_aucs else 0.0
        n  = len(t_aucs)
        test_means.append(m)
        test_lo.append(m - 1.96 * sd / np.sqrt(max(n, 1)))
        test_hi.append(m + 1.96 * sd / np.sqrt(max(n, 1)))
        train_means.append(float(np.nanmean(tr_aucs)) if tr_aucs else float("nan"))
    uci_curves[name] = {"test":test_means, "lo":test_lo, "hi":test_hi,
                        "train":train_means}
    print(f"  {name}: n=40->{test_means[2]:.3f}, n=75->{test_means[4]:.3f}")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3 - NHANES learning curves WITH 3 FEATURE SETS (#3)
# ══════════════════════════════════════════════════════════════════
print(f"\n[Exp3] NHANES learning curves (3 feature sets, N_REPS={N_REPS_NH})...")
NH_SIZES = [50,100,200,500,1000,2000,5000,len(ynh_tr)]
NH_FEATURE_SETS = ["all", "no_egfr_acr", "no_lab_surrogates"]
FS_DISPLAY = {
    "all": "All features",
    "no_egfr_acr": "No eGFR/ACR",
    "no_lab_surrogates": "No eGFR/ACR/Cr/BUN",
}

nh_curves_all_sets = {}     # feature_set -> model -> curve data
nh_plaw_all_sets   = {}
nh_full_aucs       = {}

for fs in NH_FEATURE_SETS:
    print(f"\n  --- NHANES feature set: '{fs}' ---")
    df_nh_fs, nh_fs_spec = load_nhanes(NHANES_PATH, feature_set=fs)
    Xfs_tr, Xfs_te, yfs_tr, yfs_te, _, nh_fs_feats, _, _ = split_scale(
        df_nh_fs, nh_fs_spec, test_size=0.15)
    print(f"  Features ({len(nh_fs_feats)}): {nh_fs_feats}")

    nh_curves_all_sets[fs] = {}
    nh_plaw_all_sets[fs]   = {}
    nh_full_aucs[fs]       = {}

    for name, maker in MAKERS.items():
        # Full-data AUC for this feature set
        a, _, _ = fit_auc(maker, Xfs_tr, yfs_tr, Xfs_te, yfs_te)
        nh_full_aucs[fs][name] = float(a)

        # Learning curve
        test_means, test_lo, test_hi = [], [], []
        sizes_fs = [s for s in NH_SIZES if s <= len(yfs_tr)] + \
                   ([len(yfs_tr)] if NH_SIZES[-1] != len(yfs_tr) else [])
        sizes_fs = sorted(set(sizes_fs))
        for size in sizes_fs:
            aucs = []
            for seed in range(N_REPS_NH):
                idx = stratified_subsample(yfs_tr, size, seed)
                try:
                    a_, _, _ = fit_auc(maker, Xfs_tr[idx], yfs_tr[idx],
                                       Xfs_te, yfs_te)
                    aucs.append(a_)
                except Exception:
                    pass
            m  = float(np.nanmean(aucs)) if aucs else float("nan")
            sd = float(np.nanstd(aucs))  if aucs else 0.0
            n  = len(aucs)
            test_means.append(m)
            test_lo.append(m - 1.96 * sd / np.sqrt(max(n, 1)))
            test_hi.append(m + 1.96 * sd / np.sqrt(max(n, 1)))
        nh_curves_all_sets[fs][name] = {
            "sizes": sizes_fs, "test": test_means,
            "lo": test_lo, "hi": test_hi
        }
        # Power law per feature set
        nh_plaw_all_sets[fs][name] = fit_power(sizes_fs, test_means)
        ceil_str = f"{nh_plaw_all_sets[fs][name]['ceiling']:.3f}" \
                   if nh_plaw_all_sets[fs][name] else "—"
        print(f"    {name}: full={a:.3f}, ceiling={ceil_str}")

# Maintain backward compatibility: "all" set is the primary
nh_curves = nh_curves_all_sets["all"]
nh_plaw   = nh_plaw_all_sets["all"]

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4 - Tawam learning curves (with train/test gap, #19)
# ══════════════════════════════════════════════════════════════════
print(f"\n[Exp4] Tawam learning curves (N_REPS={N_REPS_TW}, train+test for overfitting)...")
TW_SIZES = [20,30,40,50,75,100,125,150,200,len(ytw_tr)]
tw_curves = {}
for name, maker in MAKERS.items():
    test_means, test_lo, test_hi, train_means = [], [], [], []
    for size in TW_SIZES:
        t_aucs, tr_aucs = [], []
        for seed in range(N_REPS_TW):
            idx = stratified_subsample(ytw_tr, size, seed + 200)
            if len(np.unique(ytw_tr[idx])) < 2:
                continue
            try:
                tr_a, te_a, _ = fit_auc_train(maker, Xtw_tr[idx], ytw_tr[idx],
                                              Xtw_te, ytw_te, use_balanced_mlp=True)
                t_aucs.append(te_a); tr_aucs.append(tr_a)
            except Exception:
                pass
        m  = float(np.nanmean(t_aucs)) if t_aucs else float("nan")
        sd = float(np.nanstd(t_aucs))  if t_aucs else 0.0
        n  = len(t_aucs)
        test_means.append(m)
        test_lo.append(m - 1.96 * sd / np.sqrt(max(n, 1)))
        test_hi.append(m + 1.96 * sd / np.sqrt(max(n, 1)))
        train_means.append(float(np.nanmean(tr_aucs)) if tr_aucs else float("nan"))
    tw_curves[name] = {"test":test_means, "lo":test_lo, "hi":test_hi,
                       "train":train_means}
    gap = (train_means[-1] - test_means[-1]) if not np.isnan(train_means[-1]) else float("nan")
    print(f"  {name}: n=50->{test_means[3]:.3f}, full->{test_means[-1]:.3f}, train-test gap={gap:.3f}")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 5 - Single-feature AUC (REVISED per #11: logistic regression)
# ══════════════════════════════════════════════════════════════════
print("\n[Exp5] Single-feature AUC on UCI (Logistic Regression per reviewer #11)...")
# For single-feature analysis we use the original (pre-one-hot) numeric/ordinal
# columns since one-hot expands binary columns
SF_FEATS_DISPLAY = {
    "hemo":"Hemoglobin","sg":"Specific Gravity","pcv":"Packed Cell Volume",
    "sc":"Serum Creatinine","al":"Albumin","rc":"Red Blood Cells",
    "sod":"Sodium","pot":"Potassium",
    "bgr":"Blood Glucose","age":"Age","bp":"Blood Pressure",
    "wc":"White Blood Cells","su":"Sugar Level",
    "bu":"Blood Urea"
}
SF_FEATS = [f for f in SF_FEATS_DISPLAY if f in uci_feats]
SF_FEATS = [f for f in SF_FEATS_DISPLAY if f in uci_spec["numeric"]]

single_feat = {}
for feat in SF_FEATS:
    X_f = df_uci[[feat]].copy()
    y_f = df_uci["label"].values.astype(int)
    Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
        X_f, y_f, test_size=0.25, stratify=y_f, random_state=42)
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]).fit(Xf_tr)
    try:
        m = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        m.fit(pre.transform(Xf_tr), yf_tr)
        single_feat[feat] = float(roc_auc_score(
            yf_te, m.predict_proba(pre.transform(Xf_te))[:,1]))
    except Exception:
        single_feat[feat] = float("nan")
    print(f"  {feat:6s}: {single_feat[feat]:.4f}")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 6 - UCI ABLATION (reviewer #24)
#    Remove top-3 single-feature AUC features and retrain
# ══════════════════════════════════════════════════════════════════
print("\n[Exp6] UCI ablation - drop top-3 features...")
sorted_sf = sorted(single_feat.items(), key=lambda kv: -kv[1])
top3 = [s[0] for s in sorted_sf[:3]]
print(f"  Top-3 single-feature predictors: {top3}")
uci_spec_ablated = {
    "numeric": [f for f in uci_spec["numeric"] if f not in top3],
    "nominal": list(uci_spec["nominal"]),
}
uci_spec_ablated["all"] = uci_spec_ablated["numeric"] + uci_spec_ablated["nominal"]
print(f"  Original n_features={len(uci_spec['all'])}, "
      f"ablated n_features={len(uci_spec_ablated['all'])}")

Xabl_tr, Xabl_te, yabl_tr, yabl_te, _, uci_feats_ablated, _, _ = split_scale(
    df_uci, uci_spec_ablated)

ablation_results = {}
for name, maker in MAKERS.items():
    a, _, _ = fit_auc(maker, Xabl_tr, yabl_tr, Xabl_te, yabl_te)
    ablation_results[name] = float(a)
    print(f"  {name} ablated: {a:.4f} (original: {full_results[name]['UCI']:.4f})")

# Ablation learning curve (HGB only) — show whether the ceiling persists
print("\n  Ablation learning curve (Gradient Boosting)...")
ablation_curve = {"sizes": [], "test": []}
for size in UCI_SIZES:
    aucs = []
    for seed in range(N_REPS):
        idx = stratified_subsample(yabl_tr, size, seed + 300)
        if len(np.unique(yabl_tr[idx])) < 2: continue
        try:
            a_, _, _ = fit_auc(MAKERS["HGB"], Xabl_tr[idx], yabl_tr[idx],
                               Xabl_te, yabl_te)
            aucs.append(a_)
        except Exception: pass
    ablation_curve["sizes"].append(size)
    ablation_curve["test"].append(float(np.nanmean(aucs)) if aucs else float("nan"))

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 7 - CROSS-DATASET EVALUATION (reviewer #2)
#    Train on UCI, test on NHANES/Tawam via common-feature subset
# ══════════════════════════════════════════════════════════════════
print("\n[Exp7] Cross-dataset evaluation (UCI -> NHANES, UCI -> Tawam)...")

# Identify common features. UCI has age, bp, and serum creatinine
# (others are UCI-specific or missing from Tawam). NHANES has age,
# systolic blood pressure, and serum creatinine. Tawam has age,
# systolic blood pressure, and creatinine.
def common_feature_view(df_uci_raw, df_nh_raw, df_tw_raw):
    """Return (Xu, yu, Xn, yn, Xt, yt) using only common clinical features.
    Common: age, systolic BP, serum creatinine."""
    # Rebuild UCI from CSV with original column names (pre one-hot)
    df_u = pd.read_csv(UCI_PATH)
    df_u.columns = df_u.columns.str.lower().str.strip()
    df_u["label"] = df_u["classification"].astype(str).str.strip().str.lower().apply(
        lambda x: 1 if x == "ckd" else 0)
    common_u = {"age":"age", "bp":"bp_sys", "sc":"creatinine"}
    for k in common_u:
        df_u[k] = pd.to_numeric(df_u[k], errors="coerce")
    df_u_c = df_u[list(common_u.keys()) + ["label"]].copy()
    df_u_c = df_u_c.rename(columns=common_u)
    # NHANES
    df_n = pd.read_csv(NHANES_PATH).rename(columns={
        "bp_systolic":"bp_sys", "serum_creatinine":"creatinine",
        "ckd_present":"label"})
    df_n_c = df_n[["age","bp_sys","creatinine","label"]].copy()
    # Tawam
    df_t = pd.read_csv(TAWAM_PATH)
    df_t.columns = df_t.columns.str.strip()
    df_t = df_t.rename(columns={
        "AgeBaseline":"age","sBPBaseline":"bp_sys",
        "CreatnineBaseline":"creatinine","EventCKD35":"label"})
    df_t_c = df_t[["age","bp_sys","creatinine","label"]].copy()

    for df_x in (df_u_c, df_n_c, df_t_c):
        for c in ["age","bp_sys","creatinine"]:
            df_x[c] = pd.to_numeric(df_x[c], errors="coerce")

    # Drop rows with missing label
    df_u_c = df_u_c.dropna(subset=["label"])
    df_n_c = df_n_c.dropna(subset=["label"])
    df_t_c = df_t_c.dropna(subset=["label"])

    return df_u_c, df_n_c, df_t_c

df_u_cc, df_n_cc, df_t_cc = common_feature_view(df_uci, df_nh, df_tw)
COMMON_FEATS = ["age","bp_sys","creatinine"]
print(f"  Common features: {COMMON_FEATS}")

# Train UCI on common feats; scale fit on UCI training portion
Xuc = df_u_cc[COMMON_FEATS].copy()
yuc = df_u_cc["label"].values.astype(int)
Xuc_tr_df, Xuc_te_df, yuc_tr, yuc_te = train_test_split(
    Xuc, yuc, test_size=0.25, stratify=yuc, random_state=42)
cc_pre = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
]).fit(Xuc_tr_df)
Xuc_tr = cc_pre.transform(Xuc_tr_df)
Xuc_te = cc_pre.transform(Xuc_te_df)

# Apply same scaler to NHANES + Tawam common-feature representations
Xnc = cc_pre.transform(df_n_cc[COMMON_FEATS].copy())
ync = df_n_cc["label"].values.astype(int)
Xtc = cc_pre.transform(df_t_cc[COMMON_FEATS].copy())
ytc = df_t_cc["label"].values.astype(int)

cross_results = {}
for name, maker in MAKERS.items():
    m = maker()
    m.fit(Xuc_tr, yuc_tr)
    # In-sample UCI test (sanity)
    a_uci_self = roc_auc_score(yuc_te, m.predict_proba(Xuc_te)[:,1])
    # UCI -> NHANES (entire NHANES)
    a_uci_nh   = roc_auc_score(ync, m.predict_proba(Xnc)[:,1])
    # UCI -> Tawam (entire Tawam)
    a_uci_tw   = roc_auc_score(ytc, m.predict_proba(Xtc)[:,1])
    cross_results[name] = {
        "UCI_self_commonfeats": float(a_uci_self),
        "UCI_to_NHANES":        float(a_uci_nh),
        "UCI_to_Tawam":         float(a_uci_tw)
    }
    print(f"  {name}: UCI->UCI={a_uci_self:.3f}, UCI->NH={a_uci_nh:.3f}, UCI->TW={a_uci_tw:.3f}")

# ══════════════════════════════════════════════════════════════════
# 9. ROC CURVES (Tawam MLP balanced)
# ══════════════════════════════════════════════════════════════════
print("\n[Exp8] ROC curves...")
roc_uci = {}; roc_tw = {}
for name, maker in MAKERS.items():
    mu = maker(); mu.fit(Xuci_tr, yuci_tr)
    pu = mu.predict_proba(Xuci_te)[:,1]
    fpr, tpr, _ = roc_curve(yuci_te, pu)
    roc_uci[name] = {"fpr":fpr, "tpr":tpr,
                     "auc":float(roc_auc_score(yuci_te, pu))}
    mt = maker()
    mt = fit_model(mt, Xtw_tr, ytw_tr, use_balanced_mlp=True)
    pt = mt.predict_proba(Xtw_te)[:,1]
    fpr, tpr, _ = roc_curve(ytw_te, pt)
    roc_tw[name] = {"fpr":fpr, "tpr":tpr,
                    "auc":float(roc_auc_score(ytw_te, pt))}

# ══════════════════════════════════════════════════════════════════
# PRIMARY POWER-LAW FITS (for legacy figs)
# ══════════════════════════════════════════════════════════════════
print("\n[Power-law] Primary fits on NHANES 'all' feature set...")
for name in MODEL_NAMES:
    nh_plaw[name] = fit_power(NH_SIZES, nh_curves[name]["test"])

# ══════════════════════════════════════════════════════════════════
# HELPER: draw one learning-curve panel
# ══════════════════════════════════════════════════════════════════
def draw_lc(ax, sizes, curves, title, xlabel, ylim=(0.45,1.03),
            logx=False, show_train=False, ref_line=None, ref_label=None):
    for name in MODEL_NAMES:
        d = curves[name]; p = PAL[name]
        tm = list(d["test"]); lo = d["lo"]; hi = d["hi"]
        sz_v = sizes[:len(tm)] if isinstance(sizes, list) else d.get("sizes", sizes)
        sz_v = list(sz_v)[:len(tm)]
        ax.plot(sz_v, tm, color=p["line"], lw=2.2, marker=p["marker"],
                ms=6, label=p["label"], zorder=4)
        ax.fill_between(sz_v, [max(0.4,v) for v in lo],
                        [min(1.01,v) for v in hi],
                        alpha=0.13, color=p["line"])
        if show_train and "train" in d:
            tr = list(d["train"])[:len(sz_v)]
            ax.plot(sz_v, tr, color=p["line"], lw=1.2, ls="--",
                    alpha=0.55, zorder=3)
    if ref_line is not None:
        ax.axhline(ref_line, color="#C0392B", lw=1.5, ls=":", alpha=0.7,
                   label=ref_label or f"Reference = {ref_line:.3f}")
    if logx: ax.set_xscale("log")
    ax.set_ylim(ylim); ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("AUC-ROC", fontsize=11)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

# ══════════════════════════════════════════════════════════════════
# FIGURES (1-8, plus new figs 9 and 10 for ablation and feature-set comparison)
# (Figures retained from v3; new figures appended)
# ══════════════════════════════════════════════════════════════════

# FIGURE 1 - Model x dataset bar chart
print("\nFigure 1: Model comparison bar chart...")
fig, ax = plt.subplots(figsize=(12,7), facecolor="white")
names = MODEL_NAMES
x = np.arange(len(names)); w = 0.25
uci_v = [full_results[n]["UCI"] for n in names]
nh_v  = [full_results[n]["NHANES"] for n in names]
tw_v  = [full_results[n]["Tawam"] for n in names]
ax.bar(x-w, uci_v, w, label="UCI CKD",      color="#C0392B")
ax.bar(x,   nh_v,  w, label="NHANES 2021-23", color="#2471A3")
ax.bar(x+w, tw_v,  w, label="Tawam UAE",     color="#1E8449")
ax.set_xticks(x); ax.set_xticklabels([PAL[n]["label"] for n in names], rotation=20)
ax.set_ylabel("AUC-ROC"); ax.set_ylim(0,1.05)
ax.set_title("Figure 1. Model Performance Across Three CKD Datasets",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right"); plt.tight_layout()
plt.savefig(f"{OUT}/fig1_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 2 - UCI learning curves
print("Figure 2: UCI learning curves...")
fig, ax = plt.subplots(figsize=(12,7), facecolor="white")
draw_lc(ax, UCI_SIZES, uci_curves,
        f"Figure 2. UCI Learning Curves (N_REPS={N_REPS}, stratified subsampling)\n"
        "Solid = test AUC | Dashed = train AUC | Rapid saturation by n=40-60",
        "UCI Training Set Size", show_train=True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_uci_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 3 - NHANES learning curves (primary, all features)
print("Figure 3: NHANES learning curves...")
fig, ax = plt.subplots(figsize=(12,7), facecolor="white")
draw_lc(ax, NH_SIZES, nh_curves,
        f"Figure 3. NHANES Learning Curves (all features, N_REPS={N_REPS_NH})\n"
        "Genuine learning curve: AUC rises with training size",
        "NHANES Training Set Size", logx=True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_nhanes_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 4 - Tawam learning curves (with train AUC for overfitting)
print("Figure 4: Tawam learning curves...")
fig, ax = plt.subplots(figsize=(12,7), facecolor="white")
draw_lc(ax, TW_SIZES, tw_curves,
        f"Figure 4. Tawam Learning Curves (N_REPS={N_REPS_TW}, train/test gap shown)\n"
        "Dashed = train AUC | Solid = test AUC | Gap indicates overfitting risk",
        "Tawam Training Set Size", show_train=True)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_tawam_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 5 - Single-feature AUC (logistic regression now)
print("Figure 5: Single-feature AUC...")
sorted_sf = sorted(single_feat.items(), key=lambda kv: -kv[1])
fig, ax = plt.subplots(figsize=(12, max(5, 0.45*len(sorted_sf))), facecolor="white")
names_s = [SF_FEATS_DISPLAY.get(f, f) for f, _ in sorted_sf]
aucs_s  = [v for _, v in sorted_sf]
colors  = ["#C0392B" if v >= 0.90 else "#E67E22" if v >= 0.75 else "#2980B9" for v in aucs_s]
y_pos = np.arange(len(names_s))[::-1]
ax.barh(y_pos, aucs_s, color=colors, alpha=0.85)
for y, v in zip(y_pos, aucs_s):
    ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=10)
ax.axvline(0.90, color="#C0392B", lw=1.2, ls="--", alpha=0.6, label="AUC ≥ 0.90 (shortcut)")
ax.axvline(0.75, color="#E67E22", lw=1.2, ls="--", alpha=0.6, label="AUC ≥ 0.75 (informative)")
ax.axvline(0.50, color="#717D7E", lw=1.0, ls=":",  alpha=0.5, label="AUC = 0.50 (random)")
ax.set_yticks(y_pos); ax.set_yticklabels(names_s, fontsize=11)
ax.set_xlabel("Single-Feature AUC-ROC (Logistic Regression)", fontsize=13)
ax.set_xlim(0.40, 1.07)
ax.set_title("Figure 5. Individual Feature Discriminability — UCI CKD Dataset",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_single_feature_auc.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 6 - Common-feature cross-dataset transfer
print("Figure 6: Cross-dataset transfer...")
fig, ax = plt.subplots(figsize=(12,6), facecolor="white")
transfer_targets = ["UCI_self_commonfeats", "UCI_to_NHANES", "UCI_to_Tawam"]
transfer_labels = ["UCI -> UCI", "UCI -> NHANES", "UCI -> Tawam"]
x = np.arange(len(MODEL_NAMES)); w = 0.24
transfer_palette = ["#C0392B", "#2471A3", "#1E8449"]
for i, target in enumerate(transfer_targets):
    vals = [cross_results[n][target] for n in MODEL_NAMES]
    ax.bar(x + (i - 1) * w, vals, w, label=transfer_labels[i], color=transfer_palette[i])
ax.set_xticks(x)
ax.set_xticklabels([PAL[n]["label"] for n in MODEL_NAMES], rotation=20)
ax.set_ylabel("AUC-ROC")
ax.set_ylim(0, 1.05)
ax.set_title("Figure 6. UCI-Trained Models Transfer Poorly Across CKD Tasks",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_dataset_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 7 - ROC curves (UCI vs Tawam)
print("Figure 7: ROC curves...")
fig, axes = plt.subplots(1, 2, figsize=(14,6), facecolor="white")
for ax, title, roc_data in [
    (axes[0], "UCI CKD", roc_uci),
    (axes[1], "Tawam UAE", roc_tw),
]:
    for name in MODEL_NAMES:
        p = PAL[name]
        ax.plot(roc_data[name]["fpr"], roc_data[name]["tpr"],
                color=p["line"], lw=2.0,
                label=f"{p['label']} (AUC={roc_data[name]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#7F8C8D", lw=1.1, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
fig.suptitle("Figure 7. ROC Curves Contrast Ceiling-Bound and Real-Difficulty Tasks",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 8 - Power-law comparison across NHANES feature sets
print("Figure 8: Power-law comparison...")
fig, axes = plt.subplots(1, 3, figsize=(18,6), facecolor="white", sharey=True)
grid = np.geomspace(50, max(NH_SIZES), 200)
for ax, fs in zip(axes, NH_FEATURE_SETS):
    cur = nh_curves_all_sets[fs]["HGB"]
    ax.plot(cur["sizes"], cur["test"], marker="o", color=PAL["HGB"]["line"],
            lw=2.0, label="Observed HGB")
    fit = nh_plaw_all_sets[fs]["HGB"]
    if fit:
        ax.plot(grid, power_law(grid, fit["ceiling"], fit["a"], fit["b"]),
                color="#C0392B", lw=2.0, ls="--",
                label=f"Fit ceiling={fit['ceiling']:.3f}")
    ax.set_xscale("log")
    ax.set_ylim(0.55, 1.02)
    ax.set_xlabel("Training Size")
    ax.set_title(fs.replace("_", " "), fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
axes[0].set_ylabel("AUC-ROC")
fig.suptitle("Figure 8. NHANES Power-Law Fits Depend on Feature-Set Choice",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_powerlaw_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 9 (NEW) - NHANES feature-set comparison
print("Figure 9: NHANES feature-set sensitivity...")
fig, axes = plt.subplots(1, 3, figsize=(18,6), facecolor="white")
for i, fs in enumerate(NH_FEATURE_SETS):
    ax = axes[i]; cur = nh_curves_all_sets[fs]
    sizes_fs = cur[MODEL_NAMES[0]]["sizes"]
    draw_lc(ax, sizes_fs,
            {k:{"test":cur[k]["test"], "lo":cur[k]["lo"],
                "hi":cur[k]["hi"], "train":[]} for k in MODEL_NAMES},
            f"{FS_DISPLAY[fs]}", "NHANES Training Size", logx=True)
fig.suptitle("Figure 9. NHANES Sensitivity to Feature-Set Choice "
             "(Reviewer #3: label-feature circularity test)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_nhanes_feature_sets.png", dpi=150, bbox_inches="tight")
plt.close()

# FIGURE 10 (NEW) - UCI ablation (drop top-3 features)
print("Figure 10: UCI ablation (top-3 dropped)...")
fig, ax = plt.subplots(figsize=(12,6), facecolor="white")
x = np.arange(len(MODEL_NAMES)); w = 0.35
orig = [full_results[n]["UCI"] for n in MODEL_NAMES]
abl  = [ablation_results[n]    for n in MODEL_NAMES]
ax.bar(x-w/2, orig, w, label="Original UCI", color="#C0392B")
ax.bar(x+w/2, abl,  w, label=f"UCI - {top3}", color="#1E8449")
ax.set_xticks(x); ax.set_xticklabels([PAL[n]["label"] for n in MODEL_NAMES], rotation=20)
ax.set_ylabel("AUC-ROC"); ax.set_ylim(0,1.05)
ax.set_title(f"Figure 10. UCI Ablation Study (Reviewer #24)\n"
             f"Dropping top-3 features ({', '.join(top3)}) shows the ceiling persists",
             fontsize=13, fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/fig10_uci_ablation.png", dpi=150, bbox_inches="tight")
plt.close()

def write_csv(name, rows, columns=None):
    df = pd.DataFrame(rows)
    if columns is not None:
        df = df[columns]
    df.to_csv(LATEX_DATA / name, index=False, na_rep="nan")

def fit_value(fit, key):
    return (fit or {}).get(key, np.nan)

def fit_ci_value(fit, param_idx, bound_idx):
    if fit and fit.get("ci") is not None:
        return fit["ci"][bound_idx][param_idx]
    return np.nan

def fit_sensitivity_value(fit, key):
    if fit and fit.get("sensitivity_drop2_lowest"):
        return fit["sensitivity_drop2_lowest"].get(key, np.nan)
    return np.nan

def write_curve_wide(name, sizes, curves, include_train=False):
    rows = []
    for i, size in enumerate(sizes):
        row = {"train_size": size}
        for model in MODEL_NAMES:
            if i < len(curves[model]["test"]):
                row[f"{model}_test"] = curves[model]["test"][i]
                row[f"{model}_lo"] = curves[model]["lo"][i]
                row[f"{model}_hi"] = curves[model]["hi"][i]
            if include_train and i < len(curves[model].get("train", [])):
                row[f"{model}_train"] = curves[model]["train"][i]
        rows.append(row)
    write_csv(name, rows)

def write_roc_files(prefix, roc_data):
    for model in MODEL_NAMES:
        write_csv(f"{prefix}_{model}.csv", [
            {"fpr": fpr, "tpr": tpr}
            for fpr, tpr in zip(roc_data[model]["fpr"], roc_data[model]["tpr"])
        ])

# Table exports
write_csv("table1_dataset_audit.csv", dataset_audit)
write_csv("table4_full_results.csv", [
    {
        "model": n,
        "uci_auc": full_results[n]["UCI"],
        "nhanes_auc": full_results[n]["NHANES"],
        "tawam_auc": full_results[n]["Tawam"],
        "nhanes_brier": calib_results[n]["NHANES_brier"],
        "tawam_brier": calib_results[n]["Tawam_brier"],
    }
    for n in MODEL_NAMES
])
write_csv("table4_threshold_metrics.csv", [
    {
        "model": n,
        "dataset": ds,
        "threshold": threshold_results[n][ds]["threshold"],
        "sensitivity": threshold_results[n][ds]["sensitivity"],
        "specificity": threshold_results[n][ds]["specificity"],
    }
    for n in MODEL_NAMES for ds in ["NHANES", "Tawam"]
])
write_csv("table5_nhanes_feature_sets.csv", [
    {"model": n, **{fs: nh_full_aucs[fs][n] for fs in NH_FEATURE_SETS}}
    for n in MODEL_NAMES
])
write_csv("table6_powerlaw.csv", [
    {
        "feature_set": fs,
        "feature_set_label": FS_DISPLAY[fs],
        "model": n,
        "ceiling": fit_value(nh_plaw_all_sets[fs][n], "ceiling"),
        "ceiling_ci_lo": fit_ci_value(nh_plaw_all_sets[fs][n], 0, 0),
        "ceiling_ci_hi": fit_ci_value(nh_plaw_all_sets[fs][n], 0, 1),
        "a": fit_value(nh_plaw_all_sets[fs][n], "a"),
        "a_ci_lo": fit_ci_value(nh_plaw_all_sets[fs][n], 1, 0),
        "a_ci_hi": fit_ci_value(nh_plaw_all_sets[fs][n], 1, 1),
        "b": fit_value(nh_plaw_all_sets[fs][n], "b"),
        "b_ci_lo": fit_ci_value(nh_plaw_all_sets[fs][n], 2, 0),
        "b_ci_hi": fit_ci_value(nh_plaw_all_sets[fs][n], 2, 1),
        "r2": fit_value(nh_plaw_all_sets[fs][n], "r2"),
        "sensitivity_ceiling": fit_sensitivity_value(nh_plaw_all_sets[fs][n], "ceiling"),
    }
    for fs in NH_FEATURE_SETS for n in MODEL_NAMES
])

# Figure data exports
write_csv("fig1_model_comparison_wide.csv", [
    {"model": n, **{ds: full_results[n][ds] for ds in ["UCI", "NHANES", "Tawam"]}}
    for n in MODEL_NAMES
])
write_csv("fig1_model_comparison.csv", [
    {"model": n, "dataset": ds, "auc": full_results[n][ds]}
    for n in MODEL_NAMES for ds in ["UCI", "NHANES", "Tawam"]
])
write_curve_wide("fig2_uci_learning_curves_wide.csv", UCI_SIZES, uci_curves,
                 include_train=True)
write_csv("fig2_uci_learning_curves.csv", [
    {
        "model": n, "train_size": size, "test_auc": uci_curves[n]["test"][i],
        "test_lo": uci_curves[n]["lo"][i], "test_hi": uci_curves[n]["hi"][i],
        "train_auc": uci_curves[n]["train"][i],
    }
    for n in MODEL_NAMES for i, size in enumerate(UCI_SIZES)
])
write_curve_wide("fig3_nhanes_learning_curves_wide.csv",
                 nh_curves_all_sets["all"][MODEL_NAMES[0]]["sizes"],
                 nh_curves_all_sets["all"])
write_csv("fig3_nhanes_learning_curves.csv", [
    {
        "model": n, "feature_set": "all", "train_size": size,
        "test_auc": nh_curves_all_sets["all"][n]["test"][i],
        "test_lo": nh_curves_all_sets["all"][n]["lo"][i],
        "test_hi": nh_curves_all_sets["all"][n]["hi"][i],
    }
    for n in MODEL_NAMES
    for i, size in enumerate(nh_curves_all_sets["all"][n]["sizes"])
])
write_curve_wide("fig4_tawam_learning_curves_wide.csv", TW_SIZES, tw_curves,
                 include_train=True)
write_csv("fig4_tawam_learning_curves.csv", [
    {
        "model": n, "train_size": size, "test_auc": tw_curves[n]["test"][i],
        "test_lo": tw_curves[n]["lo"][i], "test_hi": tw_curves[n]["hi"][i],
        "train_auc": tw_curves[n]["train"][i],
    }
    for n in MODEL_NAMES for i, size in enumerate(TW_SIZES)
])
write_csv("fig5_single_feature_auc.csv", [
    {"feature": feat, "display_name": SF_FEATS_DISPLAY.get(feat, feat), "auc": auc}
    for feat, auc in sorted_sf
])
write_csv("fig6_cross_dataset.csv", [
    {"model": n, "target": target, "auc": cross_results[n][target]}
    for n in MODEL_NAMES for target in transfer_targets
])
write_csv("fig6_cross_dataset_wide.csv", [
    {"model": n, **{target: cross_results[n][target] for target in transfer_targets}}
    for n in MODEL_NAMES
])
write_csv("fig7_roc_uci.csv", [
    {"model": n, "fpr": fpr, "tpr": tpr, "auc": roc_uci[n]["auc"]}
    for n in MODEL_NAMES
    for fpr, tpr in zip(roc_uci[n]["fpr"], roc_uci[n]["tpr"])
])
write_csv("fig7_roc_tawam.csv", [
    {"model": n, "fpr": fpr, "tpr": tpr, "auc": roc_tw[n]["auc"]}
    for n in MODEL_NAMES
    for fpr, tpr in zip(roc_tw[n]["fpr"], roc_tw[n]["tpr"])
])
write_roc_files("fig7_roc_uci", roc_uci)
write_roc_files("fig7_roc_tawam", roc_tw)
write_csv("fig8_powerlaw_curves.csv", [
    {
        "feature_set": fs, "train_size": float(size),
        "observed_hgb": np.interp(size, nh_curves_all_sets[fs]["HGB"]["sizes"],
                                   nh_curves_all_sets[fs]["HGB"]["test"]),
        "fit_hgb": (
            power_law(size, nh_plaw_all_sets[fs]["HGB"]["ceiling"],
                      nh_plaw_all_sets[fs]["HGB"]["a"],
                      nh_plaw_all_sets[fs]["HGB"]["b"])
            if nh_plaw_all_sets[fs]["HGB"] else np.nan
        ),
    }
    for fs in NH_FEATURE_SETS for size in grid
])
for fs in NH_FEATURE_SETS:
    write_csv(f"fig8_powerlaw_{fs}.csv", [
        {
            "train_size": float(size),
            "observed_hgb": np.interp(size, nh_curves_all_sets[fs]["HGB"]["sizes"],
                                      nh_curves_all_sets[fs]["HGB"]["test"]),
            "fit_hgb": (
                power_law(size, nh_plaw_all_sets[fs]["HGB"]["ceiling"],
                          nh_plaw_all_sets[fs]["HGB"]["a"],
                          nh_plaw_all_sets[fs]["HGB"]["b"])
                if nh_plaw_all_sets[fs]["HGB"] else np.nan
            ),
        }
        for size in grid
    ])
write_csv("fig9_nhanes_feature_sets.csv", [
    {
        "model": n, "feature_set": fs, "train_size": size,
        "test_auc": nh_curves_all_sets[fs][n]["test"][i],
        "test_lo": nh_curves_all_sets[fs][n]["lo"][i],
        "test_hi": nh_curves_all_sets[fs][n]["hi"][i],
    }
    for fs in NH_FEATURE_SETS for n in MODEL_NAMES
    for i, size in enumerate(nh_curves_all_sets[fs][n]["sizes"])
])
for fs in NH_FEATURE_SETS:
    write_curve_wide(f"fig9_nhanes_{fs}_wide.csv",
                     nh_curves_all_sets[fs][MODEL_NAMES[0]]["sizes"],
                     nh_curves_all_sets[fs])
write_csv("fig10_ablation.csv", [
    {
        "model": n,
        "original_auc": full_results[n]["UCI"],
        "ablated_auc": ablation_results[n],
    }
    for n in MODEL_NAMES
])
write_csv("fig10_ablation_curve.csv", [
    {"train_size": size, "HGB_ablated": auc}
    for size, auc in zip(ablation_curve["sizes"], ablation_curve["test"])
])

# ══════════════════════════════════════════════════════════════════
# RESULTS DUMP (JSON for paper integration)
# ══════════════════════════════════════════════════════════════════
def serialize(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    return obj

results_dump = {
    "config": {"N_REPS": N_REPS, "N_REPS_NH": N_REPS_NH,
               "N_REPS_TW": N_REPS_TW, "N_BOOT": N_BOOT},
    "dataset_audit": dataset_audit,
    "full_results": full_results,
    "calib_results": calib_results,
    "threshold_results": threshold_results,
    "uci_curves": {k: {kk: [serialize(x) for x in vv] if isinstance(vv, list) else vv
                       for kk, vv in v.items()} for k, v in uci_curves.items()},
    "nh_curves_all_sets": {fs: {k: {kk: [serialize(x) for x in vv]
                                    if isinstance(vv, list) else vv
                                    for kk, vv in v.items()}
                                for k, v in nh_curves_all_sets[fs].items()}
                           for fs in NH_FEATURE_SETS},
    "nh_plaw_all_sets": nh_plaw_all_sets,
    "nh_full_aucs": nh_full_aucs,
    "tw_curves": {k: {kk: [serialize(x) for x in vv] if isinstance(vv, list) else vv
                      for kk, vv in v.items()} for k, v in tw_curves.items()},
    "single_feat": single_feat,
    "top3_features": top3,
    "ablation_results": ablation_results,
    "ablation_curve": ablation_curve,
    "cross_results": cross_results,
    "uci_sizes": UCI_SIZES, "nh_sizes": NH_SIZES, "tw_sizes": TW_SIZES,
}
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results_dump, f, indent=2, default=serialize)
print(f"\nResults saved to {OUT}/results.json")

# ══════════════════════════════════════════════════════════════════
# NUMERICAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("NUMERICAL RESULTS SUMMARY")
print("="*70)
print("\nFull-data AUC (all features):")
print(f"{'Model':<18} {'UCI':>8} {'NHANES':>8} {'Tawam':>8} {'BrierNH':>9} {'BrierTW':>9}")
print("-"*65)
for n in MODEL_NAMES:
    r = full_results[n]; c = calib_results[n]
    print(f"{n:<18} {r['UCI']:>8.4f} {r['NHANES']:>8.4f} {r['Tawam']:>8.4f} "
          f"{c['NHANES_brier']:>9.4f} {c['Tawam_brier']:>9.4f}")

print("\nNHANES full-data AUC by feature set:")
print(f"{'Model':<18}", end="")
for fs in NH_FEATURE_SETS: print(f" {fs:>20}", end="")
print()
for n in MODEL_NAMES:
    print(f"{n:<18}", end="")
    for fs in NH_FEATURE_SETS: print(f" {nh_full_aucs[fs][n]:>20.4f}", end="")
    print()

print("\nPower-law ceilings (NHANES, all features):")
for n in MODEL_NAMES:
    p = nh_plaw[n]
    if p:
        print(f"  {n:<18}: ceiling={p['ceiling']:.4f} (R²={p['r2']:.3f}), "
              f"a={p['a']:.3f}, b={p['b']:.3f}")

print("\nCross-dataset evaluation (UCI-trained, common features):")
print(f"{'Model':<18} {'UCI->UCI':>10} {'UCI->NHANES':>12} {'UCI->Tawam':>11}")
for n in MODEL_NAMES:
    r = cross_results[n]
    print(f"{n:<18} {r['UCI_self_commonfeats']:>10.3f} "
          f"{r['UCI_to_NHANES']:>12.3f} {r['UCI_to_Tawam']:>11.3f}")

print("\nUCI ablation (drop top-3):")
for n in MODEL_NAMES:
    print(f"  {n:<18}: original={full_results[n]['UCI']:.4f} "
          f"-> ablated={ablation_results[n]:.4f}")

print("\nTop-5 single-feature AUC (UCI, logistic regression):")
for feat, auc in sorted_sf[:5]:
    print(f"  {SF_FEATS_DISPLAY.get(feat,feat):<25}: {auc:.4f}")

print("\nDone.")
