"""
=============================================================================
PIPELINE 3 (generic) - full treatment for any matrix dumped by pipeline 2
=============================================================================
Splines on age, ordinal encoding, nested CV, calibration, SHAP, 6 figures.

TO RUN MODEL A (your primary model: among people who report no diagnosis,
who actually has BP >= 140/90?):

    # 1. build and dump the matrix  (about 1 minute)
    python pns2013_hypertension_pipeline2.py undiagnosed prune dump fast

    # 2. run the full treatment     (about 20 minutes; nested CV is the cost)
    python pipeline3_generic.py matrix_pruned_undiagnosed.pkl modelA

Arguments:
    argv[1]  path to the .pkl dumped by pipeline 2
    argv[2]  tag used to name every output file

To rerun Model B for comparison:
    python pns2013_hypertension_pipeline2.py nondiagnosis prune dump fast
    python pipeline3_generic.py matrix_pruned_nondiagnosis.pkl modelB

WHAT DIFFERS FROM THE MODEL B RUN
Model A is imbalanced (15% positive) where Model B was balanced (50%).
Three things follow from that and are handled below:
  - PR-AUC matters more than AUC for ranking who to screen first
  - calibration is more likely to help, so the calibrated model is reported
  - a fixed high-sensitivity operating point is reported alongside Youden,
    because for screening you usually care about missing few true cases
=============================================================================
"""

import warnings, pickle, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   OrdinalEncoder, SplineTransformer)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             average_precision_score, brier_score_loss,
                             precision_recall_curve)
from scipy.stats import loguniform, randint, uniform
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")
RS = 42
plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

MATRIX = sys.argv[1] if len(sys.argv) > 1 else "matrix_pruned_undiagnosed.pkl"
TAG    = sys.argv[2] if len(sys.argv) > 2 else "modelA"
TARGET_SENS = 0.90          # screening operating point

# label the two classes so the figures read correctly for either model
LABELS = {"modelA": ("BP normal", "BP >= 140/90"),
          "modelB": ("diagnosed", "undiagnosed")}.get(TAG, ("negative", "positive"))

# ---------------------------------------------------------------------------
# STEP 1 - load
# ---------------------------------------------------------------------------
d = pickle.load(open(MATRIX, "rb"))
X, y, NUM, CAT, spec = d["X"], d["y"], d["NUM"], d["CAT"], d["spec"]
print(f"STEP 1 | {MATRIX}: {X.shape[0]} people, {X.shape[1]} predictors, "
      f"{y.mean():.1%} positive")

# ---------------------------------------------------------------------------
# STEP 2 - ordinal vs nominal
# ---------------------------------------------------------------------------
# Ordinal variables keep their order in a single column instead of becoming
# k dummies. Fewer columns and the ranking is not thrown away.
ORD = [c for c in CAT if spec.get(c, {}).get("recode") == "ordinal"]
NOM = [c for c in CAT if c not in ORD]
NUM_PLAIN = [c for c in NUM if c != "age"]
print(f"STEP 2 | {len(ORD)} ordinal, {len(NOM)} nominal, {len(NUM)} numeric")

def ord_key(c):
    """Sort levels numerically where possible so the encoding respects order."""
    vals = list(X[c].dropna().unique())
    num = [v for v in vals if str(v).replace(".", "").replace("-", "").isdigit()]
    txt = [v for v in vals if v not in num]
    return sorted(num, key=lambda v: float(v)) + sorted(txt, key=str)

ORD_CATS = [ord_key(c) for c in ORD]

# ---------------------------------------------------------------------------
# STEP 3 - preprocessor
# ---------------------------------------------------------------------------
# Risk is not linear in age. Natural cubic splines (5 knots) let the logistic
# regression bend without hand-picking cut-points.
def build_prep(spline=True, scale=True):
    blocks = []
    if spline:
        blocks.append(("spl", Pipeline([
            ("i", SimpleImputer(strategy="median")),
            ("s", SplineTransformer(n_knots=5, degree=3, extrapolation="linear")),
            ("z", StandardScaler())]), ["age"]))
        num_cols = NUM_PLAIN
    else:
        num_cols = NUM
    steps = [("i", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("z", StandardScaler()))
    blocks += [
        ("num", Pipeline(steps), num_cols),
        ("ord", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("e", OrdinalEncoder(categories=ORD_CATS,
                                               handle_unknown="use_encoded_value",
                                               unknown_value=-1))]), ORD),
        ("nom", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore",
                                              sparse_output=False,
                                              min_frequency=20))]), NOM),
    ]
    return ColumnTransformer(blocks)

# ---------------------------------------------------------------------------
# STEP 4 - split and models
# ---------------------------------------------------------------------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y,
                                      random_state=RS)
cv = StratifiedKFold(5, shuffle=True, random_state=RS)
spw = (ytr == 0).sum() / (ytr == 1).sum()
print(f"STEP 4 | train {len(Xtr)} | test {len(Xte)}")

MODELS = {
    "Logistic + splines": (
        Pipeline([("prep", build_prep(True, True)),
                  ("clf", LogisticRegression(max_iter=4000, class_weight="balanced",
                                             random_state=RS))]),
        {"clf__C": loguniform(1e-3, 1e2), "clf__penalty": ["l1", "l2"],
         "clf__solver": ["liblinear"]}),
    "Random Forest": (
        Pipeline([("prep", build_prep(False, False)),
                  ("clf", RandomForestClassifier(class_weight="balanced",
                                                 n_jobs=-1, random_state=RS))]),
        {"clf__n_estimators": randint(250, 600),
         "clf__max_depth": [None, 6, 10, 16],
         "clf__min_samples_leaf": randint(2, 25),
         "clf__max_features": ["sqrt", 0.3, 0.5]}),
    "XGBoost": (
        Pipeline([("prep", build_prep(False, False)),
                  ("clf", XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                                        n_jobs=-1, random_state=RS))]),
        {"clf__n_estimators": randint(200, 600),
         "clf__learning_rate": loguniform(0.01, 0.3),
         "clf__max_depth": randint(2, 6),
         "clf__subsample": uniform(0.6, 0.4),
         "clf__colsample_bytree": uniform(0.5, 0.5),
         "clf__min_child_weight": randint(1, 12)}),
}

# ---------------------------------------------------------------------------
# STEP 5 - nested CV (unbiased) + tuned fit
# ---------------------------------------------------------------------------
# The search sits INSIDE an outer loop, so every reported number comes from
# data the hyper-parameter search never saw. Slower, but honest.
print("\nSTEP 5 | nested CV + tuning (this is the slow part)")
rows, fitted, best_params = [], {}, {}
for name, (pipe, grid) in MODELS.items():
    search = RandomizedSearchCV(pipe, grid, n_iter=12, cv=cv,
                                scoring="roc_auc", n_jobs=-1, random_state=RS)
    nested = cross_val_score(search, Xtr, ytr,
                             cv=StratifiedKFold(5, shuffle=True, random_state=RS + 1),
                             scoring="roc_auc", n_jobs=-1)
    search.fit(Xtr, ytr)
    est = search.best_estimator_
    fitted[name] = est
    best_params[name] = {k.replace("clf__", ""): (round(v, 4) if isinstance(v, float) else v)
                         for k, v in search.best_params_.items()}
    p = est.predict_proba(Xte)[:, 1]
    rows.append(dict(model=name,
                     nested_CV_AUC=f"{nested.mean():.3f}+/-{nested.std():.3f}",
                     test_AUC=roc_auc_score(yte, p),
                     PR_AUC=average_precision_score(yte, p),
                     Brier=brier_score_loss(yte, p)))
    print(f"  {name:<20} nested {nested.mean():.3f} | test {roc_auc_score(yte, p):.3f}")

res = pd.DataFrame(rows).set_index("model")
best_name = res["test_AUC"].idxmax()
best = fitted[best_name]
print(f"\n{res.round(3).to_string()}\n  best: {best_name}")

# ---------------------------------------------------------------------------
# STEP 6 - calibration
# ---------------------------------------------------------------------------
# class_weight / scale_pos_weight distort probabilities. Isotonic regression
# maps scores back onto observed frequencies using training folds only.
cal = CalibratedClassifierCV(best, method="isotonic", cv=cv)
cal.fit(Xtr, ytr)
p_raw, p_cal = best.predict_proba(Xte)[:, 1], cal.predict_proba(Xte)[:, 1]
print(f"\nSTEP 6 | Brier {brier_score_loss(yte, p_raw):.3f} -> "
      f"{brier_score_loss(yte, p_cal):.3f} | AUC {roc_auc_score(yte, p_cal):.3f}")

# ---------------------------------------------------------------------------
# STEP 7 - SHAP
# ---------------------------------------------------------------------------
# SHAP shows DIRECTION as well as magnitude, unlike permutation importance.
print("\nSTEP 7 | SHAP")
xgb_pipe = fitted["XGBoost"]
prep = xgb_pipe.named_steps["prep"]
Xte_t = prep.transform(Xte)
feat_names = [f.split("__", 1)[-1] for f in prep.get_feature_names_out()]
sv = shap.TreeExplainer(xgb_pipe.named_steps["clf"]).shap_values(Xte_t)

# map a transformed name (e.g. "first_care_place_11") back to its source column
# by longest-prefix match. Splitting on "_" breaks multi-word names.
originals = sorted(X.columns, key=len, reverse=True)
def source_col(f):
    for c in originals:
        if f == c or f.startswith(c + "_"):
            return c
    return f

shap_imp = pd.DataFrame({"feature": feat_names,
                         "mean_abs_shap": np.abs(sv).mean(0)})
shap_imp["source"] = shap_imp.feature.map(source_col)
shap_imp["block"] = shap_imp.source.map(
    lambda c: spec.get(c.replace("_isNA", ""), {}).get("block", "-"))
shap_imp = shap_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
shap_imp.to_csv(f"shap_importance_{TAG}.csv", index=False)
print(shap_imp.head(15).round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 8 - parsimony: how few predictors do we need?
# ---------------------------------------------------------------------------
# Rank source columns by summed SHAP, refit with only the top k. Where the
# curve flattens is your simple model; where it falls, the extra variables
# are noise.
print("\nSTEP 8 | parsimony curve")
ranked = list(shap_imp.groupby("source").mean_abs_shap.sum()
              .sort_values(ascending=False).index)
ranked = [c for c in ranked if c in X.columns]
ranked += [c for c in X.columns if c not in ranked]

ks = [1, 2, 3, 5, 8, 12, 20, 30, 50, len(X.columns)]
aucs = []
for k in ks:
    cols = ranked[:k]
    n_k = [c for c in cols if c in NUM]
    o_k = [c for c in cols if c in ORD]
    m_k = [c for c in cols if c in NOM]
    blocks = [("num", Pipeline([("i", SimpleImputer(strategy="median")),
                                ("z", StandardScaler())]), n_k)]
    if o_k:
        blocks.append(("ord", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("e", OrdinalEncoder(categories=[ord_key(c) for c in o_k],
                                 handle_unknown="use_encoded_value",
                                 unknown_value=-1))]), o_k))
    if m_k:
        blocks.append(("nom", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                min_frequency=20))]), m_k))
    pk = Pipeline([("prep", ColumnTransformer(blocks)),
                   ("clf", LogisticRegression(max_iter=4000, C=0.1,
                                              class_weight="balanced",
                                              random_state=RS))])
    a = cross_val_score(pk, X[cols], y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
    aucs.append(a)
    print(f"  top {k:>3} -> CV AUC {a:.3f}   ({', '.join(cols[:3])}...)")

pd.DataFrame({"n_features": ks, "cv_auc": aucs}).to_csv(f"parsimony_{TAG}.csv", index=False)

# ---------------------------------------------------------------------------
# STEP 9 - figures
# ---------------------------------------------------------------------------
print("\nSTEP 9 | figures")
BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"

# fig1 ROC
plt.figure(figsize=(4.6, 4.4))
for (name, est), col in zip(fitted.items(), [BLUE, ORANGE, GREEN]):
    p = est.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, p)
    plt.plot(fpr, tpr, color=col, lw=1.8,
             label=f"{name} ({roc_auc_score(yte, p):.3f})")
plt.plot([0, 1], [0, 1], "--", color=GRAY, lw=1)
plt.xlabel("1 - specificity"); plt.ylabel("Sensitivity")
plt.title(f"ROC - {TAG} (test n={len(yte)})")
plt.legend(loc="lower right", frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"fig1_roc_{TAG}.png"); plt.close()

# fig2 confusion matrices: Youden and a fixed high-sensitivity point
fpr, tpr, thr = roc_curve(yte, p_cal)
t_youden = thr[np.argmax(tpr - fpr)]
t_sens = thr[np.argmin(np.abs(tpr - TARGET_SENS))]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, t, ttl in [(axes[0], t_youden, "Youden"),
                   (axes[1], t_sens, f"sensitivity ~{TARGET_SENS:.0%}")]:
    cm = confusion_matrix(yte, (p_cal >= t).astype(int))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    tn, fp, fn, tp = cm.ravel()
    ax.set_xticks([0, 1], [f"pred {LABELS[0]}", f"pred {LABELS[1]}"], fontsize=7)
    ax.set_yticks([0, 1], LABELS, fontsize=7)
    ax.set_title(f"{ttl} (t={t:.2f})\nsens {tp/(tp+fn):.2f} | spec {tn/(tn+fp):.2f} | "
                 f"PPV {tp/(tp+fp):.2f}", fontsize=8)
plt.tight_layout(); plt.savefig(f"fig2_confusion_{TAG}.png"); plt.close()

# fig3 calibration
plt.figure(figsize=(4.4, 4.2))
for probs, lab, col in [(p_raw, "raw", ORANGE), (p_cal, "isotonic", BLUE)]:
    pt, pp = calibration_curve(yte, probs, n_bins=8, strategy="quantile")
    plt.plot(pp, pt, "o-", color=col, lw=1.6, ms=4,
             label=f"{lab} (Brier {brier_score_loss(yte, probs):.3f})")
plt.plot([0, 1], [0, 1], "--", color=GRAY, lw=1)
plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
plt.title("Calibration"); plt.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"fig3_calibration_{TAG}.png"); plt.close()

# fig4 SHAP beeswarm
plt.figure()
shap.summary_plot(sv, Xte_t, feature_names=feat_names, max_display=15, show=False)
plt.tight_layout(); plt.savefig(f"fig4_shap_{TAG}.png"); plt.close()

# fig5 parsimony
plt.figure(figsize=(5.2, 3.8))
plt.plot(ks, aucs, "o-", color=BLUE, lw=1.8, ms=5)
plt.axhline(max(aucs), ls="--", color=GRAY, lw=1)
plt.xscale("log"); plt.xticks(ks, [str(k) for k in ks])
plt.xlabel("Number of predictors (ranked by SHAP)")
plt.ylabel("Cross-validated AUC")
plt.title("How many predictors do we actually need?")
plt.tight_layout(); plt.savefig(f"fig5_parsimony_{TAG}.png"); plt.close()

# fig6 blocks
plt.figure(figsize=(5.2, 3.4))
bl = shap_imp.groupby("block").mean_abs_shap.sum().sort_values()
plt.barh(bl.index, bl.values, color=BLUE)
plt.xlabel("Summed mean |SHAP|"); plt.title("Contribution by variable block")
plt.tight_layout(); plt.savefig(f"fig6_blocks_{TAG}.png"); plt.close()

# fig7 precision-recall - more informative than ROC when prevalence is low
prec, rec, _ = precision_recall_curve(yte, p_cal)
plt.figure(figsize=(4.6, 4.0))
plt.plot(rec, prec, color=BLUE, lw=1.8,
         label=f"AP {average_precision_score(yte, p_cal):.3f}")
plt.axhline(yte.mean(), ls="--", color=GRAY, lw=1,
            label=f"prevalence {yte.mean():.2f}")
plt.xlabel("Recall (sensitivity)"); plt.ylabel("Precision (PPV)")
plt.title("Precision-recall"); plt.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"fig7_pr_{TAG}.png"); plt.close()

res.to_csv(f"results_{TAG}.csv")
json.dump(best_params, open(f"best_params_{TAG}.json", "w"), indent=2, default=str)
print(f"  saved fig1..fig7_{TAG}.png, results_{TAG}.csv, "
      f"shap_importance_{TAG}.csv, parsimony_{TAG}.csv")
