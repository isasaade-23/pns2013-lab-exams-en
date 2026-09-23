"""
=============================================================================
PNS 2013 | SHARED MODEL SIDE
=============================================================================
Everything the model notebooks share, written once so that twelve notebooks
cannot drift apart: the same rows in the same split, the same metrics, the
same export format.

This file contains no preprocessing decision. Layer 1 (build_matrix) and
Layer 2 (build_preprocessor) both live in pns_preprocess.py, which reads the
registry workbook. To change a variable, change the workbook.

What is here
    load_or_build()   the frozen matrix, rebuilt only when the inputs moved
    split()           stratified 80/20, random_state=42
    evaluate()        test-set metrics, with a bootstrap interval on AUC
    screening_point() the operating point at a target sensitivity
    permutation_report()  per feature and summed per registry block
    plot_roc(), plot_calibration(), plot_importance()
    export()          one run, one folder, one tidy metrics row
    push_results()    that folder, committed to the private run repository

Usage in a notebook
    import pns_modelkit as mk
    b = mk.load_or_build("hypertension", data=DATA, registry=REG, outdir=OUT)
    Xtr, Xte, ytr, yte = mk.split(b)
=============================================================================
"""

import json
import os
import pickle
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

import pns_preprocess as pp

RS = 42

# The three frozen builds, as verified on the real file after the four group
# decisions of 22/09/2026: pregnancy-only diagnoses and undefined pregnancy
# leave the base, insulin is declared, and the two time-since-measurement items
# return as predictors. A notebook that does not reproduce these numbers is
# running against a different data file or a different registry, and should
# stop rather than report a result.
#
# For the record, the counts before those decisions were 6,346 / 6,832 / 5,944.
EXPECTED = {
    "hypertension": dict(n=6329, prev=0.152),
    "diabetes":     dict(n=6812, prev=0.038),
    "cholesterol":  dict(n=5927, prev=0.321),
}


# ===========================================================================
# THE MATRIX
# ===========================================================================
def _tag(outcome, cohort, thr):
    return f"{outcome}_{cohort}_{'-'.join(str(t) for t in thr)}"


def load_or_build(outcome, data, registry, outdir="outputs", threshold=None,
                  cohort="undiagnosed", force=False):
    """Return the frozen bundle, rebuilding only when it has to.

    build_matrix() reads a 12 MB workbook and takes about a minute. It is
    deterministic, and it stamps the sha256 of both inputs into the bundle, so
    a cached pickle whose two hashes still match is the same matrix the build
    would produce. Anything else — a new registry, a new data file, force —
    rebuilds.
    """
    thr = tuple(threshold) if threshold else pp.OUTCOMES[outcome]["default_threshold"]
    path = os.path.join(outdir, f"matrix_{_tag(outcome, cohort, thr)}.pkl")

    if not force and os.path.exists(path):
        with open(path, "rb") as f:
            b = pickle.load(f)
        if (b.get("data_sha") == pp._sha(data)
                and b.get("registry_sha") == pp._sha(registry)):
            print(f"cached matrix reused: {path}")
            print(f"  {b['X'].shape[0]} rows x {b['X'].shape[1]} predictors, "
                  f"prevalence {b['y'].mean():.1%}")
            return b
        print("cached matrix is stale (data or registry changed), rebuilding")

    return pp.build_matrix(data, registry, outcome, threshold=thr, cohort=cohort,
                           outdir=outdir)


def check_frozen(bundle, tol_n=0, tol_prev=0.002):
    """Assert the build against the counts reported to the group."""
    e = EXPECTED.get(bundle["outcome"])
    if e is None or bundle["cohort"] != "undiagnosed" \
            or tuple(bundle["threshold"]) != tuple(
                pp.OUTCOMES[bundle["outcome"]]["default_threshold"]):
        print("non-default build, no frozen counts to check against")
        return
    n, prev = len(bundle["X"]), float(bundle["y"].mean())

    # One mismatch has a known cause, so name it rather than let the reader
    # guess. Bug 1 of the 15/09/2026 build note: the diagnosis gates must never
    # be imputed, because a zero-filled gate silently declares an unanswered
    # row undiagnosed. In the registry as shipped, dx_diabetes and
    # dx_cholesterol still carry na_rule = implied:0, which is what produces
    # these two inflated cohorts.
    if n > e["n"] + 200 and bundle["outcome"] in ("diabetes", "cholesterol"):
        raise AssertionError(
            f"{bundle['outcome']} built {n} rows, expected {e['n']}. A cohort "
            "this much larger means the gate was imputed: check that "
            f"dx_{bundle['outcome']} has na_rule = mar in the registry, so an "
            "unanswered gate leaves the analysis instead of being filled with "
            "zero and declared undiagnosed.")

    assert abs(n - e["n"]) <= tol_n, f"n is {n}, expected {e['n']}"
    assert abs(prev - e["prev"]) <= tol_prev, \
        f"prevalence is {prev:.3f}, expected {e['prev']:.3f}"
    print(f"frozen counts reproduced: {n} rows, prevalence {prev:.1%}")


def attrition(bundle, outdir):
    p = os.path.join(outdir, f"attrition_{_tag(bundle['outcome'], bundle['cohort'], bundle['threshold'])}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def split(bundle, test_size=0.2):
    """Stratified 80/20 on a deterministic matrix, so every model notebook for
    one outcome trains and tests on exactly the same rows."""
    X, y = bundle["X"], bundle["y"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          stratify=y, random_state=RS)
    print(f"train {len(Xtr)} rows, prevalence {ytr.mean():.1%}")
    print(f"test  {len(Xte)} rows, prevalence {yte.mean():.1%}")
    return Xtr, Xte, ytr, yte


def pos_weight(ytr):
    """scale_pos_weight for the boosted trees."""
    return float((ytr == 0).sum() / (ytr == 1).sum())


# ===========================================================================
# LAYER 2, WITH TWO COMPATIBILITY FIXES
# ===========================================================================
class _StringCategories(BaseEstimator, TransformerMixin):
    """Cast the one-hot columns to string, keeping a DataFrame.

    `_roles()` declares the one-hot categories with `.astype(str)` but leaves
    the columns themselves numeric, so `OneHotEncoder` is handed a float column
    and a list of strings. scikit-learn then refuses: 'Unsorted categories are
    not supported for numerical categories'. The same failure was fixed once
    before in the script generation — decision log section 11, 'all categoricals
    cast to string labels' — and did not survive into this file.

    Casting here rather than in the matrix keeps the frozen matrix untouched.
    """

    def __init__(self, columns):
        # stored exactly as given: sklearn's clone() refuses a constructor that
        # modifies its own parameter
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in list(self.columns):
            if c in X.columns:
                X[c] = X[c].astype(str).where(X[c].notna(), np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def preprocessor(bundle, spline=True, scale=True, verbose=False):
    """`pns_preprocess.build_preprocessor`, made fittable.

    Two mechanical fixes, neither of which changes a variable decision:

    1. One-hot columns are cast to string, so the declared categories and the
       column agree on a type.
    2. Ordinal categories are sorted ascending. `OrdinalEncoder` rejects
       unsorted numeric categories, and three ordinals arrive descending
       (`passive_smoke`, `diet_salt_perception`, `smoking_status`) because the
       registry still records the *pre*-reversal order while `_recode_fixes`
       already reversed the values. Ascending is what 'higher means more
       exposure' means after that recode, so this preserves the intent rather
       than reversing it a second time.

    Both are reported to the build's author; until the shared file changes,
    every notebook must go through this function rather than call Layer 2
    directly, or the twelve runs would not encode alike.
    """
    R, C = bundle["roles"], dict(bundle["categories"])

    resorted = []
    for c in R["ordinal"]:
        cats = C.get(c)
        if cats and all(isinstance(v, (int, float, np.integer, np.floating)) for v in cats):
            s = sorted(cats)
            if list(s) != list(cats):
                C[c] = s
                resorted.append(c)

    patched = dict(bundle, categories=C)
    ct = pp.build_preprocessor(patched, spline=spline, scale=scale)

    onehot = list(R["nominal"]) + list(R["expl_bin"])
    if verbose:
        print(f"one-hot columns cast to string: {len(onehot)}")
        if resorted:
            print(f"ordinal categories re-sorted ascending: {', '.join(resorted)}")

    from sklearn.pipeline import Pipeline as _Pipeline
    return _Pipeline([("cats", _StringCategories(onehot)), ("ct", ct)])


# ===========================================================================
# METRICS
# ===========================================================================
def _at_threshold(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
    return dict(threshold=float(t),
                Sensitivity=tp / (tp + fn) if (tp + fn) else np.nan,
                Specificity=tn / (tn + fp) if (tn + fp) else np.nan,
                PPV=tp / (tp + fp) if (tp + fp) else np.nan,
                NPV=tn / (tn + fn) if (tn + fn) else np.nan,
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def confusion_frame(y, p, threshold, labels=("no outcome", "outcome")):
    """The 2x2 as a labelled table: counts, and the share of each true row.

    The row percentage is the one worth reading. Of the people who do have the
    outcome, what share does the model flag; of the people who do not, what
    share does it flag anyway. At 15% prevalence the column percentage looks
    alarming and says mostly that the outcome is rare.
    """
    d = _at_threshold(y, p, threshold)
    pos, neg = d["TP"] + d["FN"], d["TN"] + d["FP"]
    frame = pd.DataFrame(
        [[d["TN"], d["FP"], neg], [d["FN"], d["TP"], pos],
         [d["TN"] + d["FN"], d["FP"] + d["TP"], neg + pos]],
        index=[f"true {labels[0]}", f"true {labels[1]}", "total"],
        columns=[f"predicted {labels[0]}", f"predicted {labels[1]}", "total"])
    frame.attrs["rates"] = {
        "threshold": d["threshold"], "sensitivity": d["Sensitivity"],
        "specificity": d["Specificity"], "PPV": d["PPV"], "NPV": d["NPV"]}
    return frame


def calibration_stats(y, p, eps=1e-6):
    """Calibration intercept and slope, plus Brier and its skill score.

    Regress the outcome on the logit of the predicted probability. A perfectly
    calibrated model gives intercept 0 and slope 1. Intercept above 0 means the
    predictions sit systematically below the observed risk, and below 0 means
    they sit above it, which is what class-imbalance corrections do: trained on
    a resampled world with far more cases than the real one, the model reports
    a risk that population never had. Slope below 1 means the predictions are
    too spread out, the signature of overfitting.

    This is the pair of numbers that makes the SMOTE question answerable, since
    AUC is invariant to any monotone rescaling of the probabilities and will not
    move when calibration breaks.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    logit = np.log(p / (1 - p))

    slope = intercept = np.nan
    if len(np.unique(y)) == 2:
        # C very large is an unpenalised fit. The filter silences an
        # OptimizeWarning some scipy/scikit-learn pairs raise about an
        # lbfgs option, which says nothing about this fit.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unknown solver options")
            lr = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
            lr.fit(logit.reshape(-1, 1), y)
        slope = float(lr.coef_[0][0])

        # Calibration in the large: the shift a that makes the mean predicted
        # risk equal the observed one, with the slope held at 1. Solved by
        # bisection because sklearn has no offset term; the function is
        # monotone in a, so a bracket of +/-20 on the logit scale is ample.
        def mean_pred(a):
            return float(np.mean(1 / (1 + np.exp(-(a + logit)))))

        target, lo, hi = float(np.mean(y)), -20.0, 20.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if mean_pred(mid) < target:
                lo = mid
            else:
                hi = mid
        intercept = (lo + hi) / 2

    brier = float(np.mean((p - y) ** 2))
    base = float(np.mean((np.mean(y) - y) ** 2))
    return dict(calib_intercept=intercept, calib_slope=slope,
                Brier=brier, Brier_skill=float(1 - brier / base) if base else np.nan,
                mean_predicted=float(np.mean(p)), observed=float(np.mean(y)))


def bootstrap_auc(y, p, n_boot=1000, alpha=0.05, random_state=RS):
    """Percentile interval on the test AUC, stratified by outcome so every
    draw keeps the same number of positives.

    Worth the twenty lines here: at 3.8% prevalence the diabetes test half
    holds about fifty positive cases, and a bare point estimate says more than
    those fifty rows can support.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    ipos, ineg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ipos, ipos.size, replace=True),
                              rng.choice(ineg, ineg.size, replace=True)])
        out.append(roc_auc_score(y[idx], np.asarray(p)[idx]))
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def screening_point(y, p, target_sensitivity=0.90):
    """The operating point a screening instrument would actually be set at:
    the highest cut-off that still reaches the target sensitivity."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.flatnonzero(tpr >= target_sensitivity)
    t = thr[ok[0]] if ok.size else thr[-1]
    d = _at_threshold(y, p, t)
    d["rule"] = f"sensitivity >= {target_sensitivity:.0%}"
    return d


def evaluate(model, Xte, yte, label, n_boot=1000):
    """Test-set metrics for one fitted pipeline. Youden's J is the reported
    cut-off, as in the hypertension notebook."""
    p = model.predict_proba(Xte)[:, 1]
    fpr, tpr, thr = roc_curve(yte, p)
    t = thr[np.argmax(tpr - fpr)]
    lo, hi = bootstrap_auc(yte, p, n_boot=n_boot)
    row = dict(model=label,
               AUC_test=float(roc_auc_score(yte, p)),
               AUC_lo=lo, AUC_hi=hi,
               PR_AUC=float(average_precision_score(yte, p)),
               Brier=float(brier_score_loss(yte, p)))
    # AUC does not move when calibration breaks, so it cannot answer whether a
    # resampling variant helped or hurt. These can.
    row.update({k: v for k, v in calibration_stats(yte, p).items()
                if k != "Brier"})
    row.update({f"youden_{k}": v for k, v in _at_threshold(yte, p, t).items()})
    row.update({f"screen90_{k}": v for k, v in
                screening_point(yte, p, 0.90).items() if k != "rule"})
    return row, p


def metrics_frame(rows):
    df = pd.DataFrame(rows)
    front = ["model", "AUC_test", "AUC_lo", "AUC_hi", "PR_AUC", "Brier",
             "youden_Sensitivity", "youden_Specificity", "youden_PPV", "youden_NPV"]
    return df[[c for c in front if c in df] + [c for c in df if c not in front]]


# ===========================================================================
# IMPORTANCE
# ===========================================================================
def permutation_report(model, Xte, yte, bundle, n_repeats=10, top=None,
                       random_state=RS, n_jobs=None):
    """Permutation importance on the test set, per feature and summed per
    registry block.

    Permuted on columns of the raw matrix, not on encoded columns, so one row
    is one questionnaire variable and the block totals mean what the decision
    log says they mean. The shuffle happens inside the full frame — the model
    still sees every column it was fitted on, which is why this is written out
    rather than handed to sklearn's helper with a column subset.

    `top` restricts the permuted set to the features most associated with the
    model's own predictions, which is what the TabPFN notebooks use: every
    permutation there is a billed round trip to the API.
    """
    base_p = model.predict_proba(Xte)[:, 1]
    base = roc_auc_score(yte, base_p)
    cols = list(Xte.columns)

    if top and top < len(cols):
        rank = []
        for c in cols:
            v = pd.to_numeric(Xte[c], errors="coerce")
            if v.notna().sum() < 2 or v.nunique(dropna=True) < 2:
                rank.append((c, 0.0))
            else:
                rank.append((c, abs(np.corrcoef(v.fillna(v.median()), base_p)[0, 1])))
        cols = [c for c, _ in sorted(rank, key=lambda r: -r[1])[:top]]

    rng = np.random.default_rng(random_state)
    rows = []
    for c in cols:
        drops = []
        for _ in range(n_repeats):
            Xp = Xte.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(base - roc_auc_score(yte, model.predict_proba(Xp)[:, 1]))
        rows.append((c, float(np.mean(drops)), float(np.std(drops))))

    imp = pd.DataFrame(rows, columns=["feature", "drop_in_auc", "sd"])

    spec = bundle["spec"]
    def block_of(f):
        base_name = f[:-5] if f.endswith("_isNA") else f
        s = spec.get(base_name)
        return s["block"] if s else "unmapped"

    imp["block"] = imp["feature"].map(block_of)
    imp = imp.sort_values("drop_in_auc", ascending=False).reset_index(drop=True)
    blocks = (imp.groupby("block")["drop_in_auc"].sum()
              .sort_values(ascending=False).rename("summed_drop_in_auc").reset_index())
    print(f"baseline test AUC {base:.3f}; permuted {len(cols)} features "
          f"x {n_repeats} repeats")
    return imp, blocks


# ===========================================================================
# FIGURES
# ===========================================================================
def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300,
                         "savefig.bbox": "tight", "font.size": 9})
    return plt


def plot_confusion(y, p, threshold, title, path=None,
                   labels=("no outcome", "outcome")):
    """The 2x2 at one operating point: count, and share of the true row."""
    plt = _plt()
    d = _at_threshold(y, p, threshold)
    counts = np.array([[d["TN"], d["FP"]], [d["FN"], d["TP"]]], dtype=float)
    rows = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, rows, out=np.zeros_like(counts), where=rows > 0)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(counts[i, j]):,}\n{share[i, j]:.0%}",
                    ha="center", va="center",
                    color="white" if share[i, j] > 0.55 else "black")
    ax.set_xticks([0, 1], [f"predicted\n{labels[0]}", f"predicted\n{labels[1]}"])
    ax.set_yticks([0, 1], [f"true\n{labels[0]}", f"true\n{labels[1]}"])
    ax.set_title(f"{title}\ncut-off {d['threshold']:.3f} · "
                 f"sens {d['Sensitivity']:.0%} · spec {d['Specificity']:.0%} · "
                 f"PPV {d['PPV']:.0%}", fontsize=8)
    if path:
        fig.savefig(path)
    return fig


def plot_roc(curves, title, path=None):
    """curves: {label: (y_true, p)}"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for label, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.6, label=f"{label} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    if path:
        fig.savefig(path)
    return fig


def plot_calibration(y, p, title, path=None, bins=10):
    from sklearn.calibration import calibration_curve
    plt = _plt()
    obs, pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot(pred, obs, "o-", lw=1.4)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    if path:
        fig.savefig(path)
    return fig


def plot_importance(imp, title, path=None, top=20):
    plt = _plt()
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * len(d) + 1.2))
    ax.barh(d["feature"], d["drop_in_auc"], xerr=d["sd"], height=0.7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Drop in test AUC when shuffled")
    ax.set_title(title)
    if path:
        fig.savefig(path)
    return fig


# ===========================================================================
# EXPORT
# ===========================================================================
def export(outdir, outcome, model_name, bundle, metrics, importance=None,
           blocks=None, best_params=None, figures=(), notes=""):
    """One run, one folder. Returns the folder path.

    The metrics file is one tidy row per fitted configuration, with the outcome,
    the model and the build hashes on every row, so the twelve runs concatenate
    into the comparison table without any further bookkeeping.
    """
    folder = os.path.join(outdir, outcome, model_name)
    os.makedirs(folder, exist_ok=True)
    stem = f"{outcome}_{model_name}"

    m = metrics.copy()
    m.insert(0, "outcome", outcome)
    m.insert(1, "family", model_name)
    m["n_train_plus_test"] = len(bundle["X"])
    m["n_predictors"] = bundle["X"].shape[1]
    m["prevalence"] = float(bundle["y"].mean())
    m["threshold_rule"] = "-".join(str(t) for t in bundle["threshold"])
    m["cohort"] = bundle["cohort"]
    m["data_sha"] = bundle["data_sha"]
    m["registry_sha"] = bundle["registry_sha"]
    m["run_utc"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    m.to_csv(os.path.join(folder, f"metrics_{stem}.csv"), index=False)

    if importance is not None:
        importance.to_csv(os.path.join(folder, f"importance_{stem}.csv"), index=False)
    if blocks is not None:
        blocks.to_csv(os.path.join(folder, f"block_importance_{stem}.csv"), index=False)
    if best_params is not None:
        with open(os.path.join(folder, f"best_params_{stem}.json"), "w") as f:
            json.dump(best_params, f, indent=2, default=str)

    for name, fig in figures:
        fig.savefig(os.path.join(folder, f"{name}_{stem}.png"))

    with open(os.path.join(folder, f"run_report_{stem}.txt"), "w", encoding="utf-8") as f:
        f.write(f"{outcome} | {model_name} | {m['run_utc'].iloc[0]} UTC\n")
        f.write(f"threshold {bundle['threshold']} ({bundle['threshold_authority']})\n")
        f.write(f"cohort {bundle['cohort']}, {len(bundle['X'])} rows, "
                f"{bundle['X'].shape[1]} predictors, prevalence "
                f"{bundle['y'].mean():.1%}\n")
        f.write(f"data sha {bundle['data_sha']} | registry sha {bundle['registry_sha']}\n\n")
        f.write(m.to_string(index=False))
        if blocks is not None:
            f.write("\n\nBlock importance\n" + blocks.to_string(index=False))
        if notes:
            f.write("\n\n" + notes + "\n")

    print(f"wrote {folder}")
    return folder


RUNS_REPO = "isasaade-23/pns2013-model-runs"


def push_results(folder, token=None, repo=RUNS_REPO, message=None,
                 workdir="/content/_runs", attempts=3):
    """Commit one run folder to the private run-output repository.

    The outputs are small — a metrics row, two importance tables, three
    figures, a text report — and belong somewhere that is neither a Drive
    mount nor a zip in the Downloads folder. `repo` is private, which is the
    point: these are unpublished results.

    The token comes from Colab's saved keys (`GITHUB_TOKEN`) or the
    environment. It is used to build the remote URL and is never written into
    the repository: the remote is rewritten to a tokenless URL before the
    function returns.

    Each run writes to its own path, so twelve notebooks pushing in parallel
    do not collide on content. They can still collide on the branch tip, so a
    rejected push rebases on the remote and tries again.
    """
    import shutil
    import subprocess

    token = token or os.environ.get("GITHUB_TOKEN")
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN")
        except Exception:
            pass
    if not token:
        raise RuntimeError(
            "no GITHUB_TOKEN. Add a fine-grained token with Contents: "
            f"read and write on {repo} to the Colab saved keys, named "
            "GITHUB_TOKEN, and enable it for this notebook.")

    auth = f"https://{token}@github.com/{repo}.git"
    clean = f"https://github.com/{repo}.git"

    def git(*args, cwd=workdir, check=True):
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        if check and r.returncode:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
        return r

    if not os.path.isdir(os.path.join(workdir, ".git")):
        shutil.rmtree(workdir, ignore_errors=True)
        r = subprocess.run(["git", "clone", "--depth", "1", auth, workdir],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"clone failed: {r.stderr.replace(token, '***').strip()}")
    git("remote", "set-url", "origin", auth)
    git("config", "user.email", "noreply@users.noreply.github.com")
    git("config", "user.name", "pns2013 model runs")

    rel = os.path.join("outputs", "models",
                       *os.path.normpath(folder).split(os.sep)[-2:])
    dest = os.path.join(workdir, rel)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(folder, dest)

    git("add", "-A")
    if not git("diff", "--cached", "--quiet", check=False).returncode:
        print("nothing changed, nothing pushed")
        git("remote", "set-url", "origin", clean)
        return None

    git("commit", "-m", message or f"Run: {rel.replace(os.sep, '/')}")
    for i in range(attempts):
        if git("push", "origin", "HEAD:main", check=False).returncode == 0:
            break
        print(f"push rejected, rebasing on the remote (attempt {i + 1})")
        git("pull", "--rebase", "origin", "main", check=False)
    else:
        git("remote", "set-url", "origin", clean)
        raise RuntimeError("could not push after rebasing; pull and retry by hand")

    git("remote", "set-url", "origin", clean)
    url = f"https://github.com/{repo}/tree/main/{rel.replace(os.sep, '/')}"
    print(f"pushed {rel.replace(os.sep, '/')}")
    print(url)
    return url


def zip_folder(folder, out_zip):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, os.path.dirname(folder)))
    print(f"wrote {out_zip}")
    return out_zip
