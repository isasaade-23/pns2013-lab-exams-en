"""Emit the twelve model notebooks. One model family x one outcome each.

Everything that is common lives in pipeline/pns_modelkit.py in the data repo;
these notebooks are the thin outer layer: configure, build, fit, report.
"""
import json
import os
import sys

REPO = "https://github.com/isasaade-23/pns2013-lab-exams-en.git"

OUTCOMES = {
    "hypertension": dict(
        title="Elevated measured blood pressure",
        definition="systolic `W00407` >= 140 **or** diastolic `W00408` >= 90 mmHg",
        authority="WHO 2021 and Diretrizes Brasileiras de Hipertensao Arterial 2020, identical at 140/90",
        gate="`Q002` (1 yes, 2 pregnancy-only -> yes, 3 no)",
        n=6346, prev="15.2%",
        alt="`THRESHOLD = (130, 80)` for the ACC/AHA 2017 cut",
        dropped="`dx_hypertension`, `med_hypertension_2w`, `last_bp_measure`",
        note=("`last_bp_measure` is dropped as condition-specific. Task 4 of the task "
              "list asks for it back; if it returns, `last_glucose_test` has to return "
              "symmetrically for diabetes. Open with the group."),
        num="10",
    ),
    "diabetes": dict(
        title="HbA1c at or above the diagnostic threshold",
        definition="`Z034` >= 6.5%",
        authority="SBD and WHO, HbA1c >= 6.5. PNS 2013 has no fasting glucose",
        gate="`Q030` (1 yes, 2 pregnancy-only -> yes, 3 no)",
        n=6832, prev="3.8%",
        alt="`THRESHOLD = (6.0,)` for the WHO/IEC prediabetes cut, or `(5.7,)` for ADA",
        dropped="`dx_diabetes`, `med_diabetes_2w`, `last_glucose_test`",
        note=("Two things are open. `Q030` code 2, 'only during pregnancy', is coded as "
              "diagnosed here to match the `Q002` treatment; the alternative reading is "
              "that a past gestational episode is a risk factor, not current diabetes. "
              "And insulin (`Q03402`) is absent from the medication block of the "
              "registry; it needs adding, and excluding from `n_medications` for this "
              "outcome. Neither is resolved in code."),
        num="20",
    ),
    "cholesterol": dict(
        title="Total cholesterol at or above the screening cut",
        definition="`Z031` >= 200 mg/dL",
        authority="Conventional screening cut. SBC uses risk-stratified LDL targets rather than one diagnostic value",
        gate="`Q060` (1 yes, 2 no)",
        n=5944, prev="32.0%",
        alt="`THRESHOLD = (190,)`; the LDL variant needs `Z033` and a registry change",
        dropped="`dx_cholesterol`",
        note=("PNS 2013 has no lipid-lowering medication item: `Q06204` records a "
              "recommendation, not use. This model therefore cannot define 'treated' "
              "the way the other two can, and the lipid definition itself is still open "
              "in the decision log: TC, LDL, HDL or any abnormal lipid give between 360 "
              "and 2,331 positives."),
        num="30",
    ),
}

MODELS = {
    "logreg": dict(
        label="Logistic Regression", short="logreg", idx=0,
        pip="optuna", scale=True, spline=True,
        why=("The champion family in the 06/08/2026 run (AUC 0.739) and the model the "
             "paper is written around: coefficients read directly, and age enters as a "
             "natural cubic spline because risk is not linear in the log-odds."),
        imports="from sklearn.linear_model import LogisticRegression",
        default_est="LogisticRegression(max_iter=3000, class_weight='balanced',\n                            solver='liblinear', random_state=mk.RS)",
        est_fn=("def make_est(params):\n"
                "    return LogisticRegression(max_iter=3000, class_weight='balanced',\n"
                "                              solver='liblinear', random_state=mk.RS, **params)"),
        space=("def suggest(t):\n"
               "    return {'C': t.suggest_float('C', 1e-3, 1e2, log=True),\n"
               "            'penalty': t.suggest_categorical('penalty', ['l1', 'l2'])}"),
        runtime="about 3 minutes at 30 trials, 10 at 100",
    ),
    "rf": dict(
        label="Random Forest", short="rf", idx=1,
        pip="optuna", scale=False, spline=False,
        why=("Non-linear baseline. No scaling and no spline: a tree splits on age "
             "directly, so the spline would only add collinear columns."),
        imports="from sklearn.ensemble import RandomForestClassifier",
        default_est="RandomForestClassifier(class_weight='balanced', n_jobs=-1,\n                                random_state=mk.RS)",
        est_fn=("def make_est(params):\n"
                "    return RandomForestClassifier(class_weight='balanced', n_jobs=-1,\n"
                "                                  random_state=mk.RS, **params)"),
        space=("_DEPTH = {'none': None, '6': 6, '10': 10, '16': 16, '24': 24}\n"
               "_FEATS = {'sqrt': 'sqrt', 'log2': 'log2', '0.3': 0.3, '0.5': 0.5}\n\n"
               "def suggest(t):\n"
               "    return {'n_estimators': t.suggest_int('n_estimators', 250, 600),\n"
               "            'max_depth': _DEPTH[t.suggest_categorical('max_depth', list(_DEPTH))],\n"
               "            'min_samples_leaf': t.suggest_int('min_samples_leaf', 1, 25),\n"
               "            'max_features': _FEATS[t.suggest_categorical('max_features', list(_FEATS))]}"),
        runtime="about 25 minutes at 30 trials, 80 at 100 — the slowest of the four",
    ),
    "xgboost": dict(
        label="XGBoost", short="xgboost", idx=2,
        pip="xgboost optuna", scale=False, spline=False,
        why=("Boosted trees, the third family of the 06/08/2026 comparison. Imbalance "
             "is handled with `scale_pos_weight` rather than resampling, which matters "
             "most for diabetes at 3.8% prevalence."),
        imports="from xgboost import XGBClassifier",
        default_est="XGBClassifier(scale_pos_weight=SPW, eval_metric='logloss',\n                       n_jobs=-1, random_state=mk.RS)",
        est_fn=("def make_est(params):\n"
                "    return XGBClassifier(scale_pos_weight=SPW, eval_metric='logloss',\n"
                "                         n_jobs=-1, random_state=mk.RS, **params)"),
        space=("def suggest(t):\n"
               "    return {'n_estimators': t.suggest_int('n_estimators', 200, 700),\n"
               "            'learning_rate': t.suggest_float('learning_rate', 0.01, 0.3, log=True),\n"
               "            'max_depth': t.suggest_int('max_depth', 2, 8),\n"
               "            'subsample': t.suggest_float('subsample', 0.6, 1.0),\n"
               "            'colsample_bytree': t.suggest_float('colsample_bytree', 0.5, 1.0),\n"
               "            'min_child_weight': t.suggest_int('min_child_weight', 1, 12),\n"
               "            'reg_lambda': t.suggest_float('reg_lambda', 0.1, 20, log=True)}"),
        runtime="about 12 minutes at 30 trials, 40 at 100",
    ),
    "tabpfn": dict(
        label="TabPFN", short="tabpfn", idx=3,
        pip="tabpfn-client", scale=False, spline=False,
        why=("A tabular foundation model: it is pre-trained and does one forward pass "
             "instead of a fit, so there is no hyper-parameter search to run. Task-list "
             "line 6 asks for it. Run through the Prior Labs API, which means the design "
             "matrix leaves the session; PNS 2013 is public IBGE microdata, so this is "
             "acceptable here and would not be for identifiable data."),
        imports="from tabpfn_client import TabPFNClassifier",
        default_est=None,
        est_fn=None, space=None,
        runtime="a few minutes, dominated by API round trips",
    ),
}


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").split("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n").split("\n")}


def _src(lines):
    """nbformat wants a list of strings, each ending in a newline except the last."""
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def build(outcome, model):
    o, m = OUTCOMES[outcome], MODELS[model]
    C = []

    # ---------------------------------------------------------------- header
    C.append(md(f"""
# {o['title']} — {m['label']} · PNS 2013

**FAPESP–Illinois undiagnosed NCD project** — Isabela Venancio da Silva (USP São Paulo) ·
Xuan Lin · Amogh Mannava (UIUC).

One model family, one outcome. Twelve notebooks share this shape, so any two of
them can be compared line for line.

| | |
|---|---|
| **Outcome** | {o['definition']} |
| **Threshold authority** | {o['authority']} |
| **Cohort** | adults who answer *no* to {o['gate']}, i.e. never told by a doctor they had this condition |
| **Expected build** | {o['n']:,} rows, prevalence {o['prev']} |
| **Model** | {m['label']} |
| **Dropped as condition-specific** | {o['dropped']} |

**Why this family.** {m['why']}

## What this notebook does not decide

Preprocessing is frozen. Every variable decision comes from
`PNS_preprocessing_registry_v1.xlsx` through `pns_preprocess.py`: 117 declared
variables in, 91 out, 28 dropped, 2 composites, 5 renamed. **To change a
variable, change the workbook, not this notebook.**

Layer 1 (`build_matrix`) runs once and is deterministic. Layer 2
(`build_preprocessor`) — imputation, splines, encoding, scaling — is fitted
inside each cross-validation fold and never sees the test rows.

## Open with the group

{o['note']}

`n_medications` and `n_chronic` are sums over their blocks and are rebuilt by
`build_matrix` **after** the condition-specific drop. This is the trap that once
produced an AUC of 0.86 with sensitivity 1.000; do not compute either counter
anywhere in this notebook.

## Three findings from testing the shared build, for Amogh

Reproducing the 15/09/2026 numbers locally turned up three things in the files as
they stand on the Drive. None is worked around silently; all three are in
`pns_modelkit.py`, documented at the point of use.

1. **The diagnosis gates are still imputed.** `dx_diabetes` and `dx_cholesterol`
   carry `na_rule = implied:0` in the registry, so an unanswered gate is filled
   with zero and the row is declared undiagnosed. That is bug 1 of the build
   note, and it builds 7,851 and 7,244 rows instead of 6,832 and 5,944. Setting
   both cells to `mar`, as `dx_hypertension` already is, reproduces the reported
   cohorts exactly. Section 2 stops with this message if it happens.
2. **One-hot columns and their declared categories disagree on type.**
   `_roles()` declares the categories as strings and leaves the columns numeric,
   so `OneHotEncoder` refuses to fit at all. Cast back to string before the
   encoder.
3. **Three ordinal orders are declared descending.** `passive_smoke`,
   `diet_salt_perception` and `smoking_status` were reverse-coded in the values
   by `_recode_fixes`, but the registry still records the pre-reversal order, and
   `OrdinalEncoder` rejects unsorted numeric categories. Sorted ascending, which
   is what 'higher means more exposure' means after that recode.
"""))

    # ----------------------------------------------------------------- setup
    C.append(md("""
---
# 1 · Setup

Installs, the data, and the run switches. Nothing here touches the science,
except section 1.3, where the research question is chosen.
"""))

    C.append(md(f"""
## 1.1 · Install

**What this does.** Installs what Colab does not ship for this family: `{m['pip']}`,
plus `openpyxl` for the workbook.

**What to look for.** Nothing, unless it errors.
"""))
    C.append(code(f"!pip install -q {m['pip']} openpyxl"))

    C.append(md("""
## 1.2 · Data and shared code

**What this does.** Clones the public repository, which holds the survey file,
both dictionaries, the frozen preprocessing (`pns_preprocess.py` and the
registry workbook) and the shared model helper (`pns_modelkit.py`).

**Drive is off by default**, since mounting asks for permission every session.
The run then writes to the session disk and zips itself at the end. Set
`MOUNT_DRIVE = True` to write into the shared folder instead, which also keeps
the built matrix between sessions and saves the minute it takes to rebuild.
Leave the `DRIVE = ...` line in place either way: the configuration cell reads
it whether or not anything is mounted.

**What to look for.** The two sha256 stamps printed by the build in section 2
must match across notebooks. They are what proves twelve runs used one matrix.
"""))
    C.append(code(f"""
import os, subprocess, sys

REPO_URL = "{REPO}"
REPO     = "/content/pns2013-lab-exams-en"

if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
sys.path.insert(0, os.path.join(REPO, "pipeline"))

DATA     = os.path.join(REPO, "data", "pns2013_lab_exams.xlsx")
REGISTRY = os.path.join(REPO, "pipeline", "PNS_preprocessing_registry_v1.xlsx")

# Drive is optional and off by default, because mounting asks for permission
# every session. Set MOUNT_DRIVE = True to write straight into the shared
# folder and to keep the built matrix between sessions; left False, the run
# writes to the session disk and zips itself at the end.
#
# DRIVE is defined either way. Do not comment this line out: the configuration
# cell below reads it.
MOUNT_DRIVE = False
DRIVE = "/content/drive/MyDrive/FAPESP_Illinois"

if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive not mounted:", e)

print("data    ", os.path.exists(DATA))
print("registry", os.path.exists(REGISTRY))
"""))

    C.append(md(f"""
## 1.3 · Configuration

**What this does.** Replaces what used to be command-line flags. Everything
downstream reads these.

| Switch | Meaning |
|---|---|
| `THRESHOLD` | `None` uses the guideline cut. The prespecified sensitivity analysis is {o['alt']} |
| `COHORT` | `"undiagnosed"` is the primary framing (Model A). `"all"` is diagnostic only |
| `N_TRIALS` | tuning budget. 30 finishes inside a free Colab session; `FULL = True` raises it to 100 |
| `RS` | 42, fixed, so the split is identical across the twelve notebooks |

**Runtime.** {m['runtime']}.
"""))

    tuned_cfg = ("FULL     = False          # True -> 100 trials, the paper run\n"
                 "N_TRIALS = 100 if FULL else 30\n") if model != "tabpfn" else ""
    C.append(code(f"""
import numpy as np, pandas as pd
import pns_modelkit as mk
import pns_preprocess as pp

OUTCOME   = "{outcome}"
MODEL     = "{m['short']}"
THRESHOLD = None          # None = guideline default; see the table above
COHORT    = "undiagnosed"
{tuned_cfg}
# outputs: the shared folder when Drive is mounted, the session disk otherwise
BASE   = f"{{DRIVE}}/02_analysis" if os.path.isdir(DRIVE) else "/content/work"
OUTDIR = os.path.join(BASE, "outputs", "models")
BUILD  = os.path.join(BASE, "outputs", "matrices")
os.makedirs(OUTDIR, exist_ok=True); os.makedirs(BUILD, exist_ok=True)

print("writing to", OUTDIR)
"""))

    # ----------------------------------------------------------------- build
    C.append(md(f"""
---
# 2 · The frozen matrix

**What this does.** Calls `build_matrix()` from `pns_preprocess.py` — or reuses
a cached build whose data and registry hashes still match — then asserts the
counts reported to the group: **{o['n']:,} rows at {o['prev']}**.

**What to look for.** If the assertion fails, stop. It means the registry or the
survey file moved, and no result from this notebook is comparable to the others
until that is understood.
"""))
    C.append(code("""
bundle = mk.load_or_build(OUTCOME, data=DATA, registry=REGISTRY,
                          outdir=BUILD, threshold=THRESHOLD, cohort=COHORT)
mk.check_frozen(bundle)

X, y = bundle["X"], bundle["y"]
print(f"\\n{X.shape[0]} rows x {X.shape[1]} predictors, prevalence {y.mean():.1%}")
print("threshold authority:", bundle["threshold_authority"])
print("data sha", bundle["data_sha"], "| registry sha", bundle["registry_sha"])

a = mk.attrition(bundle, BUILD)
display(a) if a is not None else None
"""))

    C.append(md("""
**Participant flow and roles.** The attrition table above is the row accounting
for the flow diagram. Below, where each column enters Layer 2.
"""))
    C.append(code("""
for k, v in bundle["roles"].items():
    print(f"{k:<10} {len(v):>3}  {', '.join(v[:6])}{' ...' if len(v) > 6 else ''}")

blocks = pd.Series({c: bundle["spec"][c]["block"]
                    for c in X.columns if c in bundle["spec"]}).value_counts()
print("\\npredictors per block\\n", blocks.to_string())
"""))

    # ----------------------------------------------------------------- split
    C.append(md(f"""
---
# 3 · Split and preprocessor

**What this does.** Stratified 80/20 at `random_state=42`, then builds the Layer 2
`ColumnTransformer`. For {m['label']}: scaling **{'on' if m['scale'] else 'off'}**,
age spline **{'on' if m['spline'] else 'off'}**.

**What to look for.** Train and test prevalence should match to a decimal. The
preprocessor is passed *into* the pipeline, never fitted here.
"""))
    C.append(code(f"""
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

Xtr, Xte, ytr, yte = mk.split(bundle)
SPW = mk.pos_weight(ytr)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=mk.RS)

def preproc():
    return mk.preprocessor(bundle, spline={m['spline']}, scale={m['scale']})

n_encoded = mk.preprocessor(bundle, spline={m['spline']}, scale={m['scale']},
                            verbose=True).fit_transform(Xtr).shape[1]
print(f"{{X.shape[1]}} raw predictors -> {{n_encoded}} encoded columns")
print(f"positives: train {{int(ytr.sum())}}, test {{int(yte.sum())}} "
      f"(scale_pos_weight {{SPW:.1f}})")
"""))

    # ------------------------------------------------------------------- fit
    if model != "tabpfn":
        C.append(md(f"""
---
# 4 · Fit

Two fits, reported side by side: library defaults, and an Optuna TPE search over
5-fold AUC. The comparison is the honest way to say whether tuning bought
anything — in the hypertension run it was worth about +0.006 AUC.
"""))
        C.append(md(f"""
## 4.1 · Defaults

**What to look for.** The CV AUC here is the floor. For hypertension it should
land near 0.730; the tuned run near 0.736.
"""))
        C.append(code(f"""
{m['imports']}

{m['est_fn']}

pipe_def = Pipeline([("prep", preproc()),
                     ("clf", {m['default_est']})])

cv_def = cross_val_score(pipe_def, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1)
print(f"default 5-fold CV AUC {{cv_def.mean():.3f}} +/- {{cv_def.std():.3f}}")

pipe_def.fit(Xtr, ytr)
row_def, p_def = mk.evaluate(pipe_def, Xte, yte, "{m['label']} (default)")
print(f"test AUC {{row_def['AUC_test']:.3f}} "
      f"[{{row_def['AUC_lo']:.3f}}, {{row_def['AUC_hi']:.3f}}]")
"""))

        C.append(md(f"""
## 4.2 · Tuned

**The slow cell.** {m['runtime'].capitalize()}. The objective reports fold by
fold so the pruner can stop a hopeless trial early. The search space is the one
used in the 07/08/2026 revision, unchanged.
"""))
        C.append(code(f"""
import optuna
from sklearn.metrics import roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

{m['space']}

def objective(trial):
    params, scores = suggest(trial), []
    for k, (itr, iva) in enumerate(cv.split(Xtr, ytr)):
        pipe = Pipeline([("prep", preproc()), ("clf", make_est(params))])
        pipe.fit(Xtr.iloc[itr], ytr.iloc[itr])
        scores.append(roc_auc_score(ytr.iloc[iva],
                                    pipe.predict_proba(Xtr.iloc[iva])[:, 1]))
        trial.report(float(np.mean(scores)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=mk.RS),
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_params
print(f"\\nbest CV AUC {{study.best_value:.3f}} after {{len(study.trials)}} trials")
print(best)

pipe_tuned = Pipeline([("prep", preproc()),
                       ("clf", make_est(suggest(optuna.trial.FixedTrial(best))))])
pipe_tuned.fit(Xtr, ytr)
row_tuned, p_tuned = mk.evaluate(pipe_tuned, Xte, yte, "{m['label']} (tuned)")
row_def["CV_AUC"], row_tuned["CV_AUC"] = cv_def.mean(), study.best_value
BEST_MODEL, BEST_P, BEST_PARAMS = pipe_tuned, p_tuned, best
"""))
    else:
        C.append(md("""
---
# 4 · Fit

TabPFN is pre-trained: there is no hyper-parameter search, and a "fit" is the
cost of uploading the training rows. Two configurations are reported — the
default single forward pass and a larger ensemble — which is the TabPFN
equivalent of the default-versus-tuned comparison the other notebooks run.

**The token.** Stored in Colab's saved keys as `TABPFN_TOKEN`, from
[platform.priorlabs.ai/account/api-keys](https://platform.priorlabs.ai/account/api-keys).
The cell reads it from `userdata` and falls back to the environment variable.

**What leaves the session.** The encoded design matrix, sent to the Prior Labs
API. Public IBGE microdata, so acceptable here. The offline alternative — the
open-weights `tabpfn` package on a Colab GPU — is in the commented block at the
bottom of the cell.
"""))
        C.append(code("""
import os
try:
    from google.colab import userdata
    os.environ["TABPFN_TOKEN"] = userdata.get("TABPFN_TOKEN")
except Exception as e:
    print("no Colab saved key found, falling back to the environment:", e)

assert os.environ.get("TABPFN_TOKEN"), "TABPFN_TOKEN is not set"

import tabpfn_client
tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
from tabpfn_client import TabPFNClassifier
print("tabpfn-client ready; model_path='auto' uses the current release")
"""))
        C.append(code("""
from sklearn.pipeline import Pipeline

def make_tabpfn(**kw):
    \"\"\"balance_probabilities matters here: 3.8% prevalence for diabetes.
    Older client versions do not take it, so fall back rather than fail.\"\"\"
    try:
        return TabPFNClassifier(model_path="auto", balance_probabilities=True,
                                random_state=mk.RS, **kw)
    except TypeError:
        return TabPFNClassifier(model_path="auto", random_state=mk.RS, **kw)

pipe_def = Pipeline([("prep", preproc()), ("clf", make_tabpfn())])
pipe_def.fit(Xtr, ytr)
row_def, p_def = mk.evaluate(pipe_def, Xte, yte, "TabPFN (default)")
print(f"default  test AUC {row_def['AUC_test']:.3f} "
      f"[{row_def['AUC_lo']:.3f}, {row_def['AUC_hi']:.3f}]")

pipe_tuned = Pipeline([("prep", preproc()), ("clf", make_tabpfn(n_estimators=8))])
pipe_tuned.fit(Xtr, ytr)
row_tuned, p_tuned = mk.evaluate(pipe_tuned, Xte, yte, "TabPFN (ensemble 8)")
print(f"ensemble test AUC {row_tuned['AUC_test']:.3f} "
      f"[{row_tuned['AUC_lo']:.3f}, {row_tuned['AUC_hi']:.3f}]")

BEST_MODEL, BEST_P, BEST_PARAMS = pipe_tuned, p_tuned, {"n_estimators": 8}

# Offline alternative, no data leaves the session, needs a GPU runtime:
#   !pip install -q tabpfn
#   from tabpfn import TabPFNClassifier as LocalTabPFN
#   pipe = Pipeline([("prep", preproc()), ("clf", LocalTabPFN(device="cuda"))])
"""))
        C.append(md("""
**A 5-fold CV score, for comparability with the other three notebooks.** Five
more round trips; skip it if the API budget is tight.
"""))
        C.append(code("""
RUN_CV = True

if RUN_CV:
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(Pipeline([("prep", preproc()), ("clf", make_tabpfn())]),
                                Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"5-fold CV AUC {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    row_def["CV_AUC"] = cv_scores.mean()
"""))

    # --------------------------------------------------------------- results
    C.append(md("""
---
# 5 · Results

Measured on the test half, which nothing above was fitted on.
"""))
    C.append(md("""
## 5.1 · Metrics

**What to look for.** `AUC_lo`/`AUC_hi` is a 1,000-draw stratified bootstrap
interval on the test AUC. Two operating points are reported: Youden's J, and the
screening point that holds sensitivity at about 90% — the one a screening
instrument would actually be set at, and the one where PPV shows what the
prevalence costs.
"""))
    C.append(code("""
results = mk.metrics_frame([row_def, row_tuned])
display(results)
"""))

    perm_args = ("n_repeats=3, top=20" if model == "tabpfn" else "n_repeats=10")
    perm_note = ("Restricted to the 20 strongest features with 3 repeats: every "
                 "permutation is a billed API round trip."
                 if model == "tabpfn" else
                 "Ten repeats over every predictor.")
    C.append(md(f"""
## 5.2 · Permutation importance

**What this does.** Shuffles one predictor at a time in the test set and measures
how far AUC falls, then sums the drops per registry block. {perm_note}

**What to look for.** The block table is the null-result table of the decision
log: for hypertension, access to care contributed exactly 0.000 and sleep less
than that. A block that suddenly matters for a new outcome is a finding; a block
that matters *too much* is usually leakage, and the first thing to check is
whether a counter was rebuilt.
"""))
    C.append(code(f"""
imp, blocks_imp = mk.permutation_report(BEST_MODEL, Xte, yte, bundle, {perm_args})
display(imp.head(20))
display(blocks_imp)
"""))

    C.append(md("""
## 5.3 · Figures

ROC for both configurations, calibration of the better one, and the top 20
predictors. Written at 300 dpi.
"""))
    C.append(code("""
fig_roc = mk.plot_roc({row_def["model"]: (yte, p_def),
                       row_tuned["model"]: (yte, p_tuned)},
                      f"{OUTCOME} · {row_tuned['model']}")
fig_cal = mk.plot_calibration(yte, BEST_P, f"{OUTCOME} · calibration")
fig_imp = mk.plot_importance(imp, f"{OUTCOME} · permutation importance")
"""))

    # ---------------------------------------------------------------- export
    C.append(md("""
---
# 6 · Export

One folder per run, under `outputs/models/<outcome>/<family>/`: the metrics row,
the importance tables, the best parameters, the figures and a plain-text report
carrying the build hashes. The metrics files from the twelve runs concatenate
into the comparison table without further bookkeeping.
"""))
    C.append(code("""
folder = mk.export(OUTDIR, OUTCOME, MODEL, bundle, results,
                   importance=imp, blocks=blocks_imp, best_params=BEST_PARAMS,
                   figures=[("roc", fig_roc), ("calibration", fig_cal),
                            ("importance", fig_imp)])

if not os.path.isdir(DRIVE):
    z = mk.zip_folder(folder, f"/content/{OUTCOME}_{MODEL}.zip")
    try:
        from google.colab import files
        files.download(z)
    except Exception as e:
        print(e)
"""))

    nb = {"cells": C,
          "metadata": {"colab": {"provenance": [], "toc_visible": True},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    for c in nb["cells"]:
        c["source"] = _src(c["source"])
    return nb


def main(root):
    for outcome, o in OUTCOMES.items():
        d = os.path.join(root, outcome)
        os.makedirs(d, exist_ok=True)
        for model, m in MODELS.items():
            num = int(o["num"]) + m["idx"]
            path = os.path.join(d, f"{num}_{outcome}_{m['short']}.ipynb")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(build(outcome, model), f, indent=1, ensure_ascii=False)
            print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models")
