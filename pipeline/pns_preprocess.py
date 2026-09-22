"""
=============================================================================
PNS 2013 | ONE PREPROCESSING FOR THREE OUTCOMES
=============================================================================
Hypertension, diabetes and cholesterol run through the same build. Nothing in
this file is outcome-specific except what the OUTCOMES table below declares.

Two layers, and the line between them is the point of the file.

  LAYER 1  build_matrix()   deterministic, outcome-free, runs once.
                            Load, filter rows, build the outcome, clean
                            sentinels, recode, derive, apply tier-1 missing
                            rules, build counters, subset, drop
                            condition-specific columns, rebuild counters,
                            freeze.

  LAYER 2  build_preprocessor()  fitted inside cross-validation, per fold.
                            Imputation, splines, encoding, centring, scaling.
                            Nothing here ever sees the test rows.

The registry workbook is the single source of truth. This file reads it. Do
not hard-code a variable decision here; change the workbook instead.

Usage
    python pns_preprocess.py --outcome hypertension
    python pns_preprocess.py --outcome diabetes --threshold 6.0
    python pns_preprocess.py --outcome cholesterol --cohort all
    python pns_preprocess.py --outcome hypertension --keep-condition-specific

Outputs, into --outdir:
    matrix_<outcome>_<cohort>_<threshold>.pkl   X, y, roles, categories, spec
    attrition_<...>.csv                         row-level participant flow
    build_report_<...>.txt                      everything that was done
=============================================================================
"""

import argparse, hashlib, json, os, pickle, sys
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   SplineTransformer, StandardScaler)

RS = 42

# ===========================================================================
# OUTCOME DECLARATIONS
# ===========================================================================
# threshold_authority is recorded so the methods section can cite it without
# anyone having to re-derive where the number came from.
OUTCOMES = {
    "hypertension": dict(
        sources=["W00407", "W00408"],
        rule=lambda d, t: ((d["W00407"] >= t[0]) | (d["W00408"] >= t[1])).astype(int),
        default_threshold=(140, 90),
        alt_thresholds={"acc_aha_2017": (130, 80)},
        threshold_authority="WHO 2021 and Diretrizes Brasileiras de Hipertensao Arterial 2020, identical at 140/90",
        gate="Q002", gate_map={1: 1, 2: 1, 3: 0},
        condition_specific=["dx_hypertension", "med_hypertension_2w", "last_bp_measure"],
        label="Elevated measured blood pressure",
    ),
    "diabetes": dict(
        sources=["Z034"],
        rule=lambda d, t: (d["Z034"] >= t[0]).astype(int),
        default_threshold=(6.5,),
        alt_thresholds={"who_iec_prediabetes": (6.0,), "ada_prediabetes": (5.7,)},
        threshold_authority="SBD 2026 and WHO, HbA1c >= 6.5. PNS 2013 has no fasting glucose",
        gate="Q030", gate_map={1: 1, 2: 1, 3: 0},   # note 2: pregnancy-only coded as diagnosed, per Q002
        condition_specific=["dx_diabetes", "med_diabetes_2w", "last_glucose_test"],
        label="HbA1c at or above threshold",
    ),
    "cholesterol": dict(
        sources=["Z031"],
        rule=lambda d, t: (d["Z031"] >= t[0]).astype(int),
        default_threshold=(200,),
        alt_thresholds={"tc_190": (190,)},
        threshold_authority="Conventional screening cut. SBC uses risk-stratified LDL targets rather than one diagnostic value",
        gate="Q060", gate_map={1: 1, 2: 0},
        condition_specific=["dx_cholesterol"],
        label="Total cholesterol at or above threshold",
    ),
}

# ===========================================================================
# SENTINELS AND PLAUSIBLE RANGES
# ===========================================================================
# Anything outside the range becomes NaN and is counted in the build report.
# The dictionary defines 999 as Error for the measured variables. None are
# present in the 2013 file, but a guard that only fires on bad data is the
# only kind worth having.
RANGES_RAW = {                       # applied to the source file, before anything is built
    "W00407": (60, 290), "W00408": (30, 200), "W00303": (20, 210),
    "z004": (20, 300), "z005": (100, 220),
    "Z031": (50, 600), "Z033": (10, 400), "Z034": (2, 20),
    "P029": (0, 40),                 # doses per drinking day; source allows 50
}
RANGES_X = {                         # applied to derived columns, after the build
    "wait_minutes": (0, 1440), "consult_minutes": (0, 600),
    "exercise_min_week": (0, 2520),  # 7 days x 6 hours
}
SENTINEL_999 = ["W00407", "W00408", "W00303", "z004", "z005"]

# race: 9 = Ignored. Levels 3 (Yellow) and 5 (Indigenous) are too thin to
# stand alone. Collapse declared here, never left to min_frequency.
RACE_COLLAPSE = {1: "white", 2: "black", 4: "brown", 3: "other", 5: "other", 9: np.nan}


# ===========================================================================
# LAYER 1
# ===========================================================================
def _yn(s):
    """PNS 1/2 yes-no. Codes 3 and 9 (don't know / ignored) become NaN."""
    return s.replace({1: 1, 2: 0, 3: np.nan, 9: np.nan})


def _sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def load_registry(path):
    reg = pd.read_excel(path, sheet_name="registry")
    reg = reg.fillna({"level_order": "", "from_name": "", "source_code": ""})
    return reg


def _derive(X, raw):
    """Engineered variables. Intermediates that the registry drops (income,
    weight_kg, height_cm, alcohol_days, alcohol_doses) are built here and
    removed at the end of the build, not skipped."""
    inc = ["E01602", "E01604", "E01802", "E01804", "F00102", "F00702", "F00802"]
    X["_income"] = raw[inc].fillna(0).sum(axis=1)
    X["log_income"] = np.log1p(X["_income"])

    X["_weight_kg"] = raw["z004"]
    X["_height_cm"] = raw["z005"]
    X["bmi"] = X["_weight_kg"] / (X["_height_cm"] / 100) ** 2

    X["_alcohol_days"] = raw["P028"]
    X["_alcohol_doses"] = raw["P029"]
    X["alcohol_drinks_week"] = X["_alcohol_days"] * X["_alcohol_doses"]

    X["exercise_min_week"] = ((raw["P03701"].fillna(0) * 60 + raw["P03702"].fillna(0))
                              * raw["P035"].fillna(0))
    X["wait_minutes"] = raw["X01401"] * 60 + raw["X01402"]
    X["consult_minutes"] = raw["X01501"] * 60 + raw["X01502"]

    disc = [f"X025{i:02d}" for i in range(1, 11)]
    X["n_discrimination"] = raw[disc].replace({1: 1, 2: 0, 3: np.nan}).sum(axis=1)

    # qual_index: mean of four 5-point ratings, complete raters only. Someone
    # who never consulted has no value, which is a real not-applicable and is
    # handled by the explicit_na rule, not by a zero.
    qual = ["X02001", "X02004", "X02201", "X02203"]
    q = raw[qual].replace({9: np.nan})
    X["qual_index"] = q.mean(axis=1).where(q.notna().all(axis=1))
    return X


def _recode_fixes(X, raw):
    """Codebook corrections. Each one is a factual error in the previous
    build, not a modelling preference."""
    log = []

    # P045 code 8 is "does not watch television", not the top of the scale.
    if "tv_hours" in X:
        n8 = int((X["tv_hours"] == 8).sum())
        X["tv_hours"] = X["tv_hours"].replace({8: 0})
        log.append(f"tv_hours: {n8} rows with source code 8 (does not watch TV) recoded to 0")

    # race: declared collapse, never min_frequency
    if "race" in X:
        X["race"] = X["race"].map(RACE_COLLAPSE)
        log.append("race: collapsed to white / black / brown / other, code 9 -> NaN")

    # Three ordinals run from most to least exposure in the source. Reversed
    # here so that "higher" means "more" everywhere and coefficient signs read
    # in the expected direction. The registry records the new order.
    for col, top in (("passive_smoke", 6), ("diet_salt_perception", 6),
                     ("smoking_status", 4)):
        if col in X.columns:
            X[col] = top - X[col]
            log.append(f"{col}: reverse-coded ({top} minus source), high now means more")

    return X, log


def build_matrix(data_path, registry_path, outcome, threshold=None,
                 cohort="undiagnosed", keep_condition_specific=False,
                 outdir="outputs"):
    cfg = OUTCOMES[outcome]
    thr = tuple(threshold) if threshold else cfg["default_threshold"]
    reg = load_registry(registry_path)
    rep, attr = [], []

    def say(s):
        print(s)
        rep.append(s)

    say("=" * 78)
    say(f"PNS 2013 preprocessing | outcome={outcome} threshold={thr} cohort={cohort}")
    say(f"built {datetime.now():%Y-%m-%d %H:%M} | data sha256[:16] {_sha(data_path)}")
    say(f"registry sha256[:16] {_sha(registry_path)}")
    say(f"threshold authority: {cfg['threshold_authority']}")
    say("=" * 78)

    # ---- step 1: load and filter rows -------------------------------------
    raw = pd.read_excel(data_path)
    attr.append(("rows in file", len(raw)))

    for c in SENTINEL_999:
        if c in raw.columns:
            n = int((raw[c] == 999).sum())
            if n:
                raw[c] = raw[c].replace({999: np.nan})
                say(f"  sentinel: {c} had {n} values of 999 (Error), set to NaN")

    for col, (lo, hi) in RANGES_RAW.items():
        if col in raw.columns:
            bad = int(((raw[col] < lo) | (raw[col] > hi)).sum())
            if bad:
                raw.loc[(raw[col] < lo) | (raw[col] > hi), col] = np.nan
                say(f"  range guard: {col} had {bad} values outside [{lo}, {hi}], set to NaN")

    raw = raw[raw["P005"] != 1].copy()
    attr.append(("after excluding pregnancy (P005=1)", len(raw)))

    have = raw[cfg["sources"]].notna().all(axis=1)
    raw = raw[have].copy()
    attr.append((f"after requiring {', '.join(cfg['sources'])}", len(raw)))

    # ---- step 2: outcome ---------------------------------------------------
    raw["_y"] = cfg["rule"](raw, thr)
    say(f"\nSTEP 2 | outcome '{cfg['label']}' at {thr}: "
        f"{raw['_y'].mean():.1%} of {len(raw)} adults")

    # ---- step 3: build declared variables ---------------------------------
    X = pd.DataFrame(index=raw.index)
    for _, s in reg.iterrows():
        n, code, rc = s["final_name"], str(s["source_code"]), s["recode"]
        if rc == "derived" or code == "" or code not in raw.columns:
            continue
        X[n] = _yn(raw[code]) if rc == "yes_no" else raw[code]

    # diagnosis gates have their own coding and must not go through _yn
    for oc, c in OUTCOMES.items():
        nm = {"hypertension": "dx_hypertension", "diabetes": "dx_diabetes",
              "cholesterol": "dx_cholesterol"}[oc]
        if c["gate"] in raw.columns:
            X[nm] = raw[c["gate"]].replace(c["gate_map"])

    X = _derive(X, raw)
    X, fixlog = _recode_fixes(X, raw)
    for line in fixlog:
        say("  " + line)

    # ---- step 4: range guards ---------------------------------------------
    for col, (lo, hi) in RANGES_X.items():
        if col in X.columns:
            bad = ((X[col] < lo) | (X[col] > hi)).sum()
            if bad:
                X.loc[(X[col] < lo) | (X[col] > hi), col] = np.nan
                say(f"  range guard: {col} had {int(bad)} values outside [{lo}, {hi}], set to NaN")

    # ---- step 5: tier-1 missingness ---------------------------------------
    # (a) implied skip: the skip pattern determines the answer, so impute it.
    # (b) explicit not-applicable: the question has no defined answer, so it
    #     becomes its own level (categorical) or a flag plus a value that
    #     Layer 2 centres among responders (numeric). The old 0-fill plus
    #     flag was exactly collinear and is gone.
    n_imp = n_exp = 0
    for _, s in reg.iterrows():
        n, rule, vt = s["final_name"], s["na_rule"], s["vtype"]
        if n not in X.columns:
            continue
        if isinstance(rule, str) and rule.startswith("implied:"):
            X[n] = X[n].fillna(float(rule.split(":")[1]))
            n_imp += 1
        elif rule == "explicit_na":
            if vt == "num" or s["recode"] == "derived":
                X[f"{n}_isNA"] = X[n].isna().astype(int)
                n_exp += 1
            else:
                X[n] = X[n].astype("object").where(X[n].notna(), "NA_nao_aplicavel")
                n_exp += 1
    say(f"\nSTEP 5 | tier 1: {n_imp} implied-skip imputations, "
        f"{n_exp} explicit not-applicable variables")

    # ---- step 6: counters --------------------------------------------------
    MED = [s["final_name"] for _, s in reg.iterrows()
           if str(s["in_n_medications"]) == "yes"]
    DX = [s["final_name"] for _, s in reg.iterrows()
          if str(s["in_n_chronic"]) == "yes"]

    def rebuild_counters(df, dropped):
        """n_medications and n_chronic are sums over their blocks. Any column
        dropped from the block must leave the sum, or it leaks straight back
        in. This has happened once already."""
        meds = [c for c in MED if c in df.columns and c not in dropped]
        dxs = [c for c in DX if c in df.columns and c not in dropped]
        df["n_medications"] = df[meds].sum(axis=1)
        df["n_chronic"] = df[dxs].sum(axis=1)
        assert not (set(meds) | set(dxs)) & set(dropped), \
            "counter rebuilt over a dropped column"
        return df, meds, dxs

    X, _, _ = rebuild_counters(X, set())

    # ---- step 7: cohort subset --------------------------------------------
    gate_col = {"hypertension": "dx_hypertension", "diabetes": "dx_diabetes",
                "cholesterol": "dx_cholesterol"}[outcome]
    if cohort == "undiagnosed":
        undef = int(X[gate_col].isna().sum())
        keep = X[gate_col] == 0
        X, raw = X[keep].copy(), raw[keep].copy()
        attr.append((f"{gate_col} missing, cohort undefined", -undef))
        attr.append(("undiagnosed cohort", len(X)))
        say(f"\nSTEP 7 | cohort='undiagnosed': {len(X)} adults never told they had "
            f"this condition ({undef} rows had an undefined gate)")
    else:
        attr.append(("full cohort", len(X)))
        say(f"\nSTEP 7 | cohort='all': {len(X)} adults")

    y = raw.loc[X.index, "_y"]
    weights = raw.loc[X.index, "peso_lab"] if "peso_lab" in raw.columns else None

    # ---- step 8: condition-specific columns -------------------------------
    # Leakage is per outcome. dx_diabetes is a legitimate predictor of blood
    # pressure and the gate of the diabetes model. One tier column cannot
    # express that, so the registry carries three.
    leak_col = f"leak_{outcome}"
    tier1 = [c for c in X.columns if c in cfg["condition_specific"]]
    if keep_condition_specific:
        say(f"\nSTEP 8 | --keep-condition-specific: retained {tier1}. Diagnostic only.")
        tier1 = []
    else:
        X = X.drop(columns=tier1)
        say(f"\nSTEP 8 | dropped as condition-specific for {outcome}: {tier1}")

    X, meds, dxs = rebuild_counters(X, set(tier1))
    say(f"  counters rebuilt from {len(meds)} medication and {len(dxs)} diagnosis columns")

    # ---- step 9: tidy ------------------------------------------------------
    X = X.drop(columns=[c for c in X.columns if c.startswith("_")])
    declared = set(reg["final_name"]) | {f"{n}_isNA" for n in reg["final_name"]}
    extra = [c for c in X.columns if c not in declared]
    assert not extra, f"undeclared columns reached the matrix: {extra}"

    tier2 = [c for c in X.columns
             if c in set(reg.loc[reg[leak_col] == 2, "final_name"])]
    say(f"  tier-2 (requires physical examination), kept and flagged: {tier2}")

    # ---- step 10: roles and freeze ----------------------------------------
    roles, cats = _roles(X, reg)
    say(f"\nSTEP 10 | final matrix: {X.shape[0]} rows x {X.shape[1]} predictors, "
        f"outcome prevalence {y.mean():.1%}")
    for k, v in roles.items():
        say(f"    {k:<10} {len(v):>3}")

    os.makedirs(outdir, exist_ok=True)
    tag = f"{outcome}_{cohort}_{'-'.join(str(t) for t in thr)}"
    spec = reg.set_index("final_name").to_dict("index")
    bundle = dict(X=X, y=y, weights=weights, roles=roles, categories=cats,
                  spec=spec, outcome=outcome, threshold=thr, cohort=cohort,
                  threshold_authority=cfg["threshold_authority"],
                  data_sha=_sha(data_path), registry_sha=_sha(registry_path))
    with open(os.path.join(outdir, f"matrix_{tag}.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    a = pd.DataFrame(attr, columns=["step", "n"])
    a.to_csv(os.path.join(outdir, f"attrition_{tag}.csv"), index=False)
    with open(os.path.join(outdir, f"build_report_{tag}.txt"), "w") as f:
        f.write("\n".join(rep) + "\n")
    say(f"\nwrote matrix_{tag}.pkl, attrition_{tag}.csv, build_report_{tag}.txt")
    return bundle


def _roles(X, reg):
    """Route every column to a Layer 2 branch. Categories are the domain of
    the instrument, taken from the codebook where the registry declares an
    order and from the observed values otherwise. They are a property of the
    questionnaire, not a parameter fitted to the sample, so fixing them here
    rather than per fold is not leakage."""
    by = reg.set_index("final_name")
    spline, ordinal, nominal, numeric, expl_num, expl_bin = [], [], [], [], [], []
    cats = {}

    for c in X.columns:
        base = c[:-5] if c.endswith("_isNA") else c
        if c.endswith("_isNA"):
            numeric.append(c)
            continue
        if base not in by.index:
            numeric.append(c)
            continue
        s = by.loc[base]
        if c == "age":
            spline.append(c)
        elif s["recode"] == "ordinal":
            order = [x for x in str(s["level_order"]).split(",") if x != ""]
            obs = sorted(pd.to_numeric(X[c], errors="coerce").dropna().unique())
            cats[c] = [float(x) for x in order] if order else list(obs)
            ordinal.append(c)
        elif s["recode"] == "nominal":
            cats[c] = sorted(X[c].dropna().astype(str).unique().tolist())
            nominal.append(c)
        elif s["na_rule"] == "explicit_na" and s["vtype"] == "bin":
            cats[c] = sorted(X[c].dropna().astype(str).unique().tolist())
            expl_bin.append(c)
        elif s["na_rule"] == "explicit_na":
            expl_num.append(c)
        else:
            numeric.append(c)

    return dict(spline=spline, ordinal=ordinal, nominal=nominal,
                numeric=numeric, expl_num=expl_num, expl_bin=expl_bin), cats


# ===========================================================================
# LAYER 2
# ===========================================================================
class ResponderCentred(BaseEstimator, TransformerMixin):
    """Flag plus within-responder centred value.

    A variable that is not applicable to some people cannot be zero-filled and
    handed to a linear model alongside its own not-applicable flag: the two are
    exactly collinear within the non-responder group. Instead the value is
    centred on the mean among responders in the training fold, and
    non-responders take 0, which is that mean. The flag then carries the
    not-applicable contrast and the value carries variation among responders,
    with no redundancy between them.
    """

    def fit(self, X, y=None):
        self.means_ = np.nanmean(np.asarray(X, dtype=float), axis=0)
        self.means_ = np.where(np.isnan(self.means_), 0.0, self.means_)
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        out = Xa - self.means_
        return np.where(np.isnan(out), 0.0, out)

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{f}_c" for f in input_features])


def build_preprocessor(bundle, spline=True, scale=True):
    """Everything fitted per fold. Nothing here has seen the test rows."""
    R, C = bundle["roles"], bundle["categories"]
    num = R["numeric"] + ([] if spline else R["spline"])

    blocks = []
    if spline and R["spline"]:
        blocks.append(("spl", Pipeline([
            ("i", SimpleImputer(strategy="median")),
            ("s", SplineTransformer(n_knots=5, degree=3, extrapolation="linear")),
            ("z", StandardScaler())]), R["spline"]))

    steps = [("i", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("z", StandardScaler()))
    blocks.append(("num", Pipeline(steps), num))

    if R["ordinal"]:
        blocks.append(("ord", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("e", OrdinalEncoder(categories=[C[c] for c in R["ordinal"]],
                                 handle_unknown="use_encoded_value",
                                 unknown_value=-1))]), R["ordinal"]))

    for key in ("nominal", "expl_bin"):
        if R[key]:
            blocks.append((key, Pipeline([
                ("i", SimpleImputer(strategy="most_frequent")),
                ("o", OneHotEncoder(categories=[C[c] for c in R[key]],
                                    handle_unknown="ignore",
                                    sparse_output=False))]), R[key]))

    if R["expl_num"]:
        blocks.append(("expl_num", ResponderCentred(), R["expl_num"]))

    return ColumnTransformer(blocks, verbose_feature_names_out=False)


# ===========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outcome", required=True, choices=list(OUTCOMES))
    p.add_argument("--threshold", type=float, nargs="+", default=None,
                   help="override the guideline default, e.g. 130 80 or 6.0")
    p.add_argument("--cohort", default="undiagnosed", choices=["undiagnosed", "all"])
    p.add_argument("--keep-condition-specific", action="store_true",
                   help="retain the condition-specific variables. Diagnostic only, "
                        "quantifies what they contribute on identical rows.")
    p.add_argument("--data", default=os.environ.get("PNS_DATA", "pns2013_lab_exams.xlsx"))
    p.add_argument("--registry", default=os.environ.get(
        "PNS_REGISTRY", "PNS_preprocessing_registry_v1.xlsx"))
    p.add_argument("--outdir", default="outputs")
    a = p.parse_args()

    for f in (a.data, a.registry):
        if not os.path.exists(f):
            sys.exit(f"not found: {f}")

    build_matrix(a.data, a.registry, a.outcome, a.threshold, a.cohort,
                 a.keep_condition_specific, a.outdir)


if __name__ == "__main__":
    main()
