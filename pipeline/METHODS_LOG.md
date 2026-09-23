# Decision log — undiagnosed hypertension, PNS 2013

> **Superseded, 23/09/2026.** This file records the August 2026 hypertension
> scripts, which `pipeline/pns2013_hypertension_pipeline2.py` and
> `pipeline3_generic.py` implement. It is kept because the published results of
> that generation cite it.
>
> The current work covers three outcomes over one shared preprocessing:
> `pns_preprocess.py` reads `PNS_preprocessing_registry_v1.xlsx`, and the twelve
> notebooks in `notebooks/` model each outcome with four families and four
> sampling variants. Cohorts are 6,329 at 15.2% for elevated blood pressure,
> 6,812 at 3.8% for HbA1c, and 5,927 at 32.1% for total cholesterol. The full
> reasoning lives in `decision_log.md` in the study's shared folder, sections 14
> to 17; section 17 covers what changed and why.

Project: FAPESP–Illinois NCD study Data: `EXAMES-PNS-2013-FINAL_05052023.xlsx` (lab/exams subsample) \+ variable dictionary Scope: prediction only. No survey weights, no external validation (PNS 2019 has no lab results).

---

## 1\. Outcome

| Decision | Detail |
| :---- | :---- |
| Definition of hypertension | `W00407` ≥ 140 mmHg **or** `W00408` ≥ 90 mmHg (measured, not self-reported) |
| Rows dropped | 97 with no BP measurement; pregnant women (`P005` \= 1\) |
| Analytic base | 8,855 adults; 22.5% measure ≥140/90 |
| Diagnosis variable | `Q002`, recoded 1 \= yes, 2 (pregnancy-only) \= yes, 3 \= no |

### Cross-tab that drives every framing decision

|  | No diagnosis | Diagnosed | Total |
| :---- | :---- | :---- | :---- |
| BP normal | 5,381 | 1,253 | 6,634 |
| BP ≥140/90 | 965 | 982 | 1,947 |
| Total | 6,346 | 2,235 | 8,581 |

*(8,581 \= 8,855 minus rows where `Q002` is missing, so diagnosis status is undefined.)*

---

## 2\. Model framings considered

|  | Sample | Target | n | Prev. | Status |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **A** | Reports no diagnosis | Has BP ≥140/90 | 6,346 | 15.2% | **PRIMARY** |
| B | Has BP ≥140/90 | Reports no diagnosis | 1,947 | 49.6% | Secondary |
| C | Everyone | High BP **and** no diagnosis | 8,581 | 11.2% | Robustness check, not run |
| "all" | Everyone | Has BP ≥140/90 | 8,855 | 22.5% | **Retired** — diagnostic only |

**Key reversal (recorded deliberately).** Model B was promoted to primary mid-project because the access-to-care block finally showed signal there. That was the wrong reason — optimising for an interesting finding rather than the research question. Model B conditions on *having high BP*, which is the unknown the project is trying to detect. Isa caught this. Model A is now primary.

**Model "all" retired.** It exists only to quantify how much the diagnosis variable inflates performance: on identical rows, removing the three hypertension-specific variables moved AUC 0.775 → 0.765. Worth **\+0.010**, not the \+0.036 initially implied (that gap was mostly sample composition, not variables). Keep one supplementary row; do not rerun.

---

## 3\. Predictor selection

117 variables declared across six blocks, per Isa's criteria.

| Block | Declared | Kept | Contents |
| :---- | :---- | :---- | :---- |
| A medication | 14 | 13 | last-2-weeks use \+ current use by condition, polypharmacy count |
| B sociodemographic | 13 | 13 | age, sex, race, region, education, marital, income, Bolsa Família |
| C lifestyle | 30 | 30 | tobacco, alcohol, 12 diet items, salt, physical activity, TV |
| D health | 27 | 19 | anthropometry, self-rated health, 13 diagnoses, comorbidity count |
| E access | 23 | 23 | insurance, where/how care obtained, wait times, quality ratings, discrimination |
| F sleep | 3 | 3 | `N010`, `N011`, sleep-medication items |

**Not available in this dataset** (requested but absent): family history; urban/rural. Urban/rural (`V0026`) requires merging the household file via household ID.

**Access block expanded** from 4 → 23 variables at Isa's request, including the discrimination items `X02501`–`X02510` (0% missing) — a plausible mechanism for people avoiding services.

---

## 4\. Missing data — three tiers

**Tier 1, skip patterns.** Split into two rules rather than one:

| Rule | n vars | Logic |
| :---- | :---- | :---- |
| Implied skip → impute value | 23 | The skip determines the answer. `Q006` (BP meds) is only asked of diagnosed people; someone never diagnosed is genuinely not on BP meds → 0\. A separate category would waste information *and* let the model rediscover the diagnosis. |
| Explicit not-applicable → own level | 20 | The question has no defined answer. `X019` ("via SUS?") for someone who never consulted is not "no", it's undefined → level `NA_nao_aplicavel`. Numeric versions get 0-fill \+ an `_isNA` flag. |

**Tier 2, \>30% drop.** Evaluated *after* the scenario subset, not before — missingness must be assessed in the sample actually trained on. Tier 1 was efficient enough that almost nothing exceeded threshold.

**Tier 3, MAR.** Median (numeric) / mode (categorical) with `add_indicator=True`, fitted **inside** the pipeline so it's learned per training fold.

**Decided against for now:** MICE / iterative imputation.

---

## 5\. Leakage — four tiers

| Tier | Variables | Treatment |
| :---- | :---- | :---- |
| 1 — hypertension-specific | `dx_hypertension`, `med_hypertension_2w`, `last_bp_measure` | Dropped in Model A. Not leakage strictly, but they *invert* the signal: treated patients are often controlled at the visit, so the model learns "who is on treatment". |
| 2 — requires physical exam | measured weight, height, waist, BMI | Flagged. Worth **\+0.026 AUC** (0.739 with, 0.713 without). Kept, since deployability is not a current concern. |
| 3 — lab results | `Z006`–`Z050` | Excluded entirely |
| Not leakage | other diagnoses, other medications, all access variables | Kept |

---

## 6\. Collinearity pruning

Eleven overlapping body-size variables. VIFs: `waist_height_ratio` 550, `waist_cm` 533, `weight_kg` 186, `bmi` 142\. `corr(waist_cm, waist_height_ratio)` \= 0.91.

Two harms: unstable coefficients in logistic regression (the champion model), and **corrupted importance** — permutation importance *understates* correlated features because shuffling `bmi` leaves the information in `weight_kg`.

| Kept | Univariate AUC | Dropped |
| :---- | :---- | :---- |
| `waist_cm` | 0.660 | `weight_kg`, `height_cm`, `waist_height_ratio`, `obesity`, `overweight`, `weight_at_20`, `self_weight`, `self_height` (+3 `_isNA`) |
| `bmi` | 0.604 |  |
| `self_bmi` | 0.603 | (exam-free proxy, kept for the no-exam variant) |

**Result: AUC unchanged** (0.739 → 0.739), and the importance table became readable — `waist_cm` and `bmi` moved from absent to ranks 4 and 5\.

---

## 7\. Encoding

| Type | n | Treatment |
| :---- | :---- | :---- |
| Numeric | 79 | median impute \+ standardise |
| `age` | 1 | **natural cubic splines**, 5 knots — risk is not linear in the log-odds |
| Ordinal | 16 | **OrdinalEncoder**, order preserved, one column each |
| Nominal | 16 | OneHotEncoder, `min_frequency=20` → 72 columns |

**Target encoding tested and rejected.** Fitted after the split, inside the pipeline, with internal cross-fitting (`cv=5`) and `smooth="auto"`. Leakage check clean (CV−test gap −0.007 ordinal vs −0.008 target). Gains: LR \+0.005, RF 0.000, XGB −0.001. **Not worth it** — discards real ordering, adds a fitted-from-outcome step reviewers will scrutinise, more fragile on rare levels.

**Known issue:** `min_frequency=20` silently pools rare levels of `race`, `visit_reason`, `first_care_place`, `how_got_appt`, `consult_sus` into `infrequent_sklearn`. Race levels 3 and 5 are merged — **do not report race-specific findings** without collapsing race by hand instead.

---

## 8\. Validation

| Decision | Detail |
| :---- | :---- |
| Split | 80/20 stratified, `random_state=42` |
| Tuning | `RandomizedSearchCV`, 5-fold, AUC |
| Nested CV | Added — pipeline 2 tuned and evaluated on the same split (optimistic). Nested numbers barely moved (0.784 → 0.780), so the earlier bias was small. |
| Calibration | Isotonic. Did nothing for Model B (0.189 → 0.189) at 50% prevalence; expected to matter more for Model A at 15%. |
| Threshold | Youden's J, plus a fixed \~90%-sensitivity operating point for screening |
| Interpretation | Permutation importance → SHAP (adds direction) |

---

## 9\. Results

### Model A — primary (undiagnosed sample, predict high BP)

| Configuration | LR | RF | XGB |
| :---- | :---- | :---- | :---- |
| Pruned, with exam measures | **0.739** | 0.726 | 0.735 |
| Questionnaire only (no exam) | 0.713 | 0.706 | 0.687 |

Baseline comparison (nested models, same rows, DeLong):

| Model | AUC | p vs previous |
| :---- | :---- | :---- |
| age only | 0.674 | — |
| \+ sex | 0.714 | \<0.001 |
| \+ region | 0.721 | 0.155 |
| \+ anthropometry | 0.740 | 0.032 |
| full (111 predictors) | 0.739 | — |

**Four variables match the full model.** The other \~107 add −0.001. This is the headline table for the paper.

### Model B — secondary (high-BP sample, predict non-diagnosis)

Nested CV: LR+splines 0.780, RF 0.763, XGB 0.770. Test AUC 0.777 / 0.789 / 0.779.

Parsimony curve: 5 features → 0.758, **12 → 0.774**, 20 → 0.775, 50 → 0.773, 112 → 0.769. Plateaus at 12 then *declines* — using all 112 is worse than 12\.

---

## 10\. Null results (stable across every specification)

| Block | Model A | Model B |
| :---- | :---- | :---- |
| Sociodemographic | 0.135 | 0.030 |
| Health | 0.024 | 0.035 |
| **Access to care** | **0.000** | **0.041** |
| Lifestyle | 0.002 | 0.000 |
| Medication | 0.001 | 0.001 |
| **Sleep** | **−0.000** | 0.003 |

*(Permutation importance, summed drop in AUC.)*

**Sleep contributes nothing** to either physiological or diagnostic outcomes. Report as a null result, don't quietly delete.

**Access to care contributes exactly zero to Model A** even after expanding it to 23 variables. It only matters in Model B. Interpretation: access predicts *whether you get diagnosed*, not *whether your blood pressure is high*. Since Model A is primary, this null result stands and should be reported honestly.

---

## 11\. Bugs found and fixed

| Bug | Effect | Fix |
| :---- | :---- | :---- |
| `Q002` treated with generic yes/no recode | Code 3 ("no") → NaN, so `dx_hypertension` showed 74.8% missing and was wrongly dropped | Special-cased: 1→1, 2→1, 3→0 |
| 30% rule applied before scenario subset | Missingness measured on rows never trained on | Reordered: leakage/subset → then the 30% rule |
| Binary vars with `explicit_na` routed to numeric branch | Pipeline crash (median on strings) | Three-level variables routed to one-hot |
| Mixed types in categorical columns | `OneHotEncoder` refused numeric codes \+ string level | All categoricals cast to string labels |
| **`n_medications` computed before dropping `med_hypertension_2w`** | **Outcome leaked through the sum. Model B AUC 0.86, sensitivity 1.000, NPV 1.000.** Corrected to 0.784. | Counters rebuilt from surviving classes in every scenario branch |
| SHAP block mapping split names on `_` | Most rows showed block `-` in `shap_importance_modelB.csv` | Longest-prefix match to source column |

**Standing trap:** `n_medications`, `any_medication` and `n_chronic` are sums over their blocks. Any new scenario that drops a `med_*` or `dx_*` column **must** rebuild them, or the dropped variable leaks back through the sum.

---

## 12\. Deliberately deferred

- Boruta (all-relevant selection — would convert the null blocks into a defensible statistical claim)  
- MICE  
- Survey weights (`peso_lab`) — prediction only for now; needed if population estimates are reported  
- Bootstrap confidence intervals on AUC  
- Model C (whole-sample composite outcome)  
- EBM / GAM / CatBoost — with `max_depth=2` winning, the signal is additive; expected gain is interpretability, not discrimination

---

## 13\. Code

| File | Purpose |
| :---- | :---- |
| `pns2013_hypertension_pipeline2.py` | Registry-driven feature build. Scenarios: `all`, `undiagnosed`, `nondiagnosis`. Flags: `prune`, `noexam`, `fast`, `dropbp`, `droptier1`, `dump` |
| `pipeline3_generic.py` | Splines, ordinal encoding, nested CV, calibration, SHAP, 7 figures. Takes matrix path \+ tag |
| `pipeline4_target_encoding.py` | Target-encoding comparison |
| `baseline_comparator.py` | Nested age/sex/region/anthropometry models \+ DeLong |

`VAR_SPEC` in pipeline 2 is the single source of truth — it drives both the feature build and the generated variable map, so code and documentation cannot drift.

**To run the primary model:**

python pns2013\_hypertension\_pipeline2.py undiagnosed prune dump fast

python pipeline3\_generic.py matrix\_pruned\_undiagnosed.pkl modelA  
