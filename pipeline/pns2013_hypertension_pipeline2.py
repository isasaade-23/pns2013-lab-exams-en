"""
=============================================================================
PIPELINE 2 - Predicting measured hypertension | PNS 2013 exams subsample
=============================================================================
Outcome: W00407 >= 140 mmHg OR W00408 >= 90 mmHg

What is new versus pipeline 1
    1. VAR_SPEC registry: every variable declared once, in one place. The
       registry drives the recoding AND generates the variable map, so the
       code and the documentation can never drift apart.
    2. Three-tier missing-data protocol (skip -> >30% drop -> MAR imputation)
    3. Explicit leakage tiers with a printed audit
    4. Sleep block added
    5. Access-to-care block substantially expanded
    6. 80/20 split + RandomizedSearchCV hyper-parameter tuning
       (default vs tuned reported side by side)
=============================================================================
"""

import warnings, json, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             confusion_matrix, brier_score_loss)
from xgboost import XGBClassifier
from scipy.stats import loguniform, randint, uniform

warnings.filterwarnings("ignore")
RS = 42

# ---------------------------------------------------------------------------
# SWITCHES
# ---------------------------------------------------------------------------
# Paths are resolved relative to the repository root, so the script runs from a
# fresh clone. Override with the PNS2013_DATA / PNS2013_DIC environment variables.
_ROOT       = Path(__file__).resolve().parent.parent
DATA        = os.environ.get("PNS2013_DATA", str(_ROOT / "data" / "pns2013_lab_exams.xlsx"))
DIC         = os.environ.get("PNS2013_DIC",  str(_ROOT / "dictionaries" / "dicionario_de_variaveis_PT.xlsx"))
# Scenario can be overridden from the command line:
#     python pns2013_hypertension_pipeline2.py all
#     python pns2013_hypertension_pipeline2.py undiagnosed
#     python pns2013_hypertension_pipeline2.py nondiagnosis prune
# Scenarios:
#   all          - everyone; outcome = measured BP >= 140/90
#   undiagnosed  - never-diagnosed only; outcome = measured BP >= 140/90
#   nondiagnosis - OUTCOME FLIPPED. Restricted to people who MEASURE
#                  >= 140/90; outcome = 1 if they report NO prior diagnosis.
#                  This models the diagnostic gap itself rather than
#                  physiology, which is where access-to-care variables
#                  should finally have somewhere to act.
SCENARIO    = sys.argv[1] if len(sys.argv) > 1 else "undiagnosed"
FAST        = "fast" in sys.argv        # skip the search, use fixed params
ALLOW_EXAM  = "noexam" not in sys.argv   # measured weight/height/waist available?
KEEP_TIER1  = "keeptier1" in sys.argv    # force-keep hypertension-specific vars
PRUNE       = "prune" in sys.argv        # drop collinear duplicates in the body-size block
MISS_THRESH = 0.30            # drop features above this AFTER logic imputation
TEST_SIZE   = 0.20            # 80/20
N_ITER      = 15              # RandomizedSearchCV iterations
SUFFIX      = ("_pruned" if PRUNE else "") + f"_{SCENARIO}" + ("_noexam" if not ALLOW_EXAM else "") + ("_droptier1" if "droptier1" in sys.argv else "")

# ===========================================================================
# THE VARIABLE REGISTRY
# ===========================================================================
# Fields
#   name      derived variable name
#   code      original PNS code (or a formula for engineered variables)
#   block     A_medication | B_sociodem | C_lifestyle | D_health | E_access | F_sleep
#   vtype     num | bin | cat  (how the model treats it)
#   recode    yes_no | ordinal | nominal | numeric | derived
#   na_rule   how missing values are handled BEFORE the 30% rule:
#             "implied:<v>"  skip pattern logically implies value <v>
#             "explicit_na"  skip pattern leaves the answer undefined
#                            -> becomes its own category "NA_nao_aplicavel"
#             "mar"          assumed missing at random -> median/mode later
#   leak      0 = safe | 1 = hypertension-specific (drop in screening model)
#             2 = requires physical exam | 3 = requires lab
# ---------------------------------------------------------------------------
VAR_SPEC = [
# ---- BLOCK A: medication in the last two weeks ----------------------------
 dict(name="med_hypertension_2w", code="Q006",   block="A_medication", vtype="bin", recode="yes_no",  na_rule="implied:0", leak=1, label="Took BP medication, last 2 weeks"),
 dict(name="med_diabetes_2w",     code="Q03401", block="A_medication", vtype="bin", recode="yes_no",  na_rule="implied:0", leak=0, label="Took oral diabetes medication, last 2 weeks"),
 dict(name="med_sleep_2w",        code="Q132",   block="A_medication", vtype="bin", recode="yes_no",  na_rule="mar",       leak=0, label="Took sleeping medication, last 2 weeks"),
 dict(name="med_sleep_days",      code="Q133",   block="A_medication", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days used sleeping medication, last 2 weeks"),
 dict(name="med_sleep_rx",        code="Q134",   block="A_medication", vtype="bin", recode="yes_no",  na_rule="implied:0", leak=0, label="Sleeping medication was prescribed"),
 dict(name="med_heart",     code="Q06503", block="A_medication", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Currently on heart-disease medication"),
 dict(name="med_stroke",    code="Q07205", block="A_medication", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Currently on stroke medication"),
 dict(name="med_asthma",    code="Q07701", block="A_medication", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Currently on asthma medication"),
 dict(name="med_arthritis", code="Q08103", block="A_medication", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Currently on arthritis medication"),
 dict(name="med_spine",     code="Q08603", block="A_medication", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Currently on spinal-problem medication"),
 dict(name="med_dort",      code="Q09003", block="A_medication", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Currently on work-related musculoskeletal medication"),
 dict(name="med_mental",    code="Q11402", block="A_medication", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Currently on mental-illness medication"),
 dict(name="med_lung",      code="Q11801", block="A_medication", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Currently on lung-disease medication"),
 dict(name="med_kidney",    code="Q12601", block="A_medication", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Currently on chronic kidney disease medication"),

# ---- BLOCK B: socio-demographics and SES ----------------------------------
 dict(name="age",             code="Z002",   block="B_sociodem", vtype="num", recode="numeric", na_rule="mar", leak=0, label="Age (years)"),
 dict(name="sex",             code="Z001",   block="B_sociodem", vtype="cat", recode="nominal", na_rule="mar", leak=0, label="Sex (1 male, 2 female)"),
 dict(name="race",            code="Z003",   block="B_sociodem", vtype="cat", recode="nominal", na_rule="mar", leak=0, label="Race/skin colour (9=unknown -> NaN)"),
 dict(name="region",          code="regiao", block="B_sociodem", vtype="cat", recode="nominal", na_rule="mar", leak=0, label="Region (1 N, 2 NE, 3 SE, 4 S, 5 CW)"),
 dict(name="education",       code="VDD004", block="B_sociodem", vtype="cat", recode="ordinal", na_rule="mar", leak=0, label="Highest educational level (1-7)"),
 dict(name="marital_status",  code="C011",   block="B_sociodem", vtype="cat", recode="nominal", na_rule="mar", leak=0, label="Marital status"),
 dict(name="lives_w_partner", code="C010",   block="B_sociodem", vtype="bin", recode="yes_no",  na_rule="mar", leak=0, label="Lives with a partner"),
 dict(name="household_size",  code="C001",   block="B_sociodem", vtype="num", recode="numeric", na_rule="mar", leak=0, label="Number of people in the household"),
 dict(name="employed",        code="E001",   block="B_sociodem", vtype="bin", recode="yes_no",  na_rule="mar", leak=0, label="Worked in the reference week"),
 dict(name="workforce_status",code="VDE001", block="B_sociodem", vtype="cat", recode="nominal", na_rule="mar", leak=0, label="Labour-force status"),
 dict(name="bolsa_familia",   code="F012",   block="B_sociodem", vtype="bin", recode="yes_no",  na_rule="mar", leak=0, label="Household receives Bolsa Familia (low-SES marker)"),
 dict(name="income",          code="SUM(E01602,E01604,E01802,E01804,F00102,F00702,F00802)",
                              block="B_sociodem", vtype="num", recode="derived", na_rule="implied:0", leak=0, label="Total personal monthly income (BRL)"),
 dict(name="log_income",      code="log1p(income)", block="B_sociodem", vtype="num", recode="derived", na_rule="implied:0", leak=0, label="Log of income (income is right-skewed)"),
 dict(name="income_pc",       code="income / household_size", block="B_sociodem", vtype="num", recode="derived", na_rule="mar", leak=0, label="Income per household member"),

# ---- BLOCK C: lifestyle ---------------------------------------------------
 dict(name="smoker_current", code="P050", block="C_lifestyle", vtype="cat", recode="nominal", na_rule="mar",       leak=0, label="Current tobacco use (1 daily, 2 occasional, 3 no)"),
 dict(name="smoker_past",    code="P052", block="C_lifestyle", vtype="bin", recode="yes_no",  na_rule="implied:0", leak=0, label="Ever smoked in the past"),
 dict(name="smoke_start_age",code="P053", block="C_lifestyle", vtype="num", recode="numeric", na_rule="explicit_na",leak=0, label="Age started smoking daily"),
 dict(name="passive_smoke",  code="P068", block="C_lifestyle", vtype="cat", recode="ordinal", na_rule="mar",       leak=0, label="Frequency of household smoking (passive)"),
 dict(name="alcohol_freq",   code="P027", block="C_lifestyle", vtype="cat", recode="ordinal", na_rule="mar",       leak=0, label="Alcohol consumption frequency"),
 dict(name="alcohol_days",   code="P028", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days per week drinking alcohol"),
 dict(name="alcohol_doses",  code="P029", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Doses per drinking day"),
 dict(name="binge_drink",    code="P032", block="C_lifestyle", vtype="bin", recode="yes_no",  na_rule="implied:0", leak=0, label="Binge drinking in the last 30 days"),
 dict(name="alcohol_start_age", code="P031", block="C_lifestyle", vtype="num", recode="numeric", na_rule="explicit_na", leak=0, label="Age started drinking"),
 dict(name="diet_beans",     code="P006", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week eating beans"),
 dict(name="diet_salad",     code="P007", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week eating raw salad"),
 dict(name="diet_cooked_veg",code="P009", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week eating cooked vegetables"),
 dict(name="diet_red_meat",  code="P011", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week eating red meat"),
 dict(name="diet_chicken",   code="P013", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week eating chicken"),
 dict(name="diet_fish",      code="P015", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week eating fish"),
 dict(name="diet_juice",     code="P016", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week drinking natural fruit juice"),
 dict(name="diet_fruit",     code="P018", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week eating fruit"),
 dict(name="diet_soda",      code="P020", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week drinking soda"),
 dict(name="diet_milk",      code="P023", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week drinking milk"),
 dict(name="diet_sweets",    code="P025", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week eating sweets"),
 dict(name="diet_fastfood",  code="P026", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week replacing a meal with fast food"),
 dict(name="salt_perception",code="P02601", block="C_lifestyle", vtype="cat", recode="ordinal", na_rule="mar", leak=0, label="Self-perceived salt intake (1 very high - 5 very low)"),
 dict(name="exercise_3m",    code="P034", block="C_lifestyle", vtype="bin", recode="yes_no",  na_rule="mar",       leak=0, label="Practised exercise in the last 3 months"),
 dict(name="exercise_days",  code="P035", block="C_lifestyle", vtype="num", recode="numeric", na_rule="implied:0", leak=0, label="Days/week of exercise"),
 dict(name="exercise_min_week", code="(P03701*60 + P03702) * P035", block="C_lifestyle", vtype="num", recode="derived", na_rule="implied:0", leak=0, label="Total weekly minutes of exercise"),
 dict(name="walks_at_work",  code="P038", block="C_lifestyle", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Walks a lot at work"),
 dict(name="heavy_work",     code="P039", block="C_lifestyle", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Heavy physical activity at work"),
 dict(name="active_commute", code="P040", block="C_lifestyle", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Walks/cycles to work"),
 dict(name="heavy_housework",code="P044", block="C_lifestyle", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Heavy housework"),
 dict(name="tv_hours",       code="P045", block="C_lifestyle", vtype="cat", recode="ordinal", na_rule="mar", leak=0, label="Hours/day watching TV (8 = does not watch)"),
 dict(name="public_space",   code="P046", block="C_lifestyle", vtype="bin", recode="yes_no", na_rule="mar", leak=0, label="Public space for exercise near home"),

# ---- BLOCK D: anthropometry, self-rated health, chronic disease -----------
 dict(name="weight_kg",  code="z004",   block="D_health", vtype="num", recode="numeric", na_rule="mar", leak=2, label="Measured weight (kg), 999 -> NaN"),
 dict(name="height_cm",  code="z005",   block="D_health", vtype="num", recode="numeric", na_rule="mar", leak=2, label="Measured height (cm), 999 -> NaN"),
 dict(name="bmi",        code="weight_kg/(height_cm/100)^2", block="D_health", vtype="num", recode="derived", na_rule="mar", leak=2, label="Body Mass Index"),
 dict(name="obesity",    code="bmi >= 30", block="D_health", vtype="bin", recode="derived", na_rule="mar", leak=2, label="Obesity (BMI >= 30)"),
 dict(name="overweight", code="bmi >= 25", block="D_health", vtype="bin", recode="derived", na_rule="mar", leak=2, label="Overweight (BMI >= 25)"),
 dict(name="waist_cm",   code="W00303", block="D_health", vtype="num", recode="numeric", na_rule="mar", leak=2, label="Measured waist circumference (cm)"),
 dict(name="waist_height_ratio", code="waist_cm/height_cm", block="D_health", vtype="num", recode="derived", na_rule="mar", leak=2, label="Waist-to-height ratio"),
 dict(name="knows_weight",   code="P001",   block="D_health", vtype="bin", recode="yes_no",  na_rule="mar",         leak=0, label="Knows own weight"),
 dict(name="self_weight",    code="P00101", block="D_health", vtype="num", recode="numeric", na_rule="explicit_na", leak=0, label="Self-reported weight (kg)"),
 dict(name="self_height",    code="P00401", block="D_health", vtype="num", recode="numeric", na_rule="explicit_na", leak=0, label="Self-reported height (cm)"),
 dict(name="self_bmi",       code="self_weight/(self_height/100)^2", block="D_health", vtype="num", recode="derived", na_rule="explicit_na", leak=0, label="Self-reported BMI (exam-free proxy)"),
 dict(name="weight_at_20",   code="P00301", block="D_health", vtype="num", recode="numeric", na_rule="explicit_na", leak=0, label="Recalled weight at age 20"),
 dict(name="self_rated_health", code="N001", block="D_health", vtype="cat", recode="ordinal", na_rule="mar", leak=0, label="Self-rated health (1 very good - 5 very poor)"),
 dict(name="mobility_difficulty", code="N003", block="D_health", vtype="cat", recode="ordinal", na_rule="mar", leak=0, label="Difficulty walking"),
 dict(name="chest_pain_exertion", code="N004", block="D_health", vtype="cat", recode="nominal", na_rule="mar", leak=0, label="Chest pain when climbing/walking fast"),
 dict(name="dx_hypertension", code="Q002", block="D_health", vtype="bin", recode="yes_no", na_rule="mar",       leak=1, label="Ever diagnosed with hypertension"),
 dict(name="dx_diabetes",     code="Q030", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Ever diagnosed with diabetes"),
 dict(name="dx_cholesterol",  code="Q060", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Ever diagnosed with high cholesterol"),
 dict(name="dx_heart",        code="Q063", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Ever diagnosed with heart disease"),
 dict(name="dx_stroke",       code="Q068", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Ever diagnosed with stroke"),
 dict(name="dx_asthma",       code="Q074", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Ever diagnosed with asthma"),
 dict(name="dx_arthritis",    code="Q079", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Ever diagnosed with arthritis"),
 dict(name="dx_spine",        code="Q084", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Chronic spinal problem"),
 dict(name="dx_dort",         code="Q088", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Work-related musculoskeletal disorder"),
 dict(name="dx_depression",   code="Q092", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Ever diagnosed with depression"),
 dict(name="dx_lung",         code="Q116", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Ever diagnosed with lung disease/COPD"),
 dict(name="dx_kidney",       code="Q124", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Ever diagnosed with chronic kidney disease"),
 dict(name="dx_other_chronic",code="Q128", block="D_health", vtype="bin", recode="yes_no", na_rule="implied:0", leak=0, label="Any other chronic disease diagnosis"),

# ---- BLOCK E: access to health care ---------------------------------------
 dict(name="health_insurance",   code="I001",   block="E_access", vtype="bin", recode="yes_no",  na_rule="mar",         leak=0, label="Has private health insurance"),
 dict(name="last_medical_visit", code="X001",   block="E_access", vtype="cat", recode="ordinal", na_rule="mar",         leak=0, label="Time since last medical consultation (1 <2wk - 6 never)"),
 dict(name="visit_reason",       code="X002",   block="E_access", vtype="cat", recode="nominal", na_rule="explicit_na", leak=0, label="Reason for the last consultation"),
 dict(name="first_care_place",   code="X003",   block="E_access", vtype="cat", recode="nominal", na_rule="explicit_na", leak=0, label="Where first care was sought (13 categories)"),
 dict(name="got_care_first_try", code="X004",   block="E_access", vtype="bin", recode="yes_no",  na_rule="explicit_na", leak=0, label="Obtained care on the first attempt"),
 dict(name="care_same_city",     code="X008",   block="E_access", vtype="bin", recode="yes_no",  na_rule="explicit_na", leak=0, label="Care located in the same city of residence"),
 dict(name="how_got_appt",       code="X011",   block="E_access", vtype="cat", recode="nominal", na_rule="explicit_na", leak=0, label="How the appointment was obtained"),
 dict(name="wait_minutes",       code="X01401*60 + X01402", block="E_access", vtype="num", recode="derived", na_rule="explicit_na", leak=0, label="Waiting time in the queue (minutes)"),
 dict(name="consult_minutes",    code="X01501*60 + X01502", block="E_access", vtype="num", recode="derived", na_rule="explicit_na", leak=0, label="Duration of the consultation (minutes)"),
 dict(name="doctor_type",        code="X016",   block="E_access", vtype="cat", recode="nominal", na_rule="explicit_na", leak=0, label="Type of doctor seen"),
 dict(name="consult_by_plan",    code="X017",   block="E_access", vtype="bin", recode="yes_no",  na_rule="explicit_na", leak=0, label="Consultation covered by insurance"),
 dict(name="consult_paid",       code="X018",   block="E_access", vtype="bin", recode="yes_no",  na_rule="explicit_na", leak=0, label="Paid out of pocket for the consultation"),
 dict(name="consult_sus",        code="X019",   block="E_access", vtype="cat", recode="nominal", na_rule="explicit_na", leak=0, label="Consultation through SUS (public system)"),
 dict(name="qual_availability",  code="X02001", block="E_access", vtype="cat", recode="ordinal", na_rule="explicit_na", leak=0, label="Rating: availability of the service"),
 dict(name="qual_waiting_time",  code="X02004", block="E_access", vtype="cat", recode="ordinal", na_rule="explicit_na", leak=0, label="Rating: waiting time"),
 dict(name="qual_doctor_skill",  code="X02201", block="E_access", vtype="cat", recode="ordinal", na_rule="explicit_na", leak=0, label="Rating: doctor's skills"),
 dict(name="qual_explanation",   code="X02203", block="E_access", vtype="cat", recode="ordinal", na_rule="explicit_na", leak=0, label="Rating: clarity of explanations"),
 dict(name="sought_care_2w",     code="J014",   block="E_access", vtype="bin", recode="yes_no",  na_rule="mar", leak=0, label="Sought health care in the last 2 weeks"),
 dict(name="restricted_activity_2w", code="J002", block="E_access", vtype="bin", recode="yes_no", na_rule="mar", leak=0, label="Stopped usual activities for health reasons (2 wk)"),
 dict(name="hospitalized_12m",   code="J037",   block="E_access", vtype="bin", recode="yes_no",  na_rule="mar", leak=0, label="Hospitalised >=24h in the last 12 months"),
 dict(name="last_bp_measure",    code="Q001",   block="E_access", vtype="cat", recode="ordinal", na_rule="mar", leak=1, label="Time since BP was last measured"),
 dict(name="last_glucose_test",  code="Q029",   block="E_access", vtype="cat", recode="ordinal", na_rule="mar", leak=0, label="Time since last blood glucose test"),
 dict(name="discrimination_care", code="ANY(X02501..X02510)", block="E_access", vtype="bin", recode="derived", na_rule="mar", leak=0, label="Ever felt discriminated against in a health service"),
 dict(name="n_discrimination",   code="SUM(X02501..X02510)", block="E_access", vtype="num", recode="derived", na_rule="mar", leak=0, label="Number of discrimination settings reported"),

# ---- BLOCK F: sleep -------------------------------------------------------
 dict(name="sleep_problems",  code="N010", block="F_sleep", vtype="cat", recode="ordinal", na_rule="mar", leak=0, label="Frequency of sleep problems, last 2 weeks (1 none - 4 almost daily)"),
 dict(name="unrested",        code="N011", block="F_sleep", vtype="cat", recode="ordinal", na_rule="mar", leak=0, label="Frequency of feeling unrested, last 2 weeks"),
 dict(name="poor_sleep_flag", code="N010 >= 3", block="F_sleep", vtype="bin", recode="derived", na_rule="mar", leak=0, label="Sleep problems on more than half the days"),
]

# Engineered counters, added after the main loop
EXTRA_SPEC = [
 dict(name="n_medications", code="SUM(block A binaries)", block="A_medication", vtype="num", recode="derived", na_rule="implied:0", leak=0, label="Number of medication classes in use (polypharmacy proxy)"),
 dict(name="any_medication", code="n_medications > 0",   block="A_medication", vtype="bin", recode="derived", na_rule="implied:0", leak=0, label="Uses any medication"),
 dict(name="n_chronic",      code="SUM(dx_* except hypertension)", block="D_health", vtype="num", recode="derived", na_rule="implied:0", leak=0, label="Comorbidity count"),
]

# ===========================================================================
# STEP 1 - Load and build the outcome
# ===========================================================================
print("=" * 78)
print("STEP 1 | LOADING DATA AND BUILDING THE OUTCOME")
print("=" * 78)
raw = pd.read_excel(DATA)
print(f"  raw file: {raw.shape[0]} rows x {raw.shape[1]} columns")

raw = raw[raw["W00407"].notna() & raw["W00408"].notna()].copy()
raw["hypertension"] = ((raw["W00407"] >= 140) | (raw["W00408"] >= 90)).astype(int)
raw = raw[raw["P005"] != 1].copy()                 # exclude pregnant women
print(f"  after excluding missing BP and pregnancy: {len(raw)} adults")
print(f"  measured hypertension prevalence: {raw['hypertension'].mean():.1%}")

# ===========================================================================
# STEP 2 - Build derived variables from the registry
# ===========================================================================
print("\n" + "=" * 78)
print("STEP 2 | BUILDING VARIABLES FROM THE REGISTRY")
print("=" * 78)

X = pd.DataFrame(index=raw.index)

def yn(code):
    """PNS 1/2 (Yes/No) -> 1/0. Codes 3 and 9 ('don't know'/'ignored') -> NaN."""
    return raw[code].replace({1: 1, 2: 0, 3: np.nan, 9: np.nan})

# --- simple pass-through and yes/no variables
for s in VAR_SPEC:
    c, n = s["code"], s["name"]
    if s["recode"] == "yes_no":
        X[n] = yn(c)
    elif s["recode"] in ("numeric", "ordinal", "nominal") and c in raw.columns:
        X[n] = raw[c]

# --- special recodes and engineered variables
# Q002 has its OWN coding: 1 = yes, 2 = only during pregnancy, 3 = no.
# The generic yes_no() would wrongly send code 3 to NaN, so it is redone here.
X["dx_hypertension"] = raw["Q002"].replace({1: 1, 2: 1, 3: 0})

X["race"]      = raw["Z003"].replace({9: np.nan})
X["weight_kg"] = raw["z004"].replace({999: np.nan})
X["height_cm"] = raw["z005"].replace({999: np.nan})
X["bmi"]       = X["weight_kg"] / (X["height_cm"] / 100) ** 2
X["obesity"]    = (X["bmi"] >= 30).astype(float).where(X["bmi"].notna())
X["overweight"] = (X["bmi"] >= 25).astype(float).where(X["bmi"].notna())
X["waist_height_ratio"] = X["waist_cm"] / X["height_cm"]
X["self_bmi"]  = X["self_weight"] / (X["self_height"] / 100).replace(0, np.nan) ** 2

inc_cols = ["E01602", "E01604", "E01802", "E01804", "F00102", "F00702", "F00802"]
X["income"]     = raw[inc_cols].fillna(0).sum(axis=1)
X["log_income"] = np.log1p(X["income"])
X["income_pc"]  = X["income"] / X["household_size"].replace(0, np.nan)

X["exercise_min_week"] = ((raw["P03701"].fillna(0) * 60 + raw["P03702"].fillna(0))
                          * raw["P035"].fillna(0))
X["wait_minutes"]    = raw["X01401"] * 60 + raw["X01402"]
X["consult_minutes"] = raw["X01501"] * 60 + raw["X01502"]

disc = [f"X025{i:02d}" for i in range(1, 11)]
disc_bin = raw[disc].replace({1: 1, 2: 0, 3: np.nan})
X["n_discrimination"]    = disc_bin.sum(axis=1)
X["discrimination_care"] = (X["n_discrimination"] > 0).astype(float)

X["poor_sleep_flag"] = (raw["N010"] >= 3).astype(float).where(raw["N010"].notna())
print(f"  built {X.shape[1]} variables")

# ===========================================================================
# STEP 3 - MISSING DATA, TIER 1: skip-pattern logic
# ===========================================================================
print("\n" + "=" * 78)
print("STEP 3 | MISSING TIER 1 - SKIP-PATTERN LOGIC")
print("=" * 78)
print("""  Two different situations are handled differently:

  (a) IMPLIED SKIP - the skip pattern logically determines the answer.
      Q006 'did you take BP medication?' is only asked to people who said
      yes to Q002. Someone never diagnosed is not on BP medication, so the
      true value is 0, not 'unknown'. Imputing a separate category here
      would waste information and let the model rediscover the diagnosis.
      -> impute the implied value (usually 0).

  (b) EXPLICIT NOT-APPLICABLE - the question has no defined answer.
      X019 'was the consultation through SUS?' for someone who has never
      consulted a doctor is not 'no', it is undefined. Recoding it to 0
      would be a factual error.
      -> becomes its own level, 'NA_nao_aplicavel', so the model can use
         'never consulted' as a signal in its own right. For numeric
         variables this means a 0 fill PLUS a companion _isNA flag.""")

miss_raw = X.isna().mean()
implied_log, explicit_log = [], []

for s in VAR_SPEC + EXTRA_SPEC:
    n, rule = s["name"], s["na_rule"]
    if n not in X.columns:
        continue
    if rule.startswith("implied:"):
        val = float(rule.split(":")[1])
        if X[n].isna().any():
            implied_log.append((n, X[n].isna().mean(), val))
        X[n] = X[n].fillna(val)
    elif rule == "explicit_na":
        if s["vtype"] == "num":
            X[f"{n}_isNA"] = X[n].isna().astype(int)
            X[n] = X[n].fillna(0)
        else:
            X[n] = X[n].astype("object").where(X[n].notna(), "NA_nao_aplicavel")
        explicit_log.append((n, miss_raw[n]))

print(f"\n  (a) implied skip -> value imputed: {len(implied_log)} variables")
for n, m, v in sorted(implied_log, key=lambda t: -t[1])[:10]:
    print(f"        {n:<24} {m:5.1%} missing -> {v:g}")
print(f"\n  (b) explicit not-applicable category: {len(explicit_log)} variables")
for n, m in sorted(explicit_log, key=lambda t: -t[1])[:10]:
    print(f"        {n:<24} {m:5.1%} -> level 'NA_nao_aplicavel'")

# engineered counters (after tier 1, so the components are already filled)
med_bins = [s["name"] for s in VAR_SPEC if s["block"] == "A_medication" and s["vtype"] == "bin"]
X["n_medications"]  = X[med_bins].sum(axis=1)
X["any_medication"] = (X["n_medications"] > 0).astype(int)
dx_bins = [s["name"] for s in VAR_SPEC
           if s["name"].startswith("dx_") and s["name"] != "dx_hypertension"]
X["n_chronic"] = X[dx_bins].sum(axis=1)

# ===========================================================================
# STEP 3b - COLLINEARITY PRUNING IN THE BODY-SIZE BLOCK
# ===========================================================================
# The anthropometric block was built with 11 overlapping variables. They are
# near-duplicates of each other:
#   VIF: waist_height_ratio 550, waist_cm 533, weight_kg 186, bmi 142
#   corr(waist_cm, waist_height_ratio) = 0.91
#   bmi is a deterministic function of weight_kg and height_cm
#   obesity / overweight are thresholded versions of bmi
# This hurts twice over. Logistic regression - the best model here - gets
# unstable coefficients and inflated standard errors. And permutation
# importance UNDERSTATES correlated features, because shuffling bmi leaves
# the same information sitting in weight_kg and waist_cm, so the credit is
# split across collinear twins instead of concentrating on the real effect.
#
# Kept, based on univariate AUC:
#   waist_cm  0.660  (strongest single body-size predictor; central adiposity)
#   bmi       0.604  (standard, comparable across studies)
#   self_bmi  0.603  (the only body-size variable that survives without a
#                     physical examination - kept for the noexam variant)
PRUNE_LIST = ["weight_kg", "height_cm", "waist_height_ratio", "obesity",
              "overweight", "weight_at_20", "self_weight", "self_height"]

if PRUNE:
    gone = [c for c in X.columns if c in PRUNE_LIST or
            c.replace("_isNA", "") in PRUNE_LIST]
    X = X.drop(columns=gone)
    print("\n" + "=" * 78)
    print("STEP 3b | COLLINEARITY PRUNING")
    print("=" * 78)
    print(f"  dropped {len(gone)} redundant body-size variables: {gone}")
    print("  kept: waist_cm, bmi (and self_bmi as the exam-free proxy)")

# ===========================================================================
# STEP 4 - LEAKAGE AUDIT AND SCENARIO SUBSET
# ===========================================================================
print("\n" + "=" * 78)
print("STEP 4 | LEAKAGE AUDIT AND SCENARIO SUBSET")
print("=" * 78)
print("""  TIER 1 - hypertension-specific. dx_hypertension, med_hypertension_2w and
     last_bp_measure are not leakage in the strict sense (they precede the
     measurement), but they invert the signal: treated patients are often
     CONTROLLED at the visit, so the model learns 'who is on treatment'
     rather than 'whose BP is high'. They are also unavailable by
     construction in the undiagnosed population you are targeting.
     -> removed whenever SCENARIO = 'undiagnosed'.

  TIER 2 - requires a physical examination (measured weight, height, waist,
     BMI). Not leakage of the outcome, but measured by the same team at the
     same visit as the BP. A model that needs them cannot be applied to
     people who were never examined - which is exactly your target group.
     Self-reported weight/height (self_bmi) is kept as the exam-free proxy.
     -> controlled by ALLOW_EXAM.

  TIER 3 - laboratory results (Z006-Z050). Excluded entirely: they require a
     blood draw, and the under-diagnosis target group is defined by NOT
     having one.

  NOT LEAKAGE, keep: diagnoses of OTHER diseases, medication for other
     conditions, and every access-to-care variable. These are legitimately
     observable before any BP measurement.""")

spec_by_name = {s["name"]: s for s in VAR_SPEC + EXTRA_SPEC}

def leak_of(col):
    base = col.replace("_isNA", "")
    return spec_by_name.get(base, {}).get("leak", 0)

if SCENARIO == "undiagnosed":
    X = X[X["dx_hypertension"] == 0].copy()
    raw = raw.loc[X.index]
    drop_leak = [] if KEEP_TIER1 else [c for c in X.columns if leak_of(c) == 1]
    X = X.drop(columns=drop_leak)
    print(f"\n  SCENARIO='undiagnosed': kept {len(X)} never-diagnosed adults")
    print(f"  removed tier-1 variables: {drop_leak}")
    X["n_medications"] = X[[c for c in med_bins if c in X.columns]].sum(axis=1)
    X["any_medication"] = (X["n_medications"] > 0).astype(int)
elif SCENARIO == "nondiagnosis":
    # Q002 is missing for ~3% of people; without it the outcome is undefined,
    # so those rows are dropped rather than imputed - you cannot impute the
    # thing you are trying to predict.
    keep_rows = (raw["hypertension"] == 1) & X["dx_hypertension"].notna()
    n_undef = ((raw["hypertension"] == 1) & X["dx_hypertension"].isna()).sum()
    print(f"\n  dropped {n_undef} rows with an undefined outcome (Q002 missing)")
    X = X[keep_rows].copy()
    raw = raw.loc[X.index].copy()
    # the outcome is now "measures high but reports NO diagnosis"
    raw["target_nondx"] = (1 - X["dx_hypertension"]).astype(int)

    # dx_hypertension IS the outcome; med_hypertension_2w is a near-perfect
    # function of it (you cannot take BP drugs you were never prescribed).
    # last_bp_measure is kept by default: "when was your BP last measured"
    # is a genuine access variable here, not leakage - but it sits close to
    # the causal path, so `dropbp` removes it for a sensitivity analysis.
    forced = ["dx_hypertension", "med_hypertension_2w"]
    if "dropbp" in sys.argv:
        forced.append("last_bp_measure")
    drop_leak = [c for c in X.columns if c in forced]
    X = X.drop(columns=drop_leak)
    print(f"\n  SCENARIO='nondiagnosis': kept {len(X)} adults who MEASURE >= 140/90")
    print(f"  outcome = reports no prior hypertension diagnosis")
    print(f"  removed as outcome-defining: {drop_leak}")
    # CRITICAL: n_medications / any_medication were built in STEP 3 from ALL
    # the med_* binaries, INCLUDING med_hypertension_2w. Dropping that column
    # is not enough - the count still encodes it, so the outcome leaks in
    # through the back door. Both counters are rebuilt from what survives.
    surviving_meds = [c for c in med_bins if c in X.columns]
    X["n_medications"] = X[surviving_meds].sum(axis=1)
    X["any_medication"] = (X["n_medications"] > 0).astype(int)
    print(f"  rebuilt n_medications from {len(surviving_meds)} surviving classes "
          f"(was contaminated by med_hypertension_2w)")
    if "dropbp" not in sys.argv:
        print("  NOTE: last_bp_measure retained (access variable; use 'dropbp' to test without)")

elif "droptier1" in sys.argv:
    drop_leak = [c for c in X.columns if leak_of(c) == 1]
    X = X.drop(columns=drop_leak)
    print(f"\n  SCENARIO='all' + droptier1: removed {drop_leak} on the SAME rows,")
    print("  so the AUC difference isolates exactly what those variables contribute.")
else:
    print("\n  SCENARIO='all': tier-1 variables KEPT (interpret with care)")

if not ALLOW_EXAM:
    drop_exam = [c for c in X.columns if leak_of(c) == 2]
    X = X.drop(columns=drop_exam)
    print(f"  ALLOW_EXAM=False: removed {len(drop_exam)} measured-anthropometry variables")

# ===========================================================================
# STEP 5 - MISSING TIER 2: drop features still above 30%
# ===========================================================================
print("\n" + "=" * 78)
print("STEP 5 | MISSING TIER 2 - DROP FEATURES ABOVE 30%")
print("  (evaluated AFTER the scenario subset, so the rates reflect the sample\n   the model is actually trained on)")
print("=" * 78)
miss_post = X.isna().mean()
dropped_missing = miss_post[miss_post > MISS_THRESH].sort_values(ascending=False)
if len(dropped_missing):
    print(f"  dropping {len(dropped_missing)} variables:")
    for n, m in dropped_missing.items():
        print(f"        {n:<24} {m:5.1%}")
else:
    print("  no variable exceeds the threshold after tier 1 - nothing dropped.")
X = X.drop(columns=dropped_missing.index)

TARGET = "target_nondx" if SCENARIO == "nondiagnosis" else "hypertension"
y = raw.loc[X.index, TARGET]
print(f"\n  final matrix: {X.shape[0]} rows x {X.shape[1]} predictors")
print(f"  outcome prevalence in this scenario: {y.mean():.1%}")

# ===========================================================================
# STEP 6 - Column roles for preprocessing
# ===========================================================================
def vtype_of(col):
    """Where does this column go in the ColumnTransformer?
    A binary variable that received an explicit 'not applicable' level now has
    THREE levels (yes / no / not applicable), so it is routed to the
    categorical branch and one-hot encoded instead of treated as 0/1."""
    if col.endswith("_isNA"):
        return "bin"
    sp = spec_by_name.get(col, {})
    if sp.get("na_rule") == "explicit_na" and sp.get("vtype") == "bin":
        return "cat"
    return sp.get("vtype", "num")

NUM = [c for c in X.columns if vtype_of(c) in ("num", "bin")]
CAT = [c for c in X.columns if vtype_of(c) == "cat"]
# Categorical columns can hold numeric codes (1.0, 2.0, ...) AND the string
# "NA_nao_aplicavel". OneHotEncoder refuses mixed types, so everything is cast
# to a string label; genuine NaN is left as NaN for the imputer to handle.
for c in CAT:
    X[c] = X[c].apply(lambda v: np.nan if pd.isna(v)
                      else (f"{v:g}" if isinstance(v, (int, float, np.number)) else str(v)))
print(f"  numeric/binary: {len(NUM)} | categorical: {len(CAT)}")

# ===========================================================================
# STEP 7 - 80/20 split
# ===========================================================================
print("\n" + "=" * 78)
print("STEP 7 | TRAIN/TEST SPLIT (80/20, STRATIFIED)")
print("=" * 78)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RS)
print(f"  train {len(Xtr)} ({ytr.mean():.1%} positive) | test {len(Xte)} ({yte.mean():.1%} positive)")

# ===========================================================================
# STEP 8 - MISSING TIER 3: MAR imputation inside the pipeline
# ===========================================================================
print("\n" + "=" * 78)
print("STEP 8 | MISSING TIER 3 - MAR IMPUTATION")
print("=" * 78)
print("""  Everything still missing is assumed missing at random and imputed
  INSIDE the pipeline, so the median/mode is learned on the training fold
  only. add_indicator=True adds a flag for each imputed column, so the
  model can still see that the value was absent.
    numeric     -> median (robust to the skew in income, waist, minutes)
    categorical -> most frequent, then one-hot""")

def preproc(scale=False):
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False, min_frequency=20))]), CAT),
    ])

still_missing = X[NUM + CAT].isna().mean()
still_missing = still_missing[still_missing > 0].sort_values(ascending=False)
print(f"\n  {len(still_missing)} variables go through MAR imputation:")
for n, m in still_missing.head(12).items():
    print(f"        {n:<24} {m:5.1%}")

# ===========================================================================
# STEP 9 - Models: default vs tuned
# ===========================================================================
print("\n" + "=" * 78)
print("STEP 9 | HYPER-PARAMETER TUNING (RandomizedSearchCV, 5-fold, AUC)")
print("=" * 78)

spw = (ytr == 0).sum() / (ytr == 1).sum()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

SETUP = {
    "Logistic Regression": dict(
        est=LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RS),
        scale=True,
        grid={"clf__C": loguniform(1e-3, 1e2),
              "clf__penalty": ["l1", "l2"],
              # saga is much slower here and reaches the same optimum
              "clf__solver": ["liblinear"]}),
    "Random Forest": dict(
        est=RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RS),
        scale=False,
        grid={"clf__n_estimators": randint(250, 600),
              "clf__max_depth": [None, 6, 10, 16, 24],
              "clf__min_samples_leaf": randint(1, 25),
              "clf__max_features": ["sqrt", "log2", 0.3, 0.5]}),
    "XGBoost": dict(
        est=XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                          n_jobs=-1, random_state=RS),
        scale=False,
        grid={"clf__n_estimators": randint(200, 700),
              "clf__learning_rate": loguniform(0.01, 0.3),
              "clf__max_depth": randint(2, 8),
              "clf__subsample": uniform(0.6, 0.4),
              "clf__colsample_bytree": uniform(0.5, 0.5),
              "clf__min_child_weight": randint(1, 12),
              "clf__reg_lambda": loguniform(0.1, 20)}),
}

def evaluate(pipe, tag):
    """Fit already done. Returns a metrics dict computed on the test set."""
    p = pipe.predict_proba(Xte)[:, 1]
    fpr, tpr, thr = roc_curve(yte, p)
    t = thr[np.argmax(tpr - fpr)]                      # Youden's J
    tn, fp, fn, tp = confusion_matrix(yte, (p >= t).astype(int)).ravel()
    return dict(model=tag,
                AUC_test=roc_auc_score(yte, p),
                PR_AUC=average_precision_score(yte, p),
                Brier=brier_score_loss(yte, p),
                threshold=t,
                Sensitivity=tp / (tp + fn),
                Specificity=tn / (tn + fp),
                PPV=tp / (tp + fp) if (tp + fp) else np.nan,
                NPV=tn / (tn + fn) if (tn + fn) else np.nan)

rows, best_params, tuned_models = [], {}, {}

for name, cfg in SETUP.items():
    print(f"\n  --- {name} ---")

    base = Pipeline([("prep", preproc(cfg["scale"])), ("clf", cfg["est"])])
    cv_base = cross_val_score(base, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=-1)
    base.fit(Xtr, ytr)
    r = evaluate(base, f"{name} (default)")
    r["CV_AUC_train"] = f"{cv_base.mean():.3f}+/-{cv_base.std():.3f}"
    rows.append(r)
    print(f"      default   CV AUC {cv_base.mean():.3f} | test AUC {r['AUC_test']:.3f}")

    if FAST:
        rows[-1]["model"] = f"{name}"
        tuned_models[name] = base
        best_params[name] = "fast mode - defaults used"
        continue

    search = RandomizedSearchCV(base, cfg["grid"], n_iter=N_ITER, cv=cv,
                                scoring="roc_auc", n_jobs=-1,
                                random_state=RS, refit=True)
    search.fit(Xtr, ytr)
    r = evaluate(search.best_estimator_, f"{name} (tuned)")
    r["CV_AUC_train"] = f"{search.best_score_:.3f}"
    rows.append(r)
    tuned_models[name] = search.best_estimator_
    best_params[name] = {k.replace("clf__", ""): (round(v, 4) if isinstance(v, float) else v)
                         for k, v in search.best_params_.items()}
    print(f"      tuned     CV AUC {search.best_score_:.3f} | test AUC {r['AUC_test']:.3f}")
    print(f"      best params: {best_params[name]}")

res = (pd.DataFrame(rows)
       .set_index("model")[["CV_AUC_train", "AUC_test", "PR_AUC", "Brier",
                            "threshold", "Sensitivity", "Specificity", "PPV", "NPV"]])
print("\n" + "=" * 78)
print("RESULTS - all models, default vs tuned")
print("=" * 78)
print(res.round(3).to_string())

best_name = res["AUC_test"].idxmax()
print(f"\n  Best on the test set: {best_name} (AUC {res.loc[best_name,'AUC_test']:.3f})")

# ===========================================================================
# STEP 10 - Permutation importance on the best TUNED model
# ===========================================================================
# Permutation importance shuffles one column at a time in the TEST set and
# measures how much AUC drops. It is model-agnostic and, unlike the impurity
# importance that comes free with a Random Forest, it is not biased toward
# high-cardinality variables. It is computed on data the model never saw.
print("\n" + "=" * 78)
print("STEP 10 | PERMUTATION IMPORTANCE (best tuned model, test set)")
print("=" * 78)

from sklearn.inspection import permutation_importance

best_tuned_name = best_name.replace(" (tuned)", "").replace(" (default)", "")
best_est = tuned_models[best_tuned_name]
print(f"  model: {best_tuned_name} (tuned)")

perm = permutation_importance(best_est, Xte, yte, n_repeats=10,
                              scoring="roc_auc", random_state=RS, n_jobs=-1)

imp = (pd.DataFrame({"feature": Xte.columns,
                     "drop_in_AUC": perm.importances_mean,
                     "sd": perm.importances_std})
       .sort_values("drop_in_AUC", ascending=False)
       .reset_index(drop=True))
imp["block"] = imp["feature"].map(
    lambda c: spec_by_name.get(c.replace("_isNA", ""), {}).get("block", "-"))
imp.to_csv(f"feature_importance{SUFFIX}.csv", index=False)

print("\n  Top 25 predictors:")
print(imp.head(25).round(4).to_string(index=False))

# how much does each BLOCK contribute in total?
print("\n  Total AUC contribution by block:")
print(imp.groupby("block")["drop_in_AUC"].agg(["sum", "max"])
         .sort_values("sum", ascending=False).round(4).to_string())

# ===========================================================================
# STEP 11 - Variable map
# ===========================================================================
print("\n" + "=" * 78)
print("STEP 11 | VARIABLE MAP")
print("=" * 78)

dic = pd.read_excel(DIC, sheet_name="Plan2", header=None)
dic.columns = ["cod", "desc", "cat", "catdesc", "x"]
pns_label = {}
for _, r in dic.iterrows():
    c = r["cod"]
    if isinstance(c, str) and c.strip() and c.strip() not in pns_label:
        pns_label[c.strip().upper()] = str(r["desc"])[:120]

kept = set(X.columns)
rows_map = []
for s in VAR_SPEC + EXTRA_SPEC:
    n = s["name"]
    if n in dropped_missing.index:
        status, why = "DROPPED", f">{MISS_THRESH:.0%} missing after tier 1 ({dropped_missing[n]:.1%})"
    elif n not in kept and s["leak"] == 1:
        status, why = "DROPPED", "leakage tier 1 (hypertension-specific)"
    elif n not in kept and s["leak"] == 2:
        status, why = "DROPPED", "leakage tier 2 (requires physical exam)"
    elif n not in kept and PRUNE and n in PRUNE_LIST:
        status, why = "DROPPED", "collinearity pruning (VIF/corr; see STEP 3b)"
    elif n not in kept:
        status, why = "DROPPED", "not available in this scenario"
    else:
        status, why = "KEPT", ""

    na_txt = {"mar": "MAR -> median/mode inside pipeline (+ indicator)",
              "explicit_na": "skip -> explicit 'NA_nao_aplicavel' level (or 0 + _isNA flag)"}
    rule = s["na_rule"]
    rows_map.append(dict(
        block=s["block"],
        derived_name=n,
        pns_code=s["code"],
        pns_label=pns_label.get(str(s["code"]).strip().upper(), s["label"]),
        our_label=s["label"],
        type=s["vtype"],
        recode=s["recode"],
        missing_raw=round(miss_raw.get(n, np.nan) * 100, 1) if n in miss_raw else np.nan,
        missing_after_tier1=round(miss_post.get(n, np.nan) * 100, 1) if n in miss_post else np.nan,
        na_handling=na_txt.get(rule, f"skip -> imputed {rule.split(':')[-1]} (logically implied)"),
        encoding=("standardised (LR) / raw (trees)" if s["vtype"] in ("num", "bin")
                  else "one-hot, min_frequency=20"),
        leakage_tier=s["leak"],
        status=status,
        drop_reason=why))

vmap = pd.DataFrame(rows_map).sort_values(["block", "derived_name"]).reset_index(drop=True)
vmap.to_csv(f"variable_map{SUFFIX}.csv", index=False)
res.to_csv(f"model_results{SUFFIX}.csv")
json.dump(best_params, open(f"best_hyperparameters{SUFFIX}.json", "w"), indent=2, default=str)

print(vmap.groupby(["block", "status"]).size().to_string())
print(f"\n  total declared: {len(vmap)} | kept: {(vmap.status=='KEPT').sum()} "
      f"| dropped: {(vmap.status=='DROPPED').sum()}")
print(f"\n  Saved: variable_map{SUFFIX}.csv, model_results{SUFFIX}.csv, best_hyperparameters{SUFFIX}.json, feature_importance{SUFFIX}.csv")
