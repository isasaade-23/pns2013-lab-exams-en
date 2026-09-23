"""Emit 90_compare_results.ipynb: one notebook that gathers every run."""
import json
import sys

REPO = "https://github.com/isasaade-23/pns2013-lab-exams-en.git"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").split("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n").split("\n")}


C = []

C.append(md("""
# Comparing every run · PNS 2013

**FAPESP–Illinois undiagnosed NCD project.** This notebook fits nothing. It
reads what the model notebooks wrote, for all three outcomes, all four families
and all four variants, and produces the three things the group compares:

1. **One table** — outcome x family x variant, with discrimination, calibration
   and both operating points.
2. **One AUC figure** — a panel per outcome, so families and variants are read
   against each other without confusing a 32%-prevalence outcome with a
   3.8% one.
3. **The confusion matrices** — a grid per outcome, at the screening point,
   which is where the cost of the instrument shows.

## Where it reads from

The three outcome folders on the shared drive, which is where the zips are
uploaded:

```
02_analysis/outputs/models/hypertension
02_analysis/outputs/models/diabetes
02_analysis/outputs/models/cholesterol
```

Any zip found there is extracted in place, and any run folder already extracted
is read as it is. Running this twice changes nothing.

**Duplicates resolve by `run_utc`**, the stamp the run itself wrote, not by file
date: a zip uploaded later is not necessarily a run made later. Anything under
`_superseded/` is ignored, since those were built on an earlier registry and are
not comparable.
"""))

C.append(md("""
---
# 1 · Setup

Mount the Drive, or point `BASE` at a local copy of the same folders.
"""))
C.append(code("""
import os, glob, zipfile, datetime, re
import numpy as np, pandas as pd

MOUNT_DRIVE = True
DRIVE = "/content/drive/MyDrive/FAPESP_Illinois"

if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive not mounted:", e)

BASE = (f"{DRIVE}/02_analysis/outputs/models" if os.path.isdir(DRIVE)
        else "outputs/models")
OUT = os.path.join(BASE, "_comparison")
os.makedirs(OUT, exist_ok=True)

OUTCOMES = ["hypertension", "diabetes", "cholesterol"]
print("reading from", BASE)
for o in OUTCOMES:
    p = os.path.join(BASE, o)
    print(f"  {o:<13} {'exists' if os.path.isdir(p) else 'MISSING'}")
"""))

C.append(md("""
## 1.1 · Extract whatever is there

**What to look for.** One line per zip. A zip whose `metrics_*.csv` is already
on disk is skipped rather than re-extracted, so this cell is safe to re-run
after every upload.
"""))
C.append(code("""
extracted, skipped = 0, 0
for outcome in OUTCOMES:
    folder = os.path.join(BASE, outcome)
    if not os.path.isdir(folder):
        continue
    for z in sorted(glob.glob(os.path.join(folder, "*.zip"))):
        with zipfile.ZipFile(z) as f:
            names = f.namelist()
            metric = [n for n in names if "metrics_" in n and n.endswith(".csv")]
            already = metric and os.path.exists(os.path.join(folder, metric[0]))
            if already:
                skipped += 1
                continue
            f.extractall(folder)
            extracted += 1
            print(f"extracted {os.path.basename(z)}")
print(f"\\n{extracted} extracted, {skipped} already on disk")
"""))

C.append(md("""
---
# 2 · The table

Every `metrics_*.csv` under the three outcome folders, with `_superseded`
excluded. Family and variant come from the folder the file sits in, so a run
exported before variants existed reads as `base`.
"""))
C.append(code("""
rows = []
for outcome in OUTCOMES:
    for f in glob.glob(os.path.join(BASE, outcome, "*", "metrics_*.csv")):
        if "_superseded" in f:
            continue
        d = pd.read_csv(f)
        folder = os.path.basename(os.path.dirname(f))       # family or family_variant
        fam, _, var = folder.partition("_")
        d["outcome"] = outcome
        d["family"] = fam
        # Where the variant came from matters later: a file that declares it is
        # a one-row export from a run with variants, and a file that does not is
        # an older export holding both the default and the tuned fit.
        if "variant" in d and d["variant"].notna().any():
            d["declared_variant"] = True
        else:
            d["variant"] = var or "base"
            d["declared_variant"] = False
        d["source"] = os.path.relpath(f, BASE)
        rows.append(d)

if not rows:
    raise SystemExit("no metrics files found: upload the zips first, then re-run")

raw = pd.concat(rows, ignore_index=True)

# one row per outcome x family x variant x configuration, most recent run wins
raw["run_utc"] = pd.to_datetime(raw["run_utc"], errors="coerce")
raw = (raw.sort_values("run_utc")
          .drop_duplicates(["outcome", "family", "variant", "model"], keep="last"))

print(f"{len(raw)} rows from {raw['source'].nunique()} files")
print(raw.groupby(["outcome", "family"])["variant"].apply(lambda s: ", ".join(sorted(set(s)))).to_string())
"""))

C.append(md("""
## 2.1 · The comparison table

One line per outcome, family and variant, keeping the tuned configuration of
each. `calib_intercept` and `calib_slope` are the columns that answer whether a
resampling variant paid for itself: a variant that moved AUC by nothing and
pushed the intercept away from zero made the model worse, not better.
"""))
C.append(code("""
# Which rows are the reported ones.
#
# A run exported with variants writes one file per variant, each holding a
# single row, and the variant column says which. Those rows are all kept, with
# one exception: "untuned", the library-default fit, which is a comparison
# rather than a result.
#
# A run exported before the variants existed writes both the default and the
# tuned fit into one file and carries no variant column. There the tuned row is
# the reported one, and the default is dropped by its label.
#
# The label cannot be used on the newer rows: TabPFN has no search, so its
# reported model is called "TabPFN (default)" and filtering on the word would
# throw the whole family away.
declared = raw["declared_variant"].fillna(False).astype(bool)
labelled_default = raw["model"].str.contains("default|untuned", case=False, na=False)

best = raw[(declared & (raw["variant"] != "untuned"))
           | (~declared & ~labelled_default)].copy()
print(f"{len(best)} reported rows, {len(raw) - len(best)} comparison rows set aside")

cols = ["outcome", "family", "variant", "AUC_test", "AUC_lo", "AUC_hi", "PR_AUC",
        "Brier", "calib_intercept", "calib_slope", "prevalence",
        "youden_Sensitivity", "youden_Specificity", "youden_PPV",
        "screen90_Sensitivity", "screen90_Specificity", "screen90_PPV",
        "n_train_plus_test", "n_predictors", "run_utc"]
# One line per cell of the grid. Two runs of the same cell can carry different
# model labels, as TabPFN does when the second configuration changed from an
# ensemble to a single pass, so the label cannot be part of the key here: the
# cell is the outcome, the family and the variant, and the most recent run wins.
best = (best.sort_values("run_utc")
            .drop_duplicates(["outcome", "family", "variant"], keep="last"))

table = (best[[c for c in cols if c in best]]
         .sort_values(["outcome", "family", "variant"])
         .reset_index(drop=True))
display(table.round(4))

stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M")
table.to_csv(os.path.join(OUT, f"comparison_{stamp}.csv"), index=False)
try:
    table.to_excel(os.path.join(OUT, f"comparison_{stamp}.xlsx"), index=False)
except Exception as e:
    print("xlsx not written:", e)
print("written to", OUT)
"""))

C.append(md("""
## 2.2 · Did the variants change anything?

AUC against `base`, per outcome and family. Resampling does not create
information, so the expected reading is a column of near-zeros in `d_AUC` and a
calibration intercept that moved.
"""))
C.append(code("""
# A run exported before the variants existed carries neither the calibration
# columns nor more than one variant, so each block checks before it reads.
values = [c for c in ("AUC_test", "calib_intercept")
          if c in table and table[c].notna().any()]
piv = table.pivot_table(index=["outcome", "family"], columns="variant",
                        values=values)
variants = list(table["variant"].unique())

if ("AUC_test", "base") in piv and len(variants) > 1:
    delta = pd.DataFrame({v: piv[("AUC_test", v)] - piv[("AUC_test", "base")]
                          for v in variants if ("AUC_test", v) in piv})
    print("change in test AUC against base")
    display(delta.round(3))
else:
    print("one variant only, nothing to compare against base yet")

if "calib_intercept" in values:
    cal = pd.DataFrame({v: piv[("calib_intercept", v)]
                        for v in variants if ("calib_intercept", v) in piv})
    print("\\ncalibration intercept, 0 is calibrated")
    display(cal.round(2))
else:
    print("\\nno calibration columns in these runs: they predate them. Re-run to "
          "get the intercept and slope, which are what separate the variants")
"""))

C.append(md("""
---
# 3 · The AUC figure

One panel per outcome, one row per family, one marked interval per variant.

**Why dots and intervals rather than bars.** A bar carries an implicit zero
baseline, and zero is meaningless for AUC: 0.5 is the floor, which is where the
dashed line sits. The width of each interval is the bootstrap uncertainty, and
in diabetes it is wide enough to swallow most of the differences between
families.
"""))
C.append(code("""
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

# categorical slots, fixed order, validated for colour-vision deficiency
COLOUR = {"base": "#2a78d6", "smote": "#eb6834",
          "bagging": "#1baf7a", "calib": "#eda100"}
ORDER = ["base", "smote", "bagging", "calib"]

outs = [o for o in OUTCOMES if o in set(table["outcome"])]
fig, axes = plt.subplots(1, len(outs), figsize=(4.4 * len(outs), 4.0), sharex=True)
axes = np.atleast_1d(axes)

for ax, outcome in zip(axes, outs):
    d = table[table["outcome"] == outcome]
    fams = sorted(d["family"].unique())
    variants = [v for v in ORDER if v in set(d["variant"])]
    for i, fam in enumerate(fams):
        for j, var in enumerate(variants):
            r = d[(d["family"] == fam) & (d["variant"] == var)]
            if r.empty:
                continue
            r = r.iloc[0]
            y = i + (j - (len(variants) - 1) / 2) * 0.18
            ax.plot([r["AUC_lo"], r["AUC_hi"]], [y, y],
                    color=COLOUR.get(var, "#777"), lw=2, solid_capstyle="round",
                    alpha=0.55, zorder=2)
            ax.plot(r["AUC_test"], y, "o", ms=7, color=COLOUR.get(var, "#777"),
                    markeredgecolor="white", markeredgewidth=1.4, zorder=3)
    ax.axvline(0.5, color="#999", ls="--", lw=0.9, zorder=1)
    ax.set_yticks(range(len(fams)), fams)
    ax.set_ylim(-0.6, len(fams) - 0.4)
    ax.set_xlabel("Test AUC (95% bootstrap interval)")
    prev = d["prevalence"].iloc[0] if "prevalence" in d else np.nan
    n = int(d["n_train_plus_test"].iloc[0]) if "n_train_plus_test" in d else 0
    ax.set_title(f"{outcome}\\n{n:,} people · {prev:.1%} with the outcome",
                 fontsize=9)
    ax.grid(axis="x", color="#e6e6e3", lw=0.7)
    ax.set_axisbelow(True)

handles = [plt.Line2D([], [], marker="o", ls="-", color=COLOUR[v], ms=7,
                      markeredgecolor="white", label=v)
           for v in ORDER if v in set(table["variant"])]
axes[-1].legend(handles=handles, frameon=False, loc="lower right", title="variant")
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"auc_by_outcome_{stamp}.png"))
plt.show()
"""))

C.append(md("""
---
# 4 · Confusion matrices

At the screening point, where sensitivity is held near 90%. The counts come from
the metrics file, so nothing is refitted and nothing is approximated.

**How to read a row.** Of the people who have the outcome, the right-hand cell
is how many the model would flag; of the people who do not, the right-hand cell
is how many it would flag anyway. The second number is what a screening
programme pays for the first.
"""))
C.append(code("""
POINT = "screen90"        # or "youden"

for outcome in outs:
    d = table[table["outcome"] == outcome]
    fams = sorted(d["family"].unique())
    variants = [v for v in ORDER if v in set(d["variant"])]
    fig, axes = plt.subplots(len(fams), len(variants),
                             figsize=(2.05 * len(variants), 2.0 * len(fams)),
                             squeeze=False)
    for i, fam in enumerate(fams):
        for j, var in enumerate(variants):
            ax = axes[i][j]
            src = best[(best["outcome"] == outcome) & (best["family"] == fam)
                       & (best["variant"] == var)]
            if src.empty or f"{POINT}_TP" not in src:
                ax.axis("off")
                continue
            r = src.iloc[0]
            m = np.array([[r[f"{POINT}_TN"], r[f"{POINT}_FP"]],
                          [r[f"{POINT}_FN"], r[f"{POINT}_TP"]]], dtype=float)
            share = m / m.sum(axis=1, keepdims=True)
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{int(m[a, b]):,}\\n{share[a, b]:.0%}",
                            ha="center", va="center", fontsize=7,
                            color="white" if share[a, b] > 0.55 else "#0b0b0b")
            ax.set_xticks([0, 1], ["pred -", "pred +"], fontsize=7)
            ax.set_yticks([0, 1], ["true -", "true +"], fontsize=7)
            if i == 0:
                ax.set_title(var, fontsize=9)
            if j == 0:
                ax.set_ylabel(fam, fontsize=9)
    fig.suptitle(f"{outcome} · confusion at the {POINT} operating point", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"confusion_{outcome}_{POINT}_{stamp}.png"))
    plt.show()
"""))

C.append(md("""
---
# 5 · What is still missing

Prints the cells of the grid that have no run yet, so the next session knows
what to launch rather than guessing.
"""))
C.append(code("""
FAMILIES = ["logreg", "rf", "xgboost", "tabpfn"]
have = set(zip(table["outcome"], table["family"], table["variant"]))
missing = [(o, f, v) for o in OUTCOMES for f in FAMILIES for v in ORDER
           if (o, f, v) not in have]

if missing:
    print(f"{len(missing)} of {len(OUTCOMES) * len(FAMILIES) * len(ORDER)} cells missing:")
    for o, f, v in missing:
        print(f"  {o:<13} {f:<8} {v}")
else:
    print("the grid is complete")

print("\\neverything written to", OUT)
"""))

nb = {"cells": C,
      "metadata": {"colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    src = c["source"]
    c["source"] = [l + "\n" for l in src[:-1]] + [src[-1]]

out = sys.argv[1] if len(sys.argv) > 1 else "90_compare_results.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", out, len(C), "cells")
