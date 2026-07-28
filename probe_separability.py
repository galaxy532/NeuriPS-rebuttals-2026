"""
probe_separability.py -- Is Phi r-separable in the POPULATION, or only in sample?

Everything about alpha rests on r-separability (Def. r-separability):

    exists v such that  P(y v.r >= 1) = 1,

a statement about the distribution, with probability ONE. Every separability
number we have measured so far is instead an IN-SAMPLE fit, produced by a
classifier chosen for its correspondence to the manuscript rather than for
estimating a ceiling:

    LinearSVC(C=1e6, fit_intercept=False)

C = 1e6 is effectively no regularisation and there is no bias term. On the SAE
basis that combination reports frac_correct = 1.0000 on 8192 columns against
2897 training rows, and 0.6142 held out -- i.e. it memorises. A number produced
that way cannot answer "is the population separable"; it answers "can 8192 free
parameters shatter 2897 points", and the answer to that is always yes.

This script measures the ceiling properly, and it is deliberately NOT the
manuscript's estimator. It exists to bound what any linear rule could achieve on
this representation, so that a failure of r-separability downstream can be
attributed to the data rather than to our choice of solver.

WHAT IS DIFFERENT HERE
======================
  intercept       Included. Without one the boundary is forced through the
                  origin; the manuscript's model has no bias term, but that is a
                  modelling assumption, not a fact about the data, and it must
                  not be allowed to depress a ceiling estimate.
  regularisation  Tuned by inner cross-validation over a log grid of C, rather
                  than fixed at 1e6. This is the whole point: the ceiling is what
                  a WELL-FITTED linear rule achieves, not what an overfitted one
                  does.
  evaluation      Always on held-out rows, split stratified on (y, g) so all
                  four cells survive, repeated over seeds for a spread.
  per-group       Accuracy is reported per (y, g) cell and as worst-group, which
                  is the quantity the group-robustness literature reports for
                  Waterbirds and the one the manuscript's minority rates concern.

THE LEARNING CURVE IS THE DECISIVE PART
=======================================
A held-out accuracy of 0.87 has two possible readings, and they lead to opposite
conclusions:

  (a) the ceiling is real -- no linear rule on this representation separates,
      so r-separability fails in the population and alpha is not estimable on
      Waterbirds by anyone; or
  (b) we are simply short of data -- accuracy is still climbing at n = 2897 and
      would approach 1 given more.

Training on increasing fractions of the training half distinguishes them. A
curve that has flattened supports (a); one still rising supports (b). Reporting
a single accuracy cannot separate these, which is why the curve is here.

USAGE
=====
Run from the repository root.

    python probe_separability.py
    python probe_separability.py --sae --sae-l1 0.3 --sae-epochs 400 --tau 0.05

Output names are derived exactly as in analyze.py and where_is_the_signal.py:

    -> probe_test_raw.{json,md}
    -> probe_test_sae_l1-0.3_ep-400_tau-0.05.{json,md}

The unregularised no-intercept SVM is included as one extra row per block, so
the gap between "what we have been measuring" and "what is achievable" is
visible in the same table rather than across two documents.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common import FeatureBundle, fit_margin_direction, standardize
from identify_rs import sign_flip_identify, two_concept_identify

DEFAULTS = {
    "tau": 0.15,
    "purity": 0.60,
    "sae_l1": 0.03,
    "sae_epochs": 60,
    "n_seeds": 3,
    "inner_folds": 3,
}
_TAGS = {"tau": "tau", "purity": "pur", "sae_l1": "l1", "sae_epochs": "ep",
         "n_seeds": "seeds", "inner_folds": "folds"}
C_GRID = np.logspace(-4, 2, 7)
CURVE_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


def _cell_id(y, g):
    """A single integer per (y, g) cell, for stratification and reporting.

    Stratifying on y alone would let a split empty one of the 642-row minority
    cells, which would make a per-group number reflect the split rather than the
    classifier.
    """
    return (y > 0).astype(int) * 2 + (g > 0).astype(int)


def _tune_C(X, y, folds, seed):
    """Pick C by inner stratified CV on the TRAINING rows only.

    Selecting C on the evaluation rows would leak, and the leak flatters exactly
    the quantity we are trying to bound. Returns the C with the best mean inner
    accuracy.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    best_C, best_acc = C_GRID[0], -np.inf
    for C in C_GRID:
        accs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X[tr], y[tr])
            accs.append(clf.score(X[te], y[te]))
        m = float(np.mean(accs))
        if m > best_acc:
            best_acc, best_C = m, C
    return float(best_C), float(best_acc)


def _eval_block(X, y, g, n_seeds, folds, quantile):
    """Tuned probe + the unregularised no-intercept SVM, on the same splits."""
    cells = _cell_id(y, g)
    rows = {"probe": [], "svm_noreg": [], "C": [], "train_acc": []}
    per_cell = {c: [] for c in range(4)}

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        tr, te = [], []
        for c in range(4):
            idx = np.flatnonzero(cells == c)
            if idx.size == 0:
                continue
            rng.shuffle(idx)
            cut = idx.size // 2
            tr.append(idx[:cut])
            te.append(idx[cut:])
        tr, te = np.concatenate(tr), np.concatenate(te)

        C, _ = _tune_C(X[tr], y[tr], folds, seed)
        clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[tr], y[tr])
        rows["C"].append(C)
        rows["train_acc"].append(float(clf.score(X[tr], y[tr])))
        rows["probe"].append(float(clf.score(X[te], y[te])))
        pred = clf.predict(X[te])
        for c in range(4):
            m = cells[te] == c
            if m.any():
                per_cell[c].append(float(np.mean(pred[m] == y[te][m])))

        # The estimator the rest of the pipeline uses, on the identical split.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mf = fit_margin_direction(X[tr], y[tr], g[tr], quantile=quantile,
                                      seed=seed)
        rows["svm_noreg"].append(
            float(np.mean(y[te] * (X[te] @ mf.v_hat) > 0)))

    def ms(v):
        return {"mean": float(np.mean(v)), "sd": float(np.std(v))} if v else None

    cellwise = {f"cell{c}": ms(per_cell[c]) for c in range(4) if per_cell[c]}
    worst = min((v["mean"] for v in cellwise.values()), default=float("nan"))
    return {
        "n_cols": int(X.shape[1]),
        "C_median": float(np.median(rows["C"])),
        "train_acc": ms(rows["train_acc"]),
        "holdout_acc": ms(rows["probe"]),
        "holdout_acc_svm_noreg": ms(rows["svm_noreg"]),
        "per_cell": cellwise,
        "worst_group": worst,
    }


def _learning_curve(X, y, g, folds, seed, quantile):
    """Held-out accuracy against training-set size, on a fixed evaluation set.

    The evaluation half is held constant across rungs so that the only thing
    changing is the amount of training data. C is retuned at every rung, because
    the best regularisation strength genuinely depends on n and holding it fixed
    would confound the curve with a bad penalty at the small end.
    """
    del quantile
    cells = _cell_id(y, g)
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in range(4):
        idx = np.flatnonzero(cells == c)
        rng.shuffle(idx)
        cut = idx.size // 2
        tr.append(idx[:cut])
        te.append(idx[cut:])
    tr, te = np.concatenate(tr), np.concatenate(te)
    # `tr` was built cell by cell, so it arrives sorted by cell. A prefix of it
    # would be drawn entirely from one cell -- single-class at the small rungs,
    # and never representative at any rung. Shuffle once so that tr[:k] is a
    # random subsample of the training half, with every rung nested inside the
    # next (which is what makes the curve a curve rather than five unrelated
    # fits).
    rng.shuffle(tr)

    out = []
    for frac in CURVE_FRACTIONS:
        k = max(int(len(tr) * frac), 4 * 5)
        sub = tr[:k]
        if len(np.unique(y[sub])) < 2:
            continue
        C, _ = _tune_C(X[sub], y[sub], folds, seed)
        clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[sub], y[sub])
        out.append({"frac": float(frac), "n_train": int(len(sub)),
                    "C": float(C),
                    "train_acc": float(clf.score(X[sub], y[sub])),
                    "holdout_acc": float(clf.score(X[te], y[te]))})
    return out


def analyse(bundle, tau, purity, quantile, use_sae, sae_l1, sae_epochs,
            n_seeds, folds, curve_blocks) -> dict:
    phi = standardize(bundle.phi)
    y, g = bundle.y, bundle.g

    if use_sae:
        from identify_rs import fit_sae
        sae = fit_sae(phi, seed=0, l1=sae_l1, epochs=sae_epochs)
        acts = sae["activations"]
        source = standardize(acts)
        sae_info = {"avg_active": sae["avg_active"],
                    "var_explained": sae["var_explained"],
                    "d_hidden": sae["d_hidden"]}
    else:
        source = phi
        sae_info = None

    d = source.shape[1]
    out: dict = {"n": int(source.shape[0]), "d": int(d),
                 "basis": "sae" if use_sae else "raw", "tau": tau,
                 "sae": sae_info, "meta": bundle.meta, "blocks": {}}

    columns = {"full": np.arange(d)}

    rules = {}
    if bundle.place is not None:
        rules["two-concept"] = two_concept_identify(
            source, y, bundle.place, tau=tau, purity=purity)
    rules["sign-flip"] = sign_flip_identify(source, y, g, tau=tau)

    for name, ident in rules.items():
        idx_r = np.asarray(ident["idx_r"], dtype=int)
        idx_s = np.asarray(ident["idx_s"], dtype=int)
        lab = np.zeros(d, dtype=bool)
        lab[idx_r] = True
        lab[idx_s] = True
        for bname, cols in (("r", idx_r), ("s", idx_s),
                            ("weak", np.flatnonzero(~lab))):
            if cols.size >= 2:
                columns[f"{name}:{bname}"] = cols

    for name, cols in columns.items():
        out["blocks"][name] = _eval_block(source[:, cols], y, g, n_seeds,
                                          folds, quantile)

    # -- learning curves, PER BLOCK ------------------------------------------
    # Running the curve only on the full representation was misleading on the
    # SAE basis: there "full" is all 8192 columns, which is the worst-performing
    # block (held-out 0.7947, below its own 108-column r-subset at 0.9474)
    # because the signal is swamped by thousands of rarely-active units. Its
    # curve was still climbing, but extrapolating it says nothing about whether
    # the CAUSAL block has saturated -- and that is the question r-separability
    # turns on. Each block therefore gets its own curve.
    #
    # `curve_blocks` defaults to the full representation plus each rule's
    # r-block, since the s- and weak-block curves cost the same and answer
    # nothing about separability. Pass "all" to include them anyway.
    if curve_blocks == ["all"]:
        wanted = list(columns)
    else:
        wanted = [k for k in columns
                  if k == "full" or k.endswith(":r") or k in curve_blocks]
    out["learning_curves"] = {
        name: _learning_curve(source[:, columns[name]], y, g, folds, 0, quantile)
        for name in wanted
    }
    return out


def to_text(res: dict) -> str:
    L = [f"\n### Separability probe  basis={res['basis']}  n={res['n']}  "
         f"d={res['d']}  tau={res['tau']}\n"]
    if res.get("sae"):
        s = res["sae"]
        L.append(f"SAE: L0 = {s['avg_active']:.1f} of {s['d_hidden']}, "
                 f"var explained = {s['var_explained']:.4f}\n")

    L.append("Tuned L2 logistic probe WITH intercept, C chosen by inner CV on "
             "training rows only. Splits are stratified on (y, g); mean +/- sd "
             "over seeds.\n")
    L.append("| block | cols | median C | train acc | **held-out** | "
             "worst group | held-out, C=1e6 no-intercept SVM |")
    L.append("|---|---|---|---|---|---|---|")
    for name, b in res["blocks"].items():
        h, t = b["holdout_acc"], b["train_acc"]
        sv = b["holdout_acc_svm_noreg"]
        L.append(f"| {name} | {b['n_cols']} | {b['C_median']:g} "
                 f"| {t['mean']:.4f} | **{h['mean']:.4f}** +/- {h['sd']:.4f} "
                 f"| {b['worst_group']:.4f} | {sv['mean']:.4f} |")
    L.append("\n- the last column is the estimator the rest of the pipeline "
             "uses. Where it sits far below the tuned probe, the pipeline's "
             "separability numbers reflect the solver, not the representation.")

    L.append("\n**Learning curves, per block** (evaluation set held fixed; "
             "C retuned at each size)\n")
    L.append("Read the r-block curves, not the `full` one. On the SAE basis "
             "`full` is all 8192 columns, the worst-performing block, so its "
             "curve describes noise saturating rather than signal.\n")
    for name, lc in res.get("learning_curves", {}).items():
        if not lc:
            continue
        L.append(f"\n_{name}_\n")
        L.append("| train n | C | train acc | held-out acc |")
        L.append("|---|---|---|---|")
        for r in lc:
            L.append(f"| {r['n_train']} | {r['C']:g} | {r['train_acc']:.4f} "
                     f"| {r['holdout_acc']:.4f} |")
        if len(lc) >= 3:
            g1 = lc[-2]["holdout_acc"] - lc[-3]["holdout_acc"]
            g2 = lc[-1]["holdout_acc"] - lc[-2]["holdout_acc"]
            L.append(f"- last two gains: {g1:+.4f}, then {g2:+.4f}; "
                     f"final held-out {lc[-1]['holdout_acc']:.4f}")
            if g2 < 0.005 and lc[-1]["holdout_acc"] < 0.99:
                L.append("  -> **flattened below 1.0**: the ceiling is a "
                         "property of the representation, so r-separability "
                         "fails in the population for this block and alpha is "
                         "not estimable at q -> 0. Use the q-sweep in "
                         "analyze.py to state what IS estimable.")
            elif lc[-1]["holdout_acc"] >= 0.99:
                L.append("  -> approaching 1.0: consistent with r-separability "
                         "holding for this block")
            else:
                L.append("  -> still climbing: sample-limited, so the ceiling "
                         "is not yet established and no conclusion about "
                         "r-separability should be drawn from this block")
    return "\n".join(L)


def _auto_prefix(args, res):
    split = str(res.get("meta", {}).get("split", "")) or \
        os.path.splitext(os.path.basename(args.bundle))[0].split("_")[-1]
    parts = [f"probe_{split}_{res['basis']}"]
    keys = ["tau", "purity", "n_seeds", "inner_folds"]
    if args.sae:
        keys = ["sae_l1", "sae_epochs"] + keys
    for k in keys:
        v = getattr(args, k)
        if v != DEFAULTS[k]:
            parts.append(f"{_TAGS[k]}-{v:g}")
    return "_".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bound what any linear rule achieves on this "
                    "representation, to decide whether r-separability can hold.")
    ap.add_argument("--bundle", default="features_waterbirds_test.npz")
    ap.add_argument("--sae", action="store_true")
    ap.add_argument("--sae-l1", type=float, default=DEFAULTS["sae_l1"])
    ap.add_argument("--sae-epochs", type=int, default=DEFAULTS["sae_epochs"])
    ap.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    ap.add_argument("--purity", type=float, default=DEFAULTS["purity"])
    ap.add_argument("--quantile", type=float, default=0.01)
    ap.add_argument("--n-seeds", type=int, default=DEFAULTS["n_seeds"])
    ap.add_argument("--inner-folds", type=int, default=DEFAULTS["inner_folds"])
    ap.add_argument("--curve-blocks", nargs="+", default=["default"],
                    help="which blocks get a learning curve. Default is the "
                         "full representation plus each rule's r-block, since "
                         "those are the ones r-separability turns on. Pass "
                         "'all' for every block, or name blocks explicitly "
                         "(e.g. 'sign-flip:s').")
    args = ap.parse_args()

    bundle = FeatureBundle.load(args.bundle)
    res = analyse(bundle, tau=args.tau, purity=args.purity,
                  quantile=args.quantile, use_sae=args.sae,
                  sae_l1=args.sae_l1, sae_epochs=args.sae_epochs,
                  n_seeds=args.n_seeds, folds=args.inner_folds,
                  curve_blocks=args.curve_blocks)
    res["settings"] = {k: getattr(args, k) for k in DEFAULTS}
    res["settings"].update({"bundle": args.bundle, "sae": bool(args.sae)})

    text = to_text(res)
    print(text)
    prefix = _auto_prefix(args, res)
    with open(f"{prefix}.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    with open(f"{prefix}.md", "w") as fh:
        fh.write(text + "\n")
    print(f"\n[wrote {prefix}.json and {prefix}.md]")


if __name__ == "__main__":
    main()
