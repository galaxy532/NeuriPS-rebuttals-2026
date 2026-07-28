"""
where_is_the_signal.py -- Is the identification losing the causal signal?

A diagnostic, not an estimator. It writes no results file that feeds the
rebuttal; it answers one question and stops.

THE QUESTION
============
The full representation Phi separates the labels perfectly (frac_correct = 1.0
on the Waterbirds test split), yet every Phi_r our identification rules produce
fails r-separability. Something is losing the signal between Phi and Phi_r.
There are two candidate explanations and they call for different responses:

  (i)  IDENTIFICATION IS MISPLACING IT. The signal is carried by coordinates the
       rules label "s" or "weak", so the r-block never receives it. Fix the
       rules.

  (ii) NO COORDINATE SUBSET CARRIES IT. Separability is a JOINT property spread
       thinly across many coordinates, each of which looks mediocre on its own.
       Both rules screen features one at a time on marginal statistics
       (beta_y vs beta_p, or rho_0 vs rho_1), so a signal with that shape is
       invisible to any per-feature rule, however it is tuned. Then the problem
       is the univariate screen itself, not its threshold.

WHY NOT JUST SEARCH SUBSETS FOR ONE THAT SEPARATES
==================================================
Because separability is monotone: if a subset separates, so does every superset
(zero-pad the weight vector; the margin can only improve). The property is
upward-closed, so "the separable subset" is all of Phi, and the criterion
selects everything. Asking for MINIMAL separable subsets instead is
combinatorial, and -- more seriously -- selecting features by whether they
separate makes r-separability true by construction. It is the theory's
hypothesis about the causal block, so defining the block by it removes the
possibility of ever testing it. This script therefore measures WHERE the signal
sits relative to an identification fixed in advance, and never selects on
separability.

WHAT IT MEASURES
================
  A. Block-wise separability. Fit the margin direction on Phi_r alone, Phi_s
     alone, and the weak features alone. On the TEST split place is orthogonal
     to y (cells 642/642/2255/2255), so a purely spurious feature carries no
     marginal information about y there and Phi_s should NOT separate. If it
     does, the s-block is contaminated with causal signal.

  B. Mass decomposition of v_full. Split ||v_full||^2 across the three blocks.
     Reported as ENRICHMENT -- the share of mass divided by the share of
     coordinates -- because a block holding 60% of the mass while holding 60% of
     the columns has told you nothing. Enrichment > 1 means the block carries
     more of the direction than its size alone would give.

  C. How concentrated is the signal. Rank coordinates by |v_full[k]| and refit
     on the top m for a ladder of m. The smallest m that still separates is a
     greedy upper bound on how few coordinates suffice. If that number is in the
     hundreds, no ~50-feature identification can succeed and explanation (ii)
     holds regardless of the rule.

  D. Which labels those top coordinates carry. For the smallest separating m,
     how many of those coordinates each rule called r, s, or weak. This is the
     direct read on explanation (i).

  E. Held-out accuracy throughout. Every fit above is in-sample, and ranking
     coordinates by |v_full| and then testing on the same rows is a selection
     applied to its own evidence. The bundle is therefore split in half,
     stratified on (y, g); directions are fitted on half A and accuracy is also
     reported on half B. In-sample separability with chance-level held-out
     accuracy means the subset memorised rather than found anything.

USAGE
=====
Run from the repository root, after extract_features.py. CPU only.

    python where_is_the_signal.py                       # raw Phi coordinates
    python where_is_the_signal.py --sae --sae-l1 0.3 --sae-epochs 400 --tau 0.05

Runtime is dominated by the ladder in (C): each rung is one SVM on m columns,
and the largest rungs cost the most. Expect a few minutes on the raw basis.
Use --ladder to shorten it.

Output names are derived, exactly as in analyze.py: the stem carries the split
and the basis, plus a short tag for every knob set away from its default, so two
runs that differ in any parameter land in different files automatically and a
repeat of the same settings overwrites itself.

    python where_is_the_signal.py
        -> signal_test_raw.{json,md}
    python where_is_the_signal.py --sae --sae-l1 0.3 --sae-epochs 400 --tau 0.05
        -> signal_test_sae_l1-0.3_ep-400_tau-0.05.{json,md}

The ladder is tagged too (as L<rungs>-<largest>), because `smallest_separating_m`
is the smallest rung that happened to be TESTED: a run that skips a rung cannot
report it, so two ladders are not comparable and must not share a filename.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np

from common import FeatureBundle, fit_margin_direction, standardize
from identify_rs import sign_flip_identify, two_concept_identify

# Single source of truth for the defaults, as in analyze.py: the parser reads
# them from here and `_auto_prefix` compares against the same values, so the
# filename cannot drift away from the settings it claims to describe.
DEFAULTS = {
    "tau": 0.15,
    "purity": 0.60,
    "quantile": 0.01,
    "seed": 0,
    "sae_l1": 0.03,
    "sae_epochs": 60,
}
_TAGS = {"tau": "tau", "purity": "pur", "quantile": "q", "seed": "seed",
         "sae_l1": "l1", "sae_epochs": "ep"}
_DEFAULT_LADDER = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2048]


def _auto_prefix(args, res: dict) -> str:
    """Derive the output stem from the run's own settings. See analyze.py."""
    split = str(res.get("meta", {}).get("split", "")) or \
        os.path.splitext(os.path.basename(args.bundle))[0].split("_")[-1]
    parts = [f"signal_{split}_{res['basis']}"]

    keys = ["tau", "purity", "quantile", "seed"]
    if args.sae:
        keys = ["sae_l1", "sae_epochs"] + keys
    for k in keys:
        v = getattr(args, k)
        if v != DEFAULTS[k]:
            parts.append(f"{_TAGS[k]}-{v:g}")

    # The ladder decides which subset sizes were tried at all, and
    # `smallest_separating_m` is by definition the smallest rung TESTED. A run
    # on a coarser ladder can report a larger m purely because it skipped the
    # rung that would have separated, so ladders must not share a stem.
    lad = sorted(args.ladder)
    if lad != _DEFAULT_LADDER:
        parts.append(f"L{len(lad)}-{lad[-1]}")
    return "_".join(parts)


def _fit(phi, y, g, quantile, seed):
    """Margin direction on the given columns, warnings silenced.

    fit_margin_direction warns when the block does not separate at the working
    quantile. Here that is the measurement rather than a problem, so the warning
    is suppressed and the outcome read off `separable_at_quantile` instead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_margin_direction(phi, y, g, quantile=quantile, seed=seed)


def _holdout_acc(phi, y, idx_fit, idx_eval, v):
    """Fraction of held-out points on the correct side of the fitted direction.

    Guards against the failure mode in which a subset separates the rows it was
    chosen on and predicts at chance elsewhere.
    """
    del idx_fit
    return float(np.mean(y[idx_eval] * (phi[idx_eval] @ v) > 0))


def _split_halves(y, g, seed):
    """Two halves, stratified on (y, g) so both keep all four cells.

    An unstratified split can empty the 642-row minority cells, which would make
    the held-out number reflect the split rather than the direction.
    """
    rng = np.random.default_rng(seed)
    a, b = [], []
    for yy in np.unique(y):
        for gg in np.unique(g):
            idx = np.flatnonzero((y == yy) & (g == gg))
            rng.shuffle(idx)
            cut = idx.size // 2
            a.append(idx[:cut])
            b.append(idx[cut:])
    return np.concatenate(a), np.concatenate(b)


def analyse(bundle: FeatureBundle, tau: float, purity: float, quantile: float,
            use_sae: bool, sae_l1: float, sae_epochs: int, seed: int,
            ladder: list[int]) -> dict:
    phi = standardize(bundle.phi)
    y, g = bundle.y, bundle.g

    if use_sae:
        from identify_rs import fit_sae
        sae = fit_sae(phi, seed=seed, l1=sae_l1, epochs=sae_epochs)
        acts = sae["activations"]
        # See the long note in analyze.py: the encoder ends in a ReLU, so the
        # activations are non-negative and uncentred however `phi` was scaled
        # going in. Every margin fit here runs with fit_intercept=False, which
        # forces the boundary through the origin -- unworkable on non-negative
        # data, and the reason this diagnostic previously reported below-chance
        # accuracy (0.2761 held out on the top 10 coordinates) on the SAE basis.
        source = standardize(acts)
        sae_info = {"avg_active": sae["avg_active"],
                    "var_explained": sae["var_explained"],
                    "d_hidden": sae["d_hidden"],
                    "activations_standardised": True,
                    "n_dead": int((acts > 0).sum(axis=0).__eq__(0).sum()),
                    "n_near_constant": int((acts.std(axis=0) < 1e-12).sum())}
    else:
        source = phi
        sae_info = None

    d = source.shape[1]
    out: dict = {"n": int(source.shape[0]), "d": int(d),
                 "basis": "sae" if use_sae else "raw",
                 "tau": tau, "sae": sae_info, "meta": bundle.meta,
                 "ladder": [int(m) for m in ladder], "rules": {}}

    idx_fit, idx_eval = _split_halves(y, g, seed)

    # -- The reference direction: max-margin on the FULL representation -------
    mf_full = _fit(source, y, g, quantile, seed)
    mf_full_half = _fit(source[idx_fit], y[idx_fit], g[idx_fit], quantile, seed)
    v_full = mf_full.v_hat
    out["full"] = {
        "separable_at_quantile": mf_full.separable_at_quantile,
        "frac_correct": mf_full.frac_correct,
        "holdout_acc": _holdout_acc(source, y, idx_fit, idx_eval,
                                    mf_full_half.v_hat),
    }

    # -- (C) how few coordinates suffice, ranked by |v_full| ------------------
    order = np.argsort(-np.abs(v_full))
    rungs = []
    smallest_sep = None
    for m in [x for x in ladder if x <= d]:
        cols = order[:m]
        mf_m = _fit(source[:, cols], y, g, quantile, seed)
        mf_m_h = _fit(source[idx_fit][:, cols], y[idx_fit], g[idx_fit],
                      quantile, seed)
        acc_out = _holdout_acc(source[:, cols], y, idx_fit, idx_eval,
                               mf_m_h.v_hat)
        rungs.append({"m": int(m),
                      "separable_at_quantile": mf_m.separable_at_quantile,
                      "frac_correct": mf_m.frac_correct,
                      "holdout_acc": acc_out})
        if smallest_sep is None and mf_m.separable_at_quantile:
            smallest_sep = int(m)
    out["concentration"] = {"ladder": rungs,
                            "smallest_separating_m": smallest_sep}

    # -- per identification rule ---------------------------------------------
    rules = {}
    if bundle.place is not None:
        rules["two-concept"] = two_concept_identify(
            source, y, bundle.place, tau=tau, purity=purity)
    rules["sign-flip"] = sign_flip_identify(source, y, g, tau=tau)

    for name, ident in rules.items():
        idx_r = np.asarray(ident["idx_r"], dtype=int)
        idx_s = np.asarray(ident["idx_s"], dtype=int)
        labelled = np.zeros(d, dtype=bool)
        labelled[idx_r] = True
        labelled[idx_s] = True
        idx_w = np.flatnonzero(~labelled)

        blk: dict = {"n_r": int(idx_r.size), "n_s": int(idx_s.size),
                     "n_weak": int(idx_w.size), "blocks": {}}

        # (A) block-wise separability + (B) mass share and enrichment
        total_mass = float(v_full @ v_full)
        for bname, cols in (("r", idx_r), ("s", idx_s), ("weak", idx_w)):
            entry: dict = {"n_cols": int(cols.size)}
            if cols.size:
                mass = float(v_full[cols] @ v_full[cols])
                share_mass = mass / total_mass if total_mass > 0 else float("nan")
                share_cols = cols.size / d
                entry["mass_share"] = share_mass
                entry["col_share"] = share_cols
                entry["enrichment"] = (share_mass / share_cols
                                       if share_cols > 0 else float("nan"))
            if cols.size >= 2:
                mf_b = _fit(source[:, cols], y, g, quantile, seed)
                mf_b_h = _fit(source[idx_fit][:, cols], y[idx_fit],
                              g[idx_fit], quantile, seed)
                entry.update({
                    "separable_at_quantile": mf_b.separable_at_quantile,
                    "frac_correct": mf_b.frac_correct,
                    "holdout_acc": _holdout_acc(source[:, cols], y, idx_fit,
                                                idx_eval, mf_b_h.v_hat),
                })
            else:
                entry["note"] = "fewer than 2 columns; not fitted"
            blk["blocks"][bname] = entry

        # (D) labels carried by the top coordinates of v_full
        m = smallest_sep if smallest_sep is not None else min(d, max(ladder))
        top = set(order[:m].tolist())
        blk["top_m_composition"] = {
            "m": int(m),
            "n_labelled_r": len(top & set(idx_r.tolist())),
            "n_labelled_s": len(top & set(idx_s.tolist())),
            "n_weak": len(top & set(idx_w.tolist())),
            "frac_that_rule_calls_r": (len(top & set(idx_r.tolist())) / m
                                       if m else float("nan")),
        }
        rules_out = blk
        out["rules"][name] = rules_out
    return out


def to_text(res: dict) -> str:
    L = [f"\n### Where is the signal?  basis={res['basis']}  "
         f"n={res['n']}  d={res['d']}  tau={res['tau']}\n"]
    if res.get("sae"):
        s = res["sae"]
        L.append(f"SAE: L0 = {s['avg_active']:.1f} of {s['d_hidden']}, "
                 f"var explained = {s['var_explained']:.4f}, "
                 f"dead = {s.get('n_dead', '?')}, "
                 f"near-constant = {s.get('n_near_constant', '?')}, "
                 f"standardised = {s.get('activations_standardised', False)}\n")

    f = res["full"]
    L.append(f"FULL Phi: separable = {f['separable_at_quantile']}, "
             f"frac_correct = {f['frac_correct']:.4f}, "
             f"held-out acc = {f['holdout_acc']:.4f}")
    L.append("  (held-out acc is the honest number: in-sample separability with "
             "chance held-out accuracy would mean memorisation)\n")

    c = res["concentration"]
    L.append("How concentrated is the signal (coordinates ranked by |v_full|):")
    L.append("| top m | separable | frac_correct | held-out acc |")
    L.append("|---|---|---|---|")
    for r in c["ladder"]:
        L.append(f"| {r['m']} | {r['separable_at_quantile']} | "
                 f"{r['frac_correct']:.4f} | {r['holdout_acc']:.4f} |")
    sm = c["smallest_separating_m"]
    L.append(f"\n-> smallest separating m on this ladder: {sm}")
    if sm is None:
        L.append("   No rung separated. The signal is not concentrated in any "
                 "small coordinate set, so explanation (ii) holds and no "
                 "per-feature rule can recover it.")
    else:
        L.append(f"   If {sm} is far above the number of features an "
                 "identification rule keeps, explanation (ii) holds: the block "
                 "is too small to separate no matter which columns it picks.")

    for name, blk in res["rules"].items():
        L.append(f"\n--- rule: {name}  (n_r={blk['n_r']}, n_s={blk['n_s']}, "
                 f"weak={blk['n_weak']}) ---")
        L.append("| block | cols | mass share | enrichment | separable | "
                 "frac_correct | held-out |")
        L.append("|---|---|---|---|---|---|---|")
        for bname, e in blk["blocks"].items():
            if "frac_correct" not in e:
                L.append(f"| {bname} | {e['n_cols']} | - | - | - | - | - |")
                continue
            L.append(f"| {bname} | {e['n_cols']} | {e.get('mass_share', 0):.3f} "
                     f"| {e.get('enrichment', float('nan')):.2f} "
                     f"| {e['separable_at_quantile']} | {e['frac_correct']:.4f} "
                     f"| {e['holdout_acc']:.4f} |")
        L.append("- enrichment = mass share / column share. 1.00 means the block "
                 "holds exactly the share of v_full its size would predict, "
                 "i.e. it is not special.")
        t = blk["top_m_composition"]
        L.append(f"- of the top {t['m']} coordinates of v_full, this rule calls "
                 f"{t['n_labelled_r']} r, {t['n_labelled_s']} s, "
                 f"{t['n_weak']} weak "
                 f"({t['frac_that_rule_calls_r']:.1%} labelled r)")
        L.append("  -> a low r-fraction here is direct evidence for explanation "
                 "(i): the signal sits where the rule is not looking.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diagnose whether identification is losing the causal "
                    "signal, or whether no coordinate subset carries it.")
    ap.add_argument("--bundle", default="features_waterbirds_test.npz")
    ap.add_argument("--sae", action="store_true")
    ap.add_argument("--sae-l1", type=float, default=DEFAULTS["sae_l1"])
    ap.add_argument("--sae-epochs", type=int, default=DEFAULTS["sae_epochs"])
    ap.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    ap.add_argument("--purity", type=float, default=DEFAULTS["purity"])
    ap.add_argument("--quantile", type=float, default=DEFAULTS["quantile"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--ladder", type=int, nargs="+", default=_DEFAULT_LADDER,
                    help="subset sizes to test; rungs above d are skipped")
    args = ap.parse_args()

    bundle = FeatureBundle.load(args.bundle)
    res = analyse(bundle, tau=args.tau, purity=args.purity,
                  quantile=args.quantile, use_sae=args.sae,
                  sae_l1=args.sae_l1, sae_epochs=args.sae_epochs,
                  seed=args.seed, ladder=sorted(args.ladder))
    res["settings"] = {k: getattr(args, k) for k in DEFAULTS}
    res["settings"].update({"bundle": args.bundle, "sae": bool(args.sae),
                            "ladder": sorted(args.ladder)})

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
