"""
analyze.py -- Main driver: from a frozen-feature bundle to rebuttal-ready numbers.

Run from the repository root, after extract_features.py. CPU only, except that
`--sae` fits a small autoencoder and will use a GPU if one is visible.

THE COMMANDS
============

    # 1. the default run: raw Phi coordinates
    python analyze.py
    #    -> results_test_raw.{json,md}

    # 2. the SAE basis, with the settings that actually sparsify the dictionary
    python analyze.py --sae --sae-l1 0.3 --sae-epochs 400
    #    -> results_test_sae_l1-0.3_ep-400.{json,md}

    # 3. the SAE basis at a lower classification threshold. Needed because a
    #    sparse dictionary has rarely-firing units whose 2x2 contrasts are
    #    small in pooled-SD units, so the default tau = 0.15 classifies almost
    #    nothing: the run above labels 47 of 8192 units and leaves 8145 "weak".
    #    tau = 0.05 is still ~3 standard errors (se_approx ~ 0.0158).
    python analyze.py --sae --sae-l1 0.3 --sae-epochs 400 --tau 0.05
    #    -> results_test_sae_l1-0.3_ep-400_tau-0.05.{json,md}

Nothing else needs to be typed: output names are derived (see below), so these
three commands never overwrite one another.

WHICH KNOB TO REACH FOR
=======================
`--sae-l1` and `--sae-epochs` control the DICTIONARY; `--tau` controls what is
CLASSIFIED once the dictionary exists. They fail in ways that look similar and
are not:

  avg_active near d_hidden/2  -> the L1 penalty has not moved the dictionary off
                                 its random initialisation. Raise --sae-epochs
                                 first (the default 60 is only ~720 optimiser
                                 steps at batch=512), then --sae-l1.
  avg_active low but n_weak    -> the dictionary is sparse and the THRESHOLD is
  near d_hidden                  now binding. Lower --tau. Raising --sae-l1
                                 further makes this worse, not better.

Raising --tau never fixes polysemanticity: tau gates `strength`, and a unit
responding to two concepts has HIGH strength with purity near 0.5, so a higher
threshold preferentially retains exactly the units it was meant to exclude.
Selectivity is the `--purity` gate's job, and it can only succeed on a
dictionary that contains selective units.

READING THE OUTPUT: THREE FAILURES THAT ARE REPORTED, NOT HIDDEN
================================================================
  "alpha: NOT ESTIMABLE"   The r-block is not separable at the working quantile,
                           so gamma-tilde_g does not exist. No regime is claimed.
                           A nan is never printed as "geometry dominates".
  "d* is VACUOUS"          dim K = 0, which d_s <= 2 d_r forces arithmetically.
                           The 1.0000 is an identity, not a measurement.
  "r-block is NOT          Only `frac_correct` of points are correctly
   separable"              classified. Separability and accuracy are now two
                           separate fields; they used to share one name.

The single switch chooses the BASIS -- which columns count as "features".
Both identification RULES are always run, on whichever basis was chosen:

    two-concept : the 2x2 decomposition over the (y, place) cells. Uses the
                  annotated spurious attribute directly. This is the primary
                  result.
    sign-flip   : the rule from the manuscript's Colored-MNIST experiment,
                  which needs only the group index. Reported alongside as an
                  independent cross-check.

Each rule gets the full downstream treatment -- coupling test, isotropy
diagnostics, alpha -- so the two can be compared end to end, and their
agreement is reported. Two rules reaching the same conclusion from different
information is much harder to dismiss than either alone.

Output names are derived, never typed
-------------------------------------
The output stem is built from the run itself: the split, the basis, and any
knob set away from its default. So

    python analyze.py                              -> results_test_raw.{json,md}
    python analyze.py --sae                         -> results_test_sae.{json,md}
    python analyze.py --sae --sae-l1 0.3 --sae-epochs 400
                                    -> results_test_sae_l1-0.3_ep-400.{json,md}

Two runs that differ in any parameter therefore land in different files
automatically, and a run repeated with identical settings overwrites itself,
which is what a redo should do. This is not a convenience: a sweep whose
outputs silently overwrote each other would leave numbers on disk that no
longer correspond to the settings you think produced them, and the results
files are the only record -- there is no separate log. Every knob is also
echoed inside the JSON, so a file can be identified from its contents alone
even if it is renamed.

Why the TEST split by default
-----------------------------
The coupling test conditions on the four (y, place) cells, and on the Waterbirds
TRAIN split those cells are 3498 / 184 / 56 / 1057. A 56-sample cell cannot
support a multi-output ridge from ~20 features: cross-validation keeps the
estimate from being spuriously high, but the result is unstable across seeds
while still printing to three decimals. The TEST split is built roughly balanced
(~2255/2255/642/642), so the smallest cell is ~642.

This is legitimate rather than a convenience: the coupling measurement is a
question about the frozen representation Phi -- does it entangle bird and
background information? -- and Phi is the same function whichever images pass
through it. The epsilon sweep is the opposite case: it concerns training
dynamics under group imbalance, so `eps_sweep.py` uses the TRAIN split, where
the imbalance actually lives.

Pipeline
--------
  1. Load the .npz bundle from extract_features.py and standardise Phi.
  2. Choose the basis: raw coordinates, or SAE activations if --sae.
  3. Run BOTH identification rules on that basis, and measure their agreement.
  4. For each rule: test for residual Phi_r -> Phi_s coupling WITHIN each
     (y, g) cell against a block-permutation null. This is the measurement that
     distinguishes feature-mediation from label-mediation, reported alongside
     the pooled (marginal) R^2 so the contrast is explicit.
  5. For each rule: fit the alignment operators A and B, measure the distance
     from the isotropic regime, and estimate alpha.

Everything printed is a number that can be pasted into a 10,000-character
OpenReview response; no figures are produced, by design.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import (
    FeatureBundle, compute_alpha, fit_margin_direction, fit_operators,
    iso_diagnostics, standardize, within_cell_coupling,
)
from identify_rs import (
    agreement, sign_flip_identify, tau_sensitivity, two_concept_identify,
)

RULES = ("two-concept", "sign-flip")

# Single source of truth for every tunable default. The argument parser reads
# its defaults from here, and `_auto_prefix` compares against the same values to
# decide which knobs deserve a filename tag. Defining them once is what keeps
# the two from drifting apart -- if a default were written out twice and only
# one copy updated, the stem would stop reflecting the run.
DEFAULTS = {
    "tau": 0.15,
    "purity": 0.60,
    "quantile": 0.01,
    "n_perm": 50,
    "min_cell": 200,
    "sae_l1": 0.03,
    "sae_epochs": 60,
}

# Short tags used in the output stem, e.g. sae_l1=0.3 -> "l1-0.3".
_TAGS = {
    "tau": "tau", "purity": "pur", "quantile": "q", "n_perm": "nperm",
    "min_cell": "mincell", "sae_l1": "l1", "sae_epochs": "ep",
}


def _downstream(source, y, g, ident, n_perm, quantile, seed, min_cell) -> dict:
    """Coupling test + isotropy diagnostics + alpha, for one r/s split.

    Factored out so that both identification rules receive exactly the same
    downstream treatment; any difference in the final numbers is then
    attributable to the split alone.
    """
    out: dict = {}
    if ident["n_r"] < 2 or ident["n_s"] < 2:
        out["error"] = (
            f"identification produced n_r={ident['n_r']}, n_s={ident['n_s']}; "
            "need at least 2 of each. Lower --tau, or try --sae if most "
            "features came out weak."
        )
        return out

    phi_r = source[:, ident["idx_r"]]
    phi_s = source[:, ident["idx_s"]]

    out["coupling"] = within_cell_coupling(
        phi_r, phi_s, y, g, n_perm=n_perm, seed=seed, min_cell=min_cell
    )

    mf = fit_margin_direction(phi_r, y, g, quantile=quantile)
    A_hat, B_hat = fit_operators(phi_r, phi_s, g)
    iso = iso_diagnostics(A_hat, B_hat, mf.v_hat, phi_r)
    out["margins"] = {
        "gamma_tilde_maj": mf.gam_tilde[0],
        "gamma_tilde_min": mf.gam_tilde[1],
        "orientation": mf.orientation,
        "separable_at_quantile": mf.separable_at_quantile,
        "frac_correct": mf.frac_correct,
        "separable_fraction": mf.separable_frac,
        "quantile_used": mf.quantile,
        "raw_min_margin": mf.gam_tilde_raw,
    }
    out["isotropy"] = {
        "mu_A": iso.mu_A, "mu_B": iso.mu_B, "mu": iso.mu,
        "defects": iso.defects, "defects_scaled": iso.defects_scaled,
        "degenerate": iso.degenerate, "max_defect": iso.max_defect,
        "d_star": iso.d_star, "d_star_relative": iso.d_star_relative,
        "d_star_sensitivity": iso.d_star_sensitivity,
        "d_star_informative": iso.d_star_informative,
        "dim_K": iso.dim_K, "attractive_condition_holds": iso.attractive,
        "d_s_ge_2d_r": bool(phi_s.shape[1] >= 2 * phi_r.shape[1]),
        "d_r": int(phi_r.shape[1]), "d_s": int(phi_s.shape[1]),
    }
    out["alpha"] = compute_alpha(mf, iso)
    return out


def run(bundle: FeatureBundle, tau: float = DEFAULTS["tau"],
        n_perm: int = DEFAULTS["n_perm"], quantile: float = DEFAULTS["quantile"],
        use_sae: bool = False, seed: int = 0,
        use_stored_split: bool = False, do_standardize: bool = True,
        purity: float = DEFAULTS["purity"], min_cell: int = DEFAULTS["min_cell"],
        sae_l1: float = DEFAULTS["sae_l1"],
        sae_epochs: int = DEFAULTS["sae_epochs"]) -> dict:
    phi = standardize(bundle.phi) if do_standardize else bundle.phi.copy()
    y, g = bundle.y, bundle.g
    res: dict = {
        "n": int(phi.shape[0]),
        "d": int(phi.shape[1]),
        "eps_empirical": float(np.mean(g == 1)),
        "meta": bundle.meta,
        "standardised": bool(do_standardize),
        "basis": "stored" if use_stored_split else ("sae" if use_sae else "raw"),
        "tau": tau,
        "rules": {},
    }

    # -- Validation path: the synthetic bundle carries the true split --------
    if use_stored_split:
        ident = {
            "idx_r": bundle.idx_r, "idx_s": bundle.idx_s,
            "n_r": int(bundle.idx_r.size), "n_s": int(bundle.idx_s.size),
            "n_weak": int(phi.shape[1] - bundle.idx_r.size - bundle.idx_s.size),
        }
        res["rules"]["stored"] = {
            "identification": {k: ident[k] for k in ("n_r", "n_s", "n_weak")},
            **_downstream(phi, y, g, ident, n_perm, quantile, seed, min_cell),
        }
        return res

    # -- Step 2: the basis ---------------------------------------------------
    if use_sae:
        from identify_rs import fit_sae
        sae = fit_sae(phi, seed=seed, l1=sae_l1, epochs=sae_epochs)
        # l1 and epochs are recorded next to the diagnostics they govern, so a
        # results file states the dictionary it was computed from. `avg_active`
        # is the L0 of the dictionary and is the number to read first: with
        # M ~ N(0, 1/d) and b = 0 exactly half the pre-activations are positive
        # at initialisation, so avg_active ~ d_hidden/2 means the L1 penalty has
        # not moved the dictionary off its random start and the "sparse" in
        # sparse autoencoder is not yet doing any work.
        res["sae"] = {"var_explained": sae["var_explained"],
                      "avg_active": sae["avg_active"],
                      "d_hidden": sae["d_hidden"],
                      "l1": sae_l1,
                      "epochs": sae_epochs,
                      "avg_active_at_init": sae["d_hidden"] / 2.0}
        source = sae["activations"]
    else:
        source = phi

    # -- Step 3: both rules on that basis ------------------------------------
    idents = {}
    if bundle.place is None:
        res["warning"] = (
            "no `place` annotation in this bundle, so only the sign-flip rule "
            "could be run. Re-run extract_features.py to enable the "
            "two-concept rule."
        )
    else:
        idents["two-concept"] = two_concept_identify(
            source, y, bundle.place, tau=tau, purity=purity)
    idents["sign-flip"] = sign_flip_identify(source, y, g, tau=tau)

    if len(idents) == 2:
        res["rule_agreement"] = agreement(
            idents["two-concept"], idents["sign-flip"], source.shape[1])

    res["tau_sensitivity"] = tau_sensitivity(source, y, g)

    # -- Steps 4-5: identical downstream treatment for each rule -------------
    for name, ident in idents.items():
        block = {"identification": {
            "n_r": ident["n_r"], "n_s": ident["n_s"], "n_weak": ident["n_weak"],
        }}
        if name == "two-concept":
            block["identification"].update({
                "cell_sizes": ident["cell_sizes"],
                "min_cell": ident["min_cell"],
                "se_approx": ident["se_approx"],
                "n_conjunction": ident["n_conjunction"],
                "n_strong": ident["n_strong"],
                "frac_conjunction": ident["frac_conjunction"],
                "n_conjunction_all": ident["n_conjunction_all"],
                "frac_conjunction_all": ident["frac_conjunction_all"],
                "conjunction_chance_rate_all": ident["conjunction_chance_rate_all"],
                "purity": purity,
            })
        block.update(_downstream(source, y, g, ident, n_perm, quantile,
                                 seed, min_cell))
        res["rules"][name] = block
    return res


def _rule_markdown(name: str, blk: dict) -> list[str]:
    L = [f"\n---\n\n## Rule: {name}\n"]
    idn = blk["identification"]
    L.append(f"- split: n_r = {idn['n_r']}, n_s = {idn['n_s']}, "
             f"weak = {idn['n_weak']}")
    if "cell_sizes" in idn:
        sizes = ", ".join(f"{k}: {v}" for k, v in idn["cell_sizes"].items())
        L.append(f"- (y, place) cell sizes: {sizes}  (smallest {idn['min_cell']})")
        L.append(f"- approx. SE of each contrast: {idn['se_approx']:.4f}")
        chance = idn.get("conjunction_chance_rate_all", 1.0 / 3.0)
        frac = idn.get("frac_conjunction", float("nan"))
        frac_all = idn.get("frac_conjunction_all", float("nan"))
        L.append("- conjunction-dominated features (|beta_int| the largest of "
                 "the three contrasts). Since y*place = 1 - 2g in Waterbirds, "
                 "beta_int is the group contrast, so these respond to GROUP "
                 "membership more strongly than to bird type or background -- "
                 "features the Phi = (r, s) partition has no slot for.")
        L.append(f"  - among the {idn.get('n_strong', 0)} features passing tau: "
                 f"**{idn['n_conjunction']}** ({frac:.1%})")
        L.append(f"  - over all columns, including those too weak to classify: "
                 f"{idn.get('n_conjunction_all', 0)} ({frac_all:.1%}), against a "
                 f"chance rate of {chance:.1%} under exchangeable contrasts")
        if np.isfinite(frac_all) and frac_all <= chance:
            L.append("  - the unrestricted rate is at or below chance, i.e. "
                     "consistent with noise, and is NOT evidence that the "
                     "(r, s) partition is violated")
        L.append("  - the two rates are not comparable to each other: passing "
                 "tau selects on beta_y and beta_p being large, which depresses "
                 "the restricted rate well below 1/3 even on pure noise (~0.11 "
                 "in testing). Only the unrestricted rate has 1/3 as its null; "
                 "the restricted one needs an estimated reference.")
    if "error" in blk:
        L.append(f"\n**Aborted:** {blk['error']}\n")
        return L

    c = blk["coupling"]
    L.append("\n**Within-(y,g)-cell coupling, Phi_r -> Phi_s "
             "(cross-validated R^2, block-permutation null)**\n")
    L.append(f"p-value floor with {c['n_perm']} permutations: "
             f"{c['p_value_floor']:.4f}. Cells below the minimum are refused, "
             f"not reported. Refused: {c['n_cells_refused']}; "
             f"reported but unreliable: {c['n_cells_unreliable']}.\n")
    L.append("| y | g | n | R2_cv | null mean | null q95 | p | note |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cell in c["cells"]:
        if not np.isfinite(cell["r2"]):
            L.append(f"| {cell['y']:+d} | {cell['g']} | {cell['n']} | - | - | - "
                     f"| - | {cell.get('note', '')} |")
        else:
            L.append(f"| {cell['y']:+d} | {cell['g']} | {cell['n']} | "
                     f"{cell['r2']:.3f} | {cell['null_mean']:+.4f} | "
                     f"{cell.get('null_q95', float('nan')):+.4f} | "
                     f"{cell['p']:.4f} | {cell.get('note', '')} |")
    L.append(f"\n- pooled (marginal) R2 = {c['pooled_r2']:.3f}  "
             f"vs mean within-cell R2 = {c['mean_within_cell_r2']:.3f}")
    L.append("- The pooled figure is inflated by label-mediation; the "
             "within-cell figure conditions on (y, g) and is the one that "
             "speaks to feature-mediation.")

    iso, m, a = blk["isotropy"], blk["margins"], blk["alpha"]
    L.append("\n**Distance from the isotropic regime**\n")
    L.append("| product | defect (angle) | defect (scaled) | degenerate |")
    L.append("|---|---|---|---|")
    for k in iso["defects"]:
        d = iso["defects"][k]
        L.append(f"| {k} | {d:.4f} | {iso['defects_scaled'][k]:.4f} "
                 f"| {'yes' if iso['degenerate'][k] else ''} |")
    L.append("\n- 'degenerate' marks products whose M v is negligible, where "
             "the angle-based defect is 0/0 and only the scaled column is "
             "meaningful.")
    L.append(f"- mu_A = {iso['mu_A']:.4f}, mu_B = {iso['mu_B']:.4f}, "
             f"mu = {iso['mu']:+.4f}")
    L.append(f"- attractive condition -1 <= mu < min(mu_A, mu_B): "
             f"{iso['attractive_condition_holds']}")
    L.append(f"- d_r = {iso['d_r']}, d_s = {iso['d_s']}, "
             f"d_s >= 2 d_r: {iso['d_s_ge_2d_r']}, dim K = {iso['dim_K']}")
    L.append(f"- d* = {iso['d_star']:.4f} (relative {iso['d_star_relative']:.4f}); "
             f"sensitivity to the rank cutoff: "
             + ", ".join(f"{k}: {v:.3f}" for k, v in iso["d_star_sensitivity"].items()))
    if not iso.get("d_star_informative", True):
        L.append("- **d\\* is VACUOUS on this run.** dim K = 0, which is forced "
                 "whenever d_s <= 2 d_r and the estimated operators are full "
                 "rank: M_stack is (2 d_r, d_s), so its rank saturates at d_s, "
                 "K = {0}, Pi_{K^perp} = I, and d*_relative = 1.0000 exactly as "
                 "arithmetic. It is restating the `d_s >= 2 d_r` flag, not "
                 "measuring a distance, and the rank-cutoff sweep above cannot "
                 "change that. Do not quote it.")

    L.append("\n**Margins and the exponent alpha**\n")
    L.append(f"- gamma-tilde_maj = {m['gamma_tilde_maj']:.4f}, "
             f"gamma-tilde_min = {m['gamma_tilde_min']:.4f} "
             f"(orientation: {m['orientation']}, "
             f"ess-inf proxy: {m['quantile_used']:.0%} quantile)")
    if m.get("separable_at_quantile", True):
        L.append(f"- r-block IS separable at the {m['quantile_used']:.0%} "
                 f"quantile; fraction correctly classified = "
                 f"{m['frac_correct']:.4f}")
    else:
        L.append(f"- **r-block is NOT separable at the {m['quantile_used']:.0%} "
                 f"quantile.** Only {m['frac_correct']:.2%} of points are "
                 "correctly classified by v, so gamma-tilde_g above are nan and "
                 "the orientation is undefined rather than 'mirror'. The "
                 "billions in `raw_min_margin` are an artefact of the 1e-9 "
                 "clamp, not measurements.")
    if a.get("estimable", True):
        L.append(f"- **alpha = {a['alpha']:.4f}** -> {a['regime']}; "
                 f"predicted minority exponent max(alpha,1) = "
                 f"{a['predicted_minority_exponent']:.4f}")
    else:
        L.append(f"- **alpha: NOT ESTIMABLE.** {a['failure_reason']}. No regime "
                 "is claimed for this rule.")
    return L


def to_markdown(res: dict) -> str:
    L = ["### Frozen-representation analysis\n"]
    L.append(f"- n = {res['n']}, d = {res['d']}, "
             f"empirical eps = {res['eps_empirical']:.4f}")
    L.append(f"- basis: **{res['basis']}**, tau = {res['tau']}")
    if "sae" in res:
        s = res["sae"]
        L.append(f"- SAE: d_hidden = {s['d_hidden']}, "
                 f"variance explained = {s['var_explained']:.4f}, "
                 f"avg active features = {s['avg_active']:.1f}")
    if "warning" in res:
        L.append(f"\n> **Warning:** {res['warning']}\n")
    if "rule_agreement" in res:
        ag = res["rule_agreement"]
        L.append(f"- agreement between the two rules: Jaccard r = "
                 f"{ag['jaccard_r']:.3f}, s = {ag['jaccard_s']:.3f}; "
                 f"concordance on the {ag['n_labelled_by_both']} features both "
                 f"label = {ag['concordance_on_shared']:.3f} "
                 f"({ag.get('n_discordant', 0)} discordant)")
        if ag.get("note"):
            L.append(f"  - **Caution:** {ag['note']}")
        elif np.isfinite(ag.get("discordance_upper_95", float("nan"))):
            L.append(f"  - zero discordant features, so by the rule of three the "
                     f"95% upper bound on the true discordance rate is "
                     f"{ag['discordance_upper_95']:.1%}")

    if res.get("tau_sensitivity"):
        L.append("\n**Threshold sensitivity (sign-flip rule)**\n")
        L.append("| tau | n_r | n_s |")
        L.append("|---|---|---|")
        for r in res["tau_sensitivity"]:
            L.append(f"| {r['tau']} | {r['n_r']} | {r['n_s']} |")

    for name in list(RULES) + ["stored"]:
        if name in res["rules"]:
            L += _rule_markdown(name, res["rules"][name])
    return "\n".join(L)


def _auto_prefix(args, res: dict) -> str:
    """Build the output stem from the run's own settings.

    The stem always carries the two things that change what is being measured
    rather than how precisely: the split and the basis. On top of that, every
    knob set AWAY FROM ITS DEFAULT contributes a short tag.

    Only non-default knobs are tagged, which is what keeps the common cases
    short -- a plain `python analyze.py` still writes `results_test_raw`, the
    name the earlier runs already used, so nothing that exists is orphaned.
    A sweep, by contrast, separates itself automatically:

        --sae --sae-l1 0.3 --sae-epochs 400  ->  results_test_sae_l1-0.3_ep-400

    Re-running an identical configuration overwrites its own file, which is the
    correct behaviour for a redo: the previous copy of that file was, by
    construction, computed from exactly the same settings.

    The SAE knobs are only consulted on the SAE basis. Tagging a raw-basis run
    with an l1 value would be actively misleading, since nothing in that run
    ever touched the autoencoder.
    """
    split = str(res.get("meta", {}).get("split", "")) or \
        os.path.splitext(os.path.basename(args.bundle))[0].split("_")[-1]
    parts = [f"results_{split}_{res['basis']}"]

    keys = ["tau", "purity", "quantile", "n_perm", "min_cell"]
    if args.sae:
        keys = ["sae_l1", "sae_epochs"] + keys
    for k in keys:
        v = getattr(args, k)
        if v != DEFAULTS[k]:
            parts.append(f"{_TAGS[k]}-{v:g}")

    if getattr(args, "no_standardize", False):
        parts.append("nostd")
    return "_".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Coupling test, isotropy diagnostics and alpha on a frozen "
                    "representation. Both identification rules are always run; "
                    "--sae selects the basis they run on.")
    ap.add_argument("--bundle", default="features_waterbirds_test.npz",
                    help=".npz from extract_features.py. Defaults to the TEST "
                         "split: its four (y, place) cells are roughly balanced "
                         "(~2255/2255/642/642), whereas the train split has a "
                         "56-sample cell that is too small to fit in.")
    ap.add_argument("--sae", action="store_true",
                    help="use SAE activations as the basis instead of the raw "
                         "Phi coordinates (needs torch). Worth trying only if "
                         "the raw run reports most features as weak.")
    ap.add_argument("--sae-l1", type=float, default=DEFAULTS["sae_l1"],
                    help="L1 coefficient on the SAE activations. Raise it if the "
                         "run reports avg_active anywhere near d_hidden/2, which "
                         "is the initialisation value and means the penalty has "
                         "not sparsified anything. Aim for avg_active in the "
                         "20-200 range at var_explained >= 0.8; if var_explained "
                         "collapses below ~0.7 the penalty has overshot.")
    ap.add_argument("--sae-epochs", type=int, default=DEFAULTS["sae_epochs"],
                    help="SAE training epochs. Note this is far fewer optimiser "
                         "steps than it sounds: at batch=512 a 5794-row bundle "
                         "gives 12 steps per epoch, so the default 60 is only "
                         "~720 Adam steps, where sparse autoencoders normally "
                         "need tens of thousands. If avg_active is stuck near "
                         "its initialisation value, raise this before concluding "
                         "that --sae-l1 is too small.")
    ap.add_argument("--tau", type=float, default=DEFAULTS["tau"],
                    help="minimum response strength for a feature to be classified")
    ap.add_argument("--purity", type=float, default=DEFAULTS["purity"],
                    help="two-concept: min |beta_y|/(|beta_y|+|beta_p|) to call "
                         "a feature r-type (and 1-purity for s-type)")
    ap.add_argument("--n-perm", type=int, default=DEFAULTS["n_perm"],
                    help="permutations for the coupling null; the p-value floor "
                         "is 1/(1+n_perm), so 50 gives 0.0196. The coupling test "
                         "dominates the runtime (n_perm x n_splits ridge fits per "
                         "cell, per rule), and every p-value reported so far has "
                         "sat at the floor, so the extra resolution of n_perm=200 "
                         "was not being used. Raise it only if a p-value needs to "
                         "be quoted below 0.02.")
    ap.add_argument("--min-cell", type=int, default=DEFAULTS["min_cell"],
                    help="refuse coupling cells smaller than this")
    ap.add_argument("--quantile", type=float, default=DEFAULTS["quantile"],
                    help="lower quantile used as the ess-inf proxy for margins")
    ap.add_argument("--use-stored-split", action="store_true",
                    help="validation only: use idx_r/idx_s stored in the bundle")
    ap.add_argument("--no-standardize", action="store_true",
                    help="validation only: skip per-coordinate standardisation")
    args = ap.parse_args()

    bundle = FeatureBundle.load(args.bundle)
    if bundle.idx_r.size == 0:
        bundle.idx_r = np.arange(bundle.phi.shape[1])
        bundle.idx_s = np.arange(0)

    res = run(bundle, tau=args.tau, n_perm=args.n_perm,
              quantile=args.quantile, use_sae=args.sae,
              use_stored_split=args.use_stored_split,
              do_standardize=not args.no_standardize,
              purity=args.purity, min_cell=args.min_cell,
              sae_l1=args.sae_l1, sae_epochs=args.sae_epochs)

    # Echo every setting into the JSON as well as into the filename, so a
    # results file remains self-identifying even if it is renamed or quoted
    # out of context.
    res["settings"] = {k: getattr(args, k) for k in DEFAULTS}
    res["settings"].update({"bundle": args.bundle, "sae": bool(args.sae)})

    prefix = _auto_prefix(args, res)

    with open(f"{prefix}.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    md = to_markdown(res)
    with open(f"{prefix}.md", "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[wrote {prefix}.json and {prefix}.md]")


if __name__ == "__main__":
    main()
