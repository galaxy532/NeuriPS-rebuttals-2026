"""
analyze.py -- Main driver: from a frozen-feature bundle to rebuttal-ready numbers.

Run from the repository root, after extract_features.py. CPU only, cheap.

    # the usual run: TEST split, two-concept rule
    python analyze.py

    # more permutations for tighter p-values (the floor is 1/(1+n_perm))
    python analyze.py --n-perm 1000

    # the old rule, for comparison (needs only the group index)
    python analyze.py --rule sign-flip

    # if the raw coordinates are too polysemantic, use the SAE route (needs torch)
    python analyze.py --sae --tau 0.15

Writes `results.json` and `results.md`; the `.md` is the paste-ready table.

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
  1. Load the .npz bundle written by extract_features.py and standardise Phi.
  2. Split Phi into Phi_r and Phi_s. The default rule is the two-concept 2x2
     decomposition over (y, place), which uses the annotated spurious attribute
     directly; the sign-flip rule is run alongside as an independent cross-check.
  3. Test for residual Phi_r -> Phi_s coupling WITHIN each (y, g) cell against a
     block-permutation null. This is the measurement that distinguishes
     feature-mediation from label-mediation, and it is reported alongside the
     pooled (marginal) R^2 so the contrast is explicit.
  4. Fit the alignment operators A and B, measure how far the real
     representation is from the isotropic regime, and estimate alpha.
  5. Write results.json and results.md.

Everything printed is a number that can be pasted into a 10,000-character
OpenReview response; no figures are produced, by design.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import (
    FeatureBundle, compute_alpha, fit_margin_direction, fit_operators,
    iso_diagnostics, standardize, within_cell_coupling,
)
from identify_rs import (
    agreement, sign_flip_identify, tau_sensitivity, two_concept_identify,
)


def run(bundle: FeatureBundle, tau: float, n_perm: int, quantile: float,
        use_sae: bool = False, seed: int = 0, use_stored_split: bool = False,
        do_standardize: bool = True, rule: str = "two-concept",
        purity: float = 0.60, min_cell: int = 200) -> dict:
    phi = standardize(bundle.phi) if do_standardize else bundle.phi.copy()
    y, g = bundle.y, bundle.g
    res: dict = {
        "n": int(phi.shape[0]),
        "d": int(phi.shape[1]),
        "eps_empirical": float(np.mean(g == 1)),
        "meta": bundle.meta,
        "standardised": bool(do_standardize),
    }

    # -- Step 2: identify the r/s split -------------------------------------
    if use_stored_split:
        # Synthetic bundles carry the true split; using it isolates the
        # analysis path from the identification step so the two can be
        # validated independently.
        ident = {
            "idx_r": bundle.idx_r, "idx_s": bundle.idx_s,
            "n_r": int(bundle.idx_r.size), "n_s": int(bundle.idx_s.size),
            "n_weak": int(phi.shape[1] - bundle.idx_r.size - bundle.idx_s.size),
        }
        source = phi
    elif use_sae:
        from identify_rs import sae_identify
        ident = sae_identify(phi, y, g, tau=tau, seed=seed)
        res["sae"] = {"var_explained": ident["var_explained"],
                      "avg_active": ident["avg_active"],
                      "d_hidden": ident["d_hidden"]}
        source = ident["activations"]
    elif rule == "two-concept":
        if bundle.place is None:
            raise SystemExit(
                "the two-concept rule needs the `place` annotation, which is "
                "absent from this bundle. Re-run extract_features.py (bundles "
                "written before this rule existed do not carry it), or pass "
                "--rule sign-flip."
            )
        ident = two_concept_identify(phi, y, bundle.place, tau=tau, purity=purity)
        source = phi
    else:
        ident = sign_flip_identify(phi, y, g, tau=tau)
        source = phi

    res["identification"] = {
        "n_r": ident["n_r"], "n_s": ident["n_s"], "n_weak": ident["n_weak"],
        "tau": tau,
        "tau_sensitivity": ([] if use_stored_split
                            else tau_sensitivity(source, y, g)),
        "route": ("stored" if use_stored_split
                  else "sae" if use_sae else rule),
    }
    if rule == "two-concept" and not use_stored_split and not use_sae:
        res["identification"].update({
            "cell_sizes": ident["cell_sizes"],
            "min_cell": ident["min_cell"],
            "se_approx": ident["se_approx"],
            "n_conjunction": ident["n_conjunction"],
            "purity": purity,
        })
        # Independent corroboration: the sign-flip rule uses only the group
        # index, so agreement between the two is evidence neither is an artefact.
        flip = sign_flip_identify(phi, y, g, tau=tau)
        res["identification"]["cross_check_sign_flip"] = {
            "n_r": flip["n_r"], "n_s": flip["n_s"],
            **agreement(ident, flip, phi.shape[1]),
        }
    if ident["n_r"] < 2 or ident["n_s"] < 2:
        res["error"] = (
            f"identification produced n_r={ident['n_r']}, n_s={ident['n_s']}; "
            "need at least 2 of each. Lower tau or use the SAE route."
        )
        return res

    phi_r = source[:, ident["idx_r"]]
    phi_s = source[:, ident["idx_s"]]

    # -- Step 3: the decisive coupling measurement ---------------------------
    res["coupling"] = within_cell_coupling(
        phi_r, phi_s, y, g, n_perm=n_perm, seed=seed, min_cell=min_cell
    )

    # -- Step 4: operators, isotropy defect, alpha ---------------------------
    mf = fit_margin_direction(phi_r, y, g, quantile=quantile)
    A_hat, B_hat = fit_operators(phi_r, phi_s, g)
    iso = iso_diagnostics(A_hat, B_hat, mf.v_hat, phi_r)
    est = compute_alpha(mf, iso)

    res["margins"] = {
        "gamma_tilde_maj": mf.gam_tilde[0],
        "gamma_tilde_min": mf.gam_tilde[1],
        "orientation": mf.orientation,
        "separable_fraction": mf.separable_frac,
        "quantile_used": mf.quantile,
        "raw_min_margin": mf.gam_tilde_raw,
    }
    res["isotropy"] = {
        "mu_A": iso.mu_A, "mu_B": iso.mu_B, "mu": iso.mu,
        "defects": iso.defects, "max_defect": iso.max_defect,
        "d_star": iso.d_star, "d_star_relative": iso.d_star_relative,
        "dim_K": iso.dim_K, "attractive_condition_holds": iso.attractive,
        "d_s_ge_2d_r": bool(phi_s.shape[1] >= 2 * phi_r.shape[1]),
        "d_r": int(phi_r.shape[1]), "d_s": int(phi_s.shape[1]),
    }
    res["alpha"] = est
    return res


def to_markdown(res: dict) -> str:
    L: list[str] = []
    L.append("### Frozen-representation analysis\n")
    L.append(f"- n = {res['n']}, d = {res['d']}, empirical eps = {res['eps_empirical']:.4f}")
    if "error" in res:
        L.append(f"\n**Aborted:** {res['error']}\n")
        return "\n".join(L)

    idn = res["identification"]
    L.append(f"- rule: **{idn['route']}**; split at tau = {idn['tau']}: "
             f"n_r = {idn['n_r']}, n_s = {idn['n_s']}, weak = {idn['n_weak']}")
    if "cell_sizes" in idn:
        sizes = ", ".join(f"{k}: {v}" for k, v in idn["cell_sizes"].items())
        L.append(f"- (y, place) cell sizes: {sizes}  (smallest {idn['min_cell']})")
        L.append(f"- approx. SE of each contrast: {idn['se_approx']:.4f}; "
                 f"conjunction-dominated features: {idn['n_conjunction']}")
    if "cross_check_sign_flip" in idn:
        cc = idn["cross_check_sign_flip"]
        L.append(f"- cross-check vs sign-flip rule (n_r = {cc['n_r']}, "
                 f"n_s = {cc['n_s']}): Jaccard r = {cc['jaccard_r']:.3f}, "
                 f"s = {cc['jaccard_s']:.3f}, concordance on the "
                 f"{cc['n_labelled_by_both']} features both label = "
                 f"{cc['concordance_on_shared']:.3f}")
    L.append("")

    L.append("**Threshold sensitivity (sign-flip rule)**\n")
    L.append("| tau | n_r | n_s |")
    L.append("|---|---|---|")
    for r in idn["tau_sensitivity"]:
        L.append(f"| {r['tau']} | {r['n_r']} | {r['n_s']} |")

    c = res["coupling"]
    L.append("\n**Within-(y,g)-cell coupling, Phi_r -> Phi_s "
             "(cross-validated R^2, block-permutation null)**\n")
    L.append(f"p-value floor with {c['n_perm']} permutations: "
             f"{c['p_value_floor']:.4f}. Cells below n = 200 are refused, not "
             f"reported. Refused: {c['n_cells_refused']}; "
             f"reported but unreliable: {c['n_cells_unreliable']}.\n")
    L.append("| y | g | n | R2_cv | null mean | null q95 | p | note |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cell in c["cells"]:
        if not np.isfinite(cell["r2"]):
            L.append(f"| {cell['y']:+d} | {cell['g']} | {cell['n']} | - | - | - | - "
                     f"| {cell.get('note', '')} |")
        else:
            L.append(f"| {cell['y']:+d} | {cell['g']} | {cell['n']} | "
                     f"{cell['r2']:.3f} | {cell['null_mean']:+.4f} | "
                     f"{cell.get('null_q95', float('nan')):+.4f} | {cell['p']:.4f} "
                     f"| {cell.get('note', '')} |")
    L.append(f"\n- pooled (marginal) R2 = {c['pooled_r2']:.3f}  "
             f"vs mean within-cell R2 = {c['mean_within_cell_r2']:.3f}")
    L.append("- The pooled figure is inflated by label-mediation; the "
             "within-cell figure conditions on (y, g) and is the one that "
             "speaks to feature-mediation.")

    iso, m, a = res["isotropy"], res["margins"], res["alpha"]
    L.append("\n**Distance from the isotropic regime**\n")
    L.append("| product | relative eigenvector defect |")
    L.append("|---|---|")
    for k, v in iso["defects"].items():
        L.append(f"| {k} | {v:.4f} |")
    L.append(f"\n- mu_A = {iso['mu_A']:.4f}, mu_B = {iso['mu_B']:.4f}, mu = {iso['mu']:+.4f}")
    L.append(f"- attractive condition -1 <= mu < min(mu_A, mu_B): "
             f"{iso['attractive_condition_holds']}")
    L.append(f"- d_r = {iso['d_r']}, d_s = {iso['d_s']}, "
             f"d_s >= 2 d_r (reparametrisation budget): {iso['d_s_ge_2d_r']}, "
             f"dim K = {iso['dim_K']}")
    L.append(f"- minimum displacement to an isotropic reparametrisation: "
             f"d* = {iso['d_star']:.4f} (relative {iso['d_star_relative']:.4f})")

    L.append("\n**Margins and the exponent alpha**\n")
    L.append(f"- gamma-tilde_maj = {m['gamma_tilde_maj']:.4f}, "
             f"gamma-tilde_min = {m['gamma_tilde_min']:.4f} "
             f"(orientation: {m['orientation']}, "
             f"ess-inf proxy: {m['quantile_used']:.0%} quantile)")
    L.append(f"- r-block separable fraction = {m['separable_fraction']:.4f}")
    L.append(f"- **alpha = {a['alpha']:.4f}** -> {a['regime']}; "
             f"predicted minority exponent max(alpha,1) = "
             f"{a['predicted_minority_exponent']:.4f}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="features_waterbirds_test.npz",
                    help=".npz from extract_features.py. Defaults to the TEST "
                         "split: its four (y, place) cells are roughly balanced "
                         "(~2255/2255/642/642), whereas the train split has a "
                         "56-sample cell that is too small to fit in.")
    ap.add_argument("--rule", default="two-concept",
                    choices=["two-concept", "sign-flip"],
                    help="identification rule (default: two-concept, which uses "
                         "the `place` annotation directly)")
    ap.add_argument("--purity", type=float, default=0.60,
                    help="two-concept: min |beta_y|/(|beta_y|+|beta_p|) to call "
                         "a feature r-type (and 1-purity for s-type)")
    ap.add_argument("--min-cell", type=int, default=200,
                    help="refuse coupling cells smaller than this")
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--quantile", type=float, default=0.01,
                    help="lower quantile used as the ess-inf proxy for margins")
    ap.add_argument("--sae", action="store_true", help="use the SAE route (needs torch)")
    ap.add_argument("--use-stored-split", action="store_true",
                    help="use idx_r/idx_s stored in the bundle (synthetic bundles)")
    ap.add_argument("--no-standardize", action="store_true",
                    help="skip per-coordinate standardisation (exact-recovery tests)")
    ap.add_argument("--out-prefix", default="results")
    args = ap.parse_args()

    bundle = FeatureBundle.load(args.bundle)
    if bundle.idx_r.size == 0:
        # extract_features.py leaves the split empty; identification fills it.
        bundle.idx_r = np.arange(bundle.phi.shape[1])
        bundle.idx_s = np.arange(0)

    res = run(bundle, tau=args.tau, n_perm=args.n_perm,
              quantile=args.quantile, use_sae=args.sae,
              use_stored_split=args.use_stored_split,
              do_standardize=not args.no_standardize,
              rule=args.rule, purity=args.purity, min_cell=args.min_cell)

    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    md = to_markdown(res)
    with open(f"{args.out_prefix}.md", "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[wrote {args.out_prefix}.json and {args.out_prefix}.md]")


if __name__ == "__main__":
    main()
