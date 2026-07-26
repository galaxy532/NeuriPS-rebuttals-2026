"""
analyze.py -- Main driver: from a frozen-feature bundle to rebuttal-ready numbers.

Run from the repository root, after extract_features.py. CPU only, cheap.

    # the usual run: coupling test + isotropy defect + alpha on the train split
    python analyze.py

    # more permutations for tighter p-values (the floor is 1/(1+n_perm))
    python analyze.py --n-perm 1000

    # check whether the coupling also holds on the test split
    python analyze.py --bundle features_waterbirds_test.npz --out-prefix results_test

    # if the raw coordinates are too polysemantic, use the SAE route (needs torch)
    python analyze.py --sae --tau 0.15

Writes `results.json` and `results.md`; the `.md` is the paste-ready table.

Pipeline
--------
  1. Load the .npz bundle written by extract_features.py and standardise Phi.
  2. Split Phi into Phi_r and Phi_s by the group-conditional sign-flip rule
     (identify_rs.py), reporting the sensitivity to the threshold tau.
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
from identify_rs import sign_flip_identify, tau_sensitivity


def run(bundle: FeatureBundle, tau: float, n_perm: int, quantile: float,
        use_sae: bool = False, seed: int = 0, use_stored_split: bool = False,
        do_standardize: bool = True) -> dict:
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
    else:
        ident = sign_flip_identify(phi, y, g, tau=tau)
        source = phi

    res["identification"] = {
        "n_r": ident["n_r"], "n_s": ident["n_s"], "n_weak": ident["n_weak"],
        "tau": tau,
        "tau_sensitivity": ([] if use_stored_split
                            else tau_sensitivity(source, y, g)),
        "route": "stored" if use_stored_split else ("sae" if use_sae else "sign-flip"),
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
        phi_r, phi_s, y, g, n_perm=n_perm, seed=seed
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
    L.append(f"- r/s split at tau = {idn['tau']}: "
             f"n_r = {idn['n_r']}, n_s = {idn['n_s']}, weak = {idn['n_weak']}\n")

    L.append("**Threshold sensitivity**\n")
    L.append("| tau | n_r | n_s |")
    L.append("|---|---|---|")
    for r in idn["tau_sensitivity"]:
        L.append(f"| {r['tau']} | {r['n_r']} | {r['n_s']} |")

    c = res["coupling"]
    L.append("\n**Within-(y,g)-cell coupling, Phi_r -> Phi_s "
             "(cross-validated R^2, block-permutation null)**\n")
    L.append("| y | g | n | R2_cv | null mean | null q95 | p |")
    L.append("|---|---|---|---|---|---|---|")
    for cell in c["cells"]:
        if not np.isfinite(cell["r2"]):
            L.append(f"| {cell['y']:+d} | {cell['g']} | {cell['n']} | - | - | - | - |")
        else:
            L.append(f"| {cell['y']:+d} | {cell['g']} | {cell['n']} | "
                     f"{cell['r2']:.3f} | {cell['null_mean']:+.4f} | "
                     f"{cell.get('null_q95', float('nan')):+.4f} | {cell['p']:.4f} |")
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
    ap.add_argument("--bundle", default="features_waterbirds_train.npz",
                    help=".npz from extract_features.py "
                         "(default: features_waterbirds_train.npz)")
    ap.add_argument("--tau", type=float, default=0.2)
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
              do_standardize=not args.no_standardize)

    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    md = to_markdown(res)
    with open(f"{args.out_prefix}.md", "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[wrote {args.out_prefix}.json and {args.out_prefix}.md]")


if __name__ == "__main__":
    main()
