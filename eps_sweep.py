"""
eps_sweep.py -- The last-layer group-proportion sweep on frozen features.

This is the direct test of the manuscript's central prediction, in the setting
the manuscript itself points to as its main application: last-layer retraining
on a frozen representation.

What is being tested
--------------------
Below the transition (alpha < 1) the theorem predicts, for each group,

    E_{G_g}[1 - p_y]  ~  kappa_g / (eps_g z_t),

so the group proportion is a direct multiplier on learning speed and the
rescaled quantity  eps * err_min * z_t  should COLLAPSE onto a single constant
kappa_min across the whole sweep.

Above the transition (alpha > 1) the minority error becomes

    Theta( z_t^{-alpha} (ln z_t)^{alpha - 1} ),

which does not depend on eps at all, so the raw minority curves should coincide
across the sweep while the rescaled ones fan out.

The majority group obeys kappa_maj / ((1 - eps) z_t) in both regimes and acts as
a control: its empirical exponent should sit at 1 throughout.

Practical note on the horizon. These are asymptotic statements in z_t. The
empirical exponent approaches its predicted value slowly from below, so a run
that is too short will systematically under-report beta_min. Use `--report-drift`
to print the exponent measured over successive windows: what matters for the
rebuttal is that beta_min is still climbing toward max(alpha, 1) and that the
majority control sits at 1, not that a short run has already converged.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import (
    FeatureBundle, local_slope, logistic_gd, standardize, subsample_to_eps,
)


def sweep(bundle: FeatureBundle, eps_list, h: float, T: int, seed: int = 0,
          alpha_hat: float | None = None, n_ckpt: int = 60) -> dict:
    phi = standardize(bundle.phi)
    rows = []
    curves = {}
    m_exp = max(alpha_hat, 1.0) if alpha_hat is not None else 1.0
    for eps in eps_list:
        rng = np.random.default_rng(seed)
        idx = subsample_to_eps(bundle.y, bundle.g, eps, rng)
        X, y, g = phi[idx], bundle.y[idx], bundle.g[idx]
        tr = logistic_gd(X, y, g, h=h, T=T, n_ckpt=n_ckpt)
        curves[f"{eps:g}"] = {k: v.tolist() for k, v in tr.items()}

        b_min = local_slope(tr["z"], tr["err_min"])
        b_maj = local_slope(tr["z"], tr["err_maj"])
        # Drift of the exponent across the second and final thirds of the run.
        b_min_mid = local_slope(tr["z"][: 2 * len(tr["z"]) // 3], tr["err_min"][: 2 * len(tr["z"]) // 3])
        rows.append({
            "eps": eps,
            "n": int(len(idx)),
            "n_min": int((g == 1).sum()),
            "beta_min": b_min,
            "beta_min_earlier_window": b_min_mid,
            "beta_maj": b_maj,
            "err_min_final": float(tr["err_min"][-1]),
            "err_maj_final": float(tr["err_maj"][-1]),
            "z_final": float(tr["z"][-1]),
            "inv_min": float(eps * tr["err_min"][-1] * tr["z"][-1] ** m_exp),
            "inv_maj": float((1 - eps) * tr["err_maj"][-1] * tr["z"][-1]),
        })

    inv = np.array([r["inv_min"] for r in rows], dtype=float)
    inv_j = np.array([r["inv_maj"] for r in rows], dtype=float)
    return {
        "rows": rows,
        "curves": curves,
        "minority_invariant_cv": float(inv.std() / inv.mean()) if inv.mean() > 0 else float("nan"),
        "majority_invariant_cv": float(inv_j.std() / inv_j.mean()) if inv_j.mean() > 0 else float("nan"),
        "rescaling_exponent_used": m_exp,
        "h": h, "T": T,
    }


def to_markdown(res: dict) -> str:
    L = ["### Group-proportion sweep on frozen features "
         f"(full-batch GD, h = {res['h']}, T = {res['T']:,}, "
         f"z_T = {res['rows'][0]['z_final']:,.0f})\n"]
    m = res["rescaling_exponent_used"]
    L.append(f"| eps | n | n_min | beta_min | beta_min (earlier window) | beta_maj "
             f"| eps*err_min*z^{m:g} | (1-eps)*err_maj*z |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in res["rows"]:
        L.append(f"| {r['eps']:g} | {r['n']} | {r['n_min']} | {r['beta_min']:.3f} "
                 f"| {r['beta_min_earlier_window']:.3f} | {r['beta_maj']:.3f} "
                 f"| {r['inv_min']:.4g} | {r['inv_maj']:.4g} |")
    L.append(f"\n- minority invariant coefficient of variation across the sweep: "
             f"{res['minority_invariant_cv']:.3f} (0 = perfect collapse)")
    L.append(f"- majority invariant coefficient of variation: "
             f"{res['majority_invariant_cv']:.3f}")
    L.append("- The majority exponent is the control: theory predicts 1.0 "
             "regardless of regime.")
    L.append("- beta_min approaches max(alpha, 1) from below as z_T grows; "
             "compare the two windows to see the drift.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--eps", default="0.01,0.02,0.05,0.1,0.25",
                    help="comma-separated target minority fractions")
    ap.add_argument("--h", type=float, default=0.05, help="constant step size")
    ap.add_argument("--T", type=int, default=500_000)
    ap.add_argument("--alpha", type=float, default=None,
                    help="alpha-hat from analyze.py; sets the rescaling exponent")
    ap.add_argument("--out-prefix", default="eps_sweep")
    args = ap.parse_args()

    bundle = FeatureBundle.load(args.bundle)
    eps_list = [float(x) for x in args.eps.split(",")]
    res = sweep(bundle, eps_list, h=args.h, T=args.T, alpha_hat=args.alpha)

    with open(f"{args.out_prefix}.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    md = to_markdown(res)
    with open(f"{args.out_prefix}.md", "w") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[wrote {args.out_prefix}.json and {args.out_prefix}.md]")


if __name__ == "__main__":
    main()
