"""
common.py -- Core estimators for the rebuttal experiments.

Every quantity below is named after the object it estimates in the manuscript,
and each function docstring states the equation it implements. Nothing is
introduced as an unexplained alias.

Notation map (manuscript -> code)
---------------------------------
    Phi(x) = (r, s)              -> phi = [phi_r | phi_s]
    v (Def. r-separability)      -> v_svm  (margin-1 scaling), v_hat (unit norm)
    gamma-tilde_g (r-margins)    -> gam_tilde[g]
    A, B (alignment operators)   -> A_hat, B_hat
    mu_A, mu_B, mu               -> mu_A, mu_B, mu
    alpha = gam_min(1+mu)/(1+mu_A) -> alpha
    z_t = sum_k h_k              -> z (cumulative learning steps)
    E_{G_g}[1 - p_y]             -> per-group soft error

Two normalisations of the margin direction are used, and they are NOT
interchangeable:

  * v_svm  solves  min ||theta||^2  s.t.  y theta.r >= 1  (Def. r-separability).
           Group r-margins gamma-tilde_g are read off THIS scaling, so that
           min_g gamma-tilde_g = 1 as the manuscript's WLOG requires.
  * v_hat = v_svm / ||v_svm||  is the UNIT vector. The isotropic-regime
           eigenvalue conditions (Def. Isotropic Regime) are statements about a
           unit eigenvector, so mu_A = ||A v_hat||^2 etc. are computed with
           v_hat. Using v_svm here would rescale all three eigenvalues by
           ||v_svm||^2 and silently corrupt alpha.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------
# Feature bundle
# --------------------------------------------------------------------------


@dataclass
class FeatureBundle:
    """A frozen representation together with labels and group labels.

    Attributes
    ----------
    phi : (N, d) float64
        The frozen representation Phi(x).
    y : (N,) int, values in {-1, +1}
        Binary label, in the +/-1 convention used throughout the manuscript.
    g : (N,) int, values in {0, 1}
        Group index. g == 0 is the majority group G_maj (s = A r + xi);
        g == 1 is the minority group G_min (s = B r + xi).
    idx_r, idx_s : int arrays
        Column indices of phi identified as causal (r-type) and spurious
        (s-type). For synthetic data these are known by construction; for real
        data they come from the SAE sign-flip rule in identify_rs.py.
    """

    phi: np.ndarray
    y: np.ndarray
    g: np.ndarray
    idx_r: np.ndarray
    idx_s: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def phi_r(self) -> np.ndarray:
        return self.phi[:, self.idx_r]

    @property
    def phi_s(self) -> np.ndarray:
        return self.phi[:, self.idx_s]

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, phi=self.phi, y=self.y, g=self.g,
            idx_r=self.idx_r, idx_s=self.idx_s,
            meta=np.array(repr(self.meta), dtype=object),
        )

    @staticmethod
    def load(path: str) -> "FeatureBundle":
        z = np.load(path, allow_pickle=True)
        meta = {}
        if "meta" in z:
            try:
                meta = eval(str(z["meta"].item()))  # noqa: S307 - our own file
            except Exception:
                meta = {}
        return FeatureBundle(
            phi=z["phi"].astype(np.float64),
            y=z["y"].astype(int),
            g=z["g"].astype(int),
            idx_r=z["idx_r"].astype(int),
            idx_s=z["idx_s"].astype(int),
            meta=meta,
        )


def standardize(phi: np.ndarray) -> np.ndarray:
    """Centre and scale each coordinate to unit variance.

    The manuscript assumes compactly supported features (Assumption: bounded
    features). Standardising does not change any of the *dimensionless*
    quantities we estimate -- alpha is a ratio of margins to eigenvalues and is
    invariant to a common rescaling of r and s -- but it makes the ridge fits
    and the logistic GD numerically well conditioned.
    """
    mu = phi.mean(axis=0, keepdims=True)
    sd = phi.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (phi - mu) / sd


# --------------------------------------------------------------------------
# Step 1 -- the margin direction v and the group r-margins gamma-tilde_g
# --------------------------------------------------------------------------


@dataclass
class MarginFit:
    v_svm: np.ndarray        # margin-1 scaling: y v.r >= 1 on (almost) all data
    v_hat: np.ndarray        # unit norm, used for the eigenvalue conditions
    gam_tilde: dict          # {0: gamma-tilde_maj, 1: gamma-tilde_min}, min == 1
    gam_tilde_raw: dict      # before the min_g -> 1 renormalisation
    separable_frac: float    # fraction of points with y v.r >= 1 (soft-margin diag.)
    quantile: float          # lower quantile used in place of the ess-inf
    orientation: str         # "standard" (gam_min >= gam_maj) or "mirror"


def fit_margin_direction(
    phi_r: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    C: float = 1e6,
    quantile: float = 0.01,
) -> MarginFit:
    """Estimate v and the group r-margins gamma-tilde_g.

    Implements Definition (r-separability):

        v = argmin ||theta||^2   s.t.   P(y theta.r >= 1) = 1,

    as a hard-margin linear SVM without intercept on the r-block. We use a
    large-C soft-margin solver rather than a strict hard-margin QP because on
    real frozen features the r-block is not guaranteed to be separable; `C` is
    taken large enough that the solution is numerically hard-margin whenever
    separability does hold.

    The manuscript defines the group r-margins as essential infima,

        gamma-tilde_g := ess inf_{r | G_g}  y v.r .

    The empirical plug-in for an ess-inf is the sample minimum, but the sample
    minimum is driven by a single point and is badly non-robust on real data.
    We therefore report the `quantile`-level lower quantile as the primary
    estimate (default 1%) and expose the raw minimum for comparison. Both are
    returned so the sensitivity can be stated openly.

    The manuscript's WLOG sets the SMALLER of the two group r-margins to 1. We
    renormalise to enforce that, and record whether the data is in the
    "standard" orientation (gamma-tilde_min >= gamma-tilde_maj, i.e. the
    minority holds the geometric margin advantage) or in the "mirror" case, for
    which the manuscript prescribes substituting (mu_A, gamma-tilde_min) <-
    (mu_B, gamma-tilde_maj).
    """
    svm = LinearSVC(C=C, fit_intercept=False, loss="hinge", max_iter=200_000, tol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm.fit(phi_r, y)
    v = svm.coef_.ravel().astype(np.float64)

    margins = y * (phi_r @ v)
    # Rescale so that the working margin level is exactly 1 (Def. r-separability).
    level = np.quantile(margins, quantile)
    if level <= 0:
        warnings.warn(
            "The r-block is not separable at the requested quantile "
            f"(quantile-{quantile} margin = {level:.4g} <= 0). alpha is only "
            "meaningful under r-separability; treat downstream numbers as "
            "diagnostic, not as a verification of the theory.",
            RuntimeWarning,
        )
        level = max(level, 1e-9)
    v_svm = v / level
    margins_scaled = y * (phi_r @ v_svm)

    raw = {}
    for gg in (0, 1):
        m = margins_scaled[g == gg]
        raw[gg] = float(np.quantile(m, quantile)) if m.size else float("nan")

    base = min(raw[0], raw[1])
    if not np.isfinite(base) or base <= 0:
        gam = {0: float("nan"), 1: float("nan")}
    else:
        gam = {gg: raw[gg] / base for gg in (0, 1)}
        v_svm = v_svm / base

    orientation = "standard" if gam[1] >= gam[0] else "mirror"
    nrm = np.linalg.norm(v_svm)
    return MarginFit(
        v_svm=v_svm,
        v_hat=v_svm / (nrm if nrm > 0 else 1.0),
        gam_tilde=gam,
        gam_tilde_raw={gg: float(np.min(margins_scaled[g == gg])) for gg in (0, 1)},
        separable_frac=float(np.mean(margins_scaled >= 1.0)),
        quantile=quantile,
        orientation=orientation,
    )


# --------------------------------------------------------------------------
# Step 2 -- the alignment operators A and B
# --------------------------------------------------------------------------


def fit_operators(
    phi_r: np.ndarray,
    phi_s: np.ndarray,
    g: np.ndarray,
    ridge_alpha: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate A and B from the conditional law s | r within each group.

    The manuscript's Definition (features-mediated linear spurious correlation)
    states

        s | r  ~  A r + xi   with probability 1 - eps   (group G_maj, g == 0)
        s | r  ~  B r + xi   with probability eps       (group G_min, g == 1)

    so A and B are exactly the within-group linear regressions of the spurious
    block on the causal block. We fit them by ridge regression without an
    intercept (the operators in the manuscript are linear, not affine; the
    features have already been centred by `standardize`).

    Returns A_hat, B_hat with shape (d_s, d_r), matching the manuscript's
    convention A in R^{d_s x d_r}.
    """
    out = []
    for gg in (0, 1):
        m = g == gg
        rr = Ridge(alpha=ridge_alpha, fit_intercept=False)
        rr.fit(phi_r[m], phi_s[m])
        # sklearn stores coef_ as (n_targets, n_features) = (d_s, d_r).
        out.append(np.atleast_2d(rr.coef_).astype(np.float64))
    return out[0], out[1]


# --------------------------------------------------------------------------
# Step 3 -- isotropy diagnostics and the exponent alpha
# --------------------------------------------------------------------------


@dataclass
class IsoDiagnostics:
    mu_A: float
    mu_B: float
    mu: float
    defects: dict            # angle-based deviation from the common-eigenvector condition
    defects_scaled: dict     # off-axis component normalised by ||M||_2 (safe when Mv ~ 0)
    degenerate: dict         # per-product flag: is ||Mv|| too small for the angle to mean anything
    max_defect: float        # max over the NON-degenerate products
    d_star: float            # minimum displacement to an isotropic reparametrisation
    d_star_relative: float   # d_star normalised by K * (||A v|| + ||B v||)
    d_star_sensitivity: dict # d_star_relative as a function of the rank tolerance
    dim_K: int               # dim of the kernel subspace K in the reparametrisation
    attractive: bool         # whether -1 <= mu < min(mu_A, mu_B) holds
    rank_rtol: float


def _kperp_residual(A, B, v, rtol):
    """Relative size of the part of (Av, Bv) that falls outside the subspace K.

    K := ker(Pi_perp . A^T) cap ker(Pi_perp . B^T) is computed as a numerical
    null space. The rank cutoff matters: with ESTIMATED operators the "zero"
    singular values sit at the noise floor, not at machine epsilon, so a cutoff
    of `eps` would classify noise directions as part of the range and shrink K
    to nothing. We therefore cut relative to the largest singular value and
    expose `rtol` so its influence can be reported rather than hidden.
    """
    d_r = v.shape[0]
    d_s = A.shape[0]
    Pi_perp = np.eye(d_r) - np.outer(v, v)
    M_stack = np.vstack([Pi_perp @ A.T, Pi_perp @ B.T])
    _, sv, Vt = np.linalg.svd(M_stack, full_matrices=True)
    cut = rtol * (sv[0] if sv.size else 0.0)
    rank = int((sv > cut).sum())
    K_basis = Vt[rank:].T
    proj_K = K_basis @ K_basis.T if K_basis.size else np.zeros((d_s, d_s))
    Pi_Kperp = np.eye(d_s) - proj_K
    Av, Bv = A @ v, B @ v
    resid = np.linalg.norm(Pi_Kperp @ Av) + np.linalg.norm(Pi_Kperp @ Bv)
    scale = np.linalg.norm(Av) + np.linalg.norm(Bv)
    return resid, scale, int(K_basis.shape[1]) if K_basis.size else 0


def iso_diagnostics(
    A: np.ndarray,
    B: np.ndarray,
    v_hat: np.ndarray,
    phi_r: np.ndarray,
    rank_rtol: float = 1e-2,
    degenerate_tol: float = 1e-2,
) -> IsoDiagnostics:
    """Measure how far the real operators are from the isotropic regime.

    The Isotropic Regime definition asks that v be a common eigenvector of the
    four alignment products A^T A, A^T B, B^T B, B^T A, with

        A^T A v = mu_A v,  A^T B v = mu v,  B^T B v = mu_B v,  B^T A v = mu v.

    On real data this holds only approximately, so for each product M we report
    the *relative eigenvector defect*

        defect(M) = || M v - (v^T M v) v || / || M v ||,

    which is the sine of the angle between M v and v: it is 0 exactly when v is
    an eigenvector of M, and 1 when M v is orthogonal to v. This is the number
    that answers, on real data, the reviewers' question of how restrictive the
    isotropic regime actually is.

    We additionally compute d*, the minimum displacement to an isotropic
    reparametrisation (manuscript, Scope of the Isotropic Regime):

        A = A' + u_A v^T,   B = B' + u_B v^T,
        K  = ker(Pi_perp . A^T) cap ker(Pi_perp . B^T),
        d* = K_sup * ( ||Pi_{K^perp} A v|| + ||Pi_{K^perp} B v|| ),

    where Pi_perp is the projector onto Span{v}^perp inside R^{d_r} and K_sup is
    the radius sup ||r|| of the support. d* == 0 exactly when (A, B) is already
    isotropic, so d* quantifies the size of the perturbation the reparametrisation
    argument has to absorb into the noise.
    """
    v = v_hat / np.linalg.norm(v_hat)
    Av, Bv = A @ v, B @ v
    mu_A = float(Av @ Av)
    mu_B = float(Bv @ Bv)
    mu = float(Av @ Bv)

    defects, defects_scaled, degenerate = {}, {}, {}
    for name, M in (
        ("A^T A", A.T @ A), ("A^T B", A.T @ B),
        ("B^T B", B.T @ B), ("B^T A", B.T @ A),
    ):
        Mv = M @ v
        off = float(np.linalg.norm(Mv - (v @ Mv) * v))   # component orthogonal to v
        nMv = float(np.linalg.norm(Mv))
        nM = float(np.linalg.norm(M, 2))                 # spectral norm of M
        # The angle-based defect is 0/0 when M v is itself negligible, which
        # happens exactly when the corresponding eigenvalue is near zero (e.g.
        # mu ~ 0 for A^T B). In that case the ratio is meaningless and would
        # spuriously report "isotropy badly violated" on data that is exactly
        # isotropic, so we flag it and fall back to the ||M||-normalised
        # off-axis magnitude.
        is_deg = bool(nM > 0 and nMv / nM < degenerate_tol)
        degenerate[name] = is_deg
        defects[name] = float(off / nMv) if nMv > 1e-15 else float("nan")
        defects_scaled[name] = float(off / nM) if nM > 1e-15 else 0.0

    live = [d for k, d in defects.items() if not degenerate[k] and np.isfinite(d)]

    K_sup = float(np.max(np.linalg.norm(phi_r, axis=1)))
    resid, scale, dim_K = _kperp_residual(A, B, v, rank_rtol)
    sens = {}
    for rt in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        r_, s_, _ = _kperp_residual(A, B, v, rt)
        sens[f"{rt:g}"] = float(r_ / s_) if s_ > 1e-15 else 0.0

    return IsoDiagnostics(
        mu_A=mu_A, mu_B=mu_B, mu=mu,
        defects=defects,
        defects_scaled=defects_scaled,
        degenerate=degenerate,
        max_defect=float(max(live)) if live else 0.0,
        d_star=float(K_sup * resid),
        d_star_relative=float(resid / scale) if scale > 1e-15 else 0.0,
        d_star_sensitivity=sens,
        dim_K=dim_K,
        attractive=bool(-1.0 <= mu < min(mu_A, mu_B)),
        rank_rtol=rank_rtol,
    )


def compute_alpha(mf: MarginFit, iso: IsoDiagnostics) -> dict:
    """Combine the margin fit and the eigenvalues into the exponent alpha.

    Manuscript (Isotropic regime theorem):

        alpha := gamma-tilde_min (1 + mu) / (1 + mu_A),

    stated under the WLOG gamma-tilde_maj = 1, gamma-tilde_min >= 1. When the
    data is in the mirror orientation (the *majority* holds the margin
    advantage) the manuscript prescribes the substitution
    (mu_A, gamma-tilde_min) <- (mu_B, gamma-tilde_maj), which is what the
    `mirror` branch below applies.

    The predicted minority decay exponent is max(alpha, 1): the error decays as
    1/(eps z_t) below the transition and as z_t^{-alpha} (ln z_t)^{alpha-1}
    above it.
    """
    if mf.orientation == "standard":
        gam_adv, mu_diag = mf.gam_tilde[1], iso.mu_A
        advantaged = "minority"
    else:
        gam_adv, mu_diag = mf.gam_tilde[0], iso.mu_B
        advantaged = "majority"

    denom = 1.0 + mu_diag
    alpha = float(gam_adv * (1.0 + iso.mu) / denom) if denom != 0 else float("nan")
    return {
        "alpha": alpha,
        "orientation": mf.orientation,
        "advantaged_group": advantaged,
        "gamma_tilde_advantaged": float(gam_adv),
        "mu_diagonal_used": float(mu_diag),
        "mu": float(iso.mu),
        "regime": "alpha < 1 (balancing helps)" if alpha < 1
                  else "alpha >= 1 (geometry dominates)",
        "predicted_minority_exponent": float(max(alpha, 1.0)),
        "critical_gamma": float(denom / (1.0 + iso.mu))
                          if (1.0 + iso.mu) != 0 else float("nan"),
    }


# --------------------------------------------------------------------------
# Step 4 -- the within-cell coupling test
# --------------------------------------------------------------------------


def _cv_r2(X: np.ndarray, Y: np.ndarray, ridge_alpha: float, n_splits: int, seed: int) -> float:
    """Cross-validated, variance-weighted multi-output R^2 of the map X -> Y.

    Held-out rather than in-sample R^2 is essential here: within a single
    (y, g) cell the sample size can be comparable to the number of r-features,
    and in-sample R^2 would be inflated towards 1 by pure overfitting, which is
    exactly the artefact a sceptical reviewer would suspect.
    """
    n = X.shape[0]
    k = min(n_splits, n)
    if k < 2:
        return float("nan")
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    num = np.zeros(Y.shape[1])
    den = np.zeros(Y.shape[1])
    for tr, te in kf.split(X):
        rr = Ridge(alpha=ridge_alpha, fit_intercept=True)
        rr.fit(X[tr], Y[tr])
        pred = rr.predict(X[te])
        num += ((Y[te] - pred) ** 2).sum(axis=0)
        den += ((Y[te] - Y[tr].mean(axis=0)) ** 2).sum(axis=0)
    den_tot = den.sum()
    return float(1.0 - num.sum() / den_tot) if den_tot > 1e-15 else float("nan")


def within_cell_coupling(
    phi_r: np.ndarray,
    phi_s: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    n_perm: int = 200,
    ridge_alpha: float = 1.0,
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """Test for residual Phi_r -> Phi_s coupling *within* each (y, g) cell.

    This is the decisive measurement. Marginal (pooled) correlation between
    Phi_r and Phi_s is large on any spurious-correlation benchmark simply
    because both blocks track the label y, and that is precisely the
    label-mediated situation, in which s is conditionally independent of r
    given (y, g). Feature-mediation is the claim that coupling SURVIVES
    conditioning on (y, g).

    Conditioning on the cell is exactly conditioning on (y, g): within a cell,
    y and g are constant, so any remaining predictive power of Phi_r for Phi_s
    is residual coupling and cannot be explained by label mediation.

    The null distribution is obtained by permuting the ROWS of Phi_s as a block
    within the cell. Block permutation destroys the pairing between r and s
    while preserving the internal covariance structure of the s-features, so
    the null accounts for dimensionality and for correlations among s-features.

    Returns per-cell held-out R^2, the null mean, an empirical p-value, and the
    pooled (marginal) R^2 for contrast.
    """
    rng = np.random.default_rng(seed)
    cells = []
    for yy in (-1, 1):
        for gg in (0, 1):
            m = (y == yy) & (g == gg)
            n = int(m.sum())
            if n < 20:
                cells.append({
                    "y": yy, "g": gg, "n": n, "r2": float("nan"),
                    "null_mean": float("nan"), "p": float("nan"),
                    "note": "cell too small (n < 20)",
                })
                continue
            X, Y = phi_r[m], phi_s[m]
            r2 = _cv_r2(X, Y, ridge_alpha, n_splits, seed)
            null = np.empty(n_perm)
            for b in range(n_perm):
                null[b] = _cv_r2(X, Y[rng.permutation(n)], ridge_alpha, n_splits, seed)
            null = null[np.isfinite(null)]
            p = float((1.0 + (null >= r2).sum()) / (1.0 + null.size)) if null.size else float("nan")
            cells.append({
                "y": yy, "g": gg, "n": n, "r2": r2,
                "null_mean": float(null.mean()) if null.size else float("nan"),
                "null_q95": float(np.quantile(null, 0.95)) if null.size else float("nan"),
                "p": p, "note": "",
            })

    pooled = _cv_r2(phi_r, phi_s, ridge_alpha, n_splits, seed)
    finite = [c["r2"] for c in cells if np.isfinite(c["r2"])]
    return {
        "cells": cells,
        "pooled_r2": pooled,
        "mean_within_cell_r2": float(np.mean(finite)) if finite else float("nan"),
        "n_perm": n_perm,
    }


# --------------------------------------------------------------------------
# Step 5 -- last-layer full-batch gradient descent
# --------------------------------------------------------------------------


def logistic_gd(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    h: float = 0.01,
    T: int = 100_000,
    n_ckpt: int = 60,
    w0: np.ndarray | None = None,
) -> dict:
    """Full-batch gradient descent on the logistic risk, tracking per-group error.

    The manuscript studies population gradient descent on the expected risk

        L(w) = E[ log(1 + exp(-y w.Phi(x))) ],
        w^{t+1} = w^t - h_t grad L(w^t),
        z_t := sum_{k<t} h_k   (cumulative learning steps).

    With a constant step size h this gives z_t = h t. The empirical mean over a
    sample whose minority fraction equals eps is the natural finite-sample
    stand-in for the population expectation, since the population gradient is
    (1 - eps) E_maj[.] + eps E_min[.] and the sample mean reproduces exactly
    that weighting.

    Tracked quantity, per group: the soft error E_{G_g}[1 - p_y] with
    p_y = sigmoid(y w.Phi(x)), which is what the manuscript's rates describe.
    Checkpoints are log-spaced because the predicted decay is polynomial in z_t.
    """
    N, d = X.shape
    Xy = (y[:, None] * X).astype(np.float64)
    w = np.zeros(d) if w0 is None else w0.astype(np.float64).copy()

    ckpts = np.unique(np.round(np.logspace(0, np.log10(T), n_ckpt)).astype(int))
    ckpts = ckpts[(ckpts >= 1) & (ckpts <= T)]
    ck = set(int(c) for c in ckpts)

    m_maj, m_min = (g == 0), (g == 1)
    rec = {"t": [], "z": [], "err_maj": [], "err_min": [], "train_loss": []}

    for t in range(1, T + 1):
        u = Xy @ w
        # 1 - p_y = sigmoid(-u), computed stably.
        one_minus_p = np.where(u >= 0, np.exp(-u) / (1.0 + np.exp(-u)),
                               1.0 / (1.0 + np.exp(u)))
        w += (h / N) * (Xy.T @ one_minus_p)
        if t in ck:
            rec["t"].append(t)
            rec["z"].append(h * t)
            rec["err_maj"].append(float(one_minus_p[m_maj].mean()) if m_maj.any() else np.nan)
            rec["err_min"].append(float(one_minus_p[m_min].mean()) if m_min.any() else np.nan)
            rec["train_loss"].append(float(np.mean(np.logaddexp(0.0, -u))))

    return {k: np.asarray(v) for k, v in rec.items()}


def subsample_to_eps(
    y: np.ndarray, g: np.ndarray, eps: float, rng: np.random.Generator
) -> np.ndarray:
    """Return indices realising a target minority fraction eps.

    eps is the manuscript's group proportion, eps = P(G_min). We hold the
    majority group fixed and subsample the minority group, which is the
    standard way this sweep is done in the group-robustness literature and
    keeps the majority statistics identical across the sweep so that the
    majority curve is a control.
    """
    idx_maj = np.flatnonzero(g == 0)
    idx_min = np.flatnonzero(g == 1)
    n_maj = idx_maj.size
    # eps = n_min / (n_maj + n_min)  =>  n_min = eps n_maj / (1 - eps)
    n_min = int(round(eps * n_maj / max(1e-12, 1.0 - eps)))
    n_min = min(n_min, idx_min.size)
    if n_min < 1:
        raise ValueError(f"eps={eps} needs at least 1 minority sample; got {n_min}.")
    keep_min = rng.choice(idx_min, size=n_min, replace=False)
    return np.concatenate([idx_maj, keep_min])


def local_slope(z: np.ndarray, err: np.ndarray, lo_frac: float = 0.5) -> float:
    """Empirical decay exponent beta from a log-log fit over the late window.

    The manuscript's predictions are asymptotic in z_t, so the exponent is
    estimated on the last `1 - lo_frac` fraction of the (log-spaced)
    checkpoints. Returns beta in  err ~ z^{-beta}.
    """
    m = np.isfinite(z) & np.isfinite(err) & (err > 0) & (z > 0)
    zz, ee = z[m], err[m]
    if zz.size < 4:
        return float("nan")
    k = int(len(zz) * lo_frac)
    zz, ee = zz[k:], ee[k:]
    if zz.size < 3:
        return float("nan")
    A_ = np.vstack([np.log(zz), np.ones_like(zz)]).T
    sol, *_ = np.linalg.lstsq(A_, np.log(ee), rcond=None)
    return float(-sol[0])
