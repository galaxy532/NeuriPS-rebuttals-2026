"""
common.py -- Core estimators for the rebuttal experiments.

This is a library, not a command-line script: it is imported by analyze.py,
eps_sweep.py and the validation harness in ../Estimator_Validation/. To use it
directly from the repository root:

    python -c "
    from common import FeatureBundle, fit_margin_direction, standardize
    b = FeatureBundle.load('features_waterbirds_train.npz')
    b.phi = standardize(b.phi)
    mf = fit_margin_direction(b.phi, b.y, b.g)
    print(mf.gam_tilde, mf.separable_frac)
    "

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
    place: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def phi_r(self) -> np.ndarray:
        return self.phi[:, self.idx_r]

    @property
    def phi_s(self) -> np.ndarray:
        return self.phi[:, self.idx_s]

    def save(self, path: str) -> None:
        extra = {} if self.place is None else {"place": self.place}
        np.savez_compressed(
            path, phi=self.phi, y=self.y, g=self.g,
            idx_r=self.idx_r, idx_s=self.idx_s,
            meta=np.array(repr(self.meta), dtype=object), **extra,
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
            place=z["place"].astype(int) if "place" in z else None,
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
    quantile: float          # lower quantile used in place of the ess-inf
    orientation: str         # "standard", "mirror", or "undefined" (see below)

    # -- separability, reported as two fields that cannot be confused ---------
    #
    # An earlier version reported a single `separable_frac`, and it silently
    # meant two different things. When the r-block IS separable at the working
    # quantile it equalled `mean(margins_scaled >= 1)`, which is identically
    # 1 - quantile (0.99 at the default) because the scaling is DEFINED by that
    # quantile -- a constant dressed as a measurement. When the block is NOT
    # separable, the 1e-9 clamp below turns the same expression into the plain
    # fraction of correctly classified points. One name, two quantities, and no
    # way for a reader to tell which one they were looking at.
    #
    # The two are now separate and each means one thing always:
    separable_at_quantile: bool   # is the `quantile`-level margin strictly > 0?
    frac_correct: float           # mean(y v.r > 0); a plain training accuracy,
                                  # meaningful whether or not the block separates
    separable_frac: float         # kept for continuity; equals 1 - quantile when
                                  # separable_at_quantile is True. Prefer the two
                                  # fields above.


def fit_margin_direction(
    phi_r: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    C: float = 1e6,
    quantile: float = 0.01,
    seed: int = 0,
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

    A note on the solver tolerances. An earlier version used tol=1e-9 with
    max_iter=200_000. At C=1e6 liblinear's dual coordinate descent cannot drive
    the dual violation anywhere near 1e-9, so `n_iter_` hit `max_iter` on every
    input tested: the solver always burned all 200,000 passes over the data and
    then returned a non-converged `v` regardless. That is a fixed cost of
    minutes at d_r in the hundreds and over ten minutes at d_r ~ 2000, bought
    nothing, and -- because it never converged -- gave no accuracy guarantee to
    trade away. tol=1e-4 with max_iter=20_000 is an attainable stopping
    criterion rather than an unreachable one. If liblinear still warns about
    convergence, that is informative: it means the r-block is far from
    separable at this C, which is exactly the situation the RuntimeWarning
    below is meant to surface.

    Determinism (`seed`)
    --------------------
    `LinearSVC` defaults to `random_state=None`, and with `loss="hinge"` the
    solver is dual coordinate descent, which SHUFFLES the order in which
    coordinates are updated using that generator. Leaving it unset therefore
    made this function non-deterministic: repeated fits on byte-identical input
    returned different directions, and because `level` is read off a quantile of
    the resulting margins, the group r-margins gamma-tilde_g -- and hence alpha,
    which is linear in them -- inherited that variation directly. Measured on
    one bundle, the 1% quantile margin ranged over [0.367, 0.925] across four
    identical calls; with `random_state` fixed the four agreed to every digit.

    On the Waterbirds raw/two-concept split this moved alpha from 0.811 to 1.144
    between two runs of unchanged code -- across the alpha = 1 boundary that
    separates the manuscript's two regimes. Seeding makes the number
    REPRODUCIBLE; it does not make it stable. The underlying cause is that at
    C = 1e6 the problem is close to degenerate and the solve does not converge,
    so different coordinate orders stop in different places. The real remedy is
    to solve the hard-margin program directly, which has a unique solution
    whenever the r-block separates -- and when it does not separate, `v` does
    not exist and the honest output is the non-estimable branch below.
    """
    svm = LinearSVC(C=C, fit_intercept=False, loss="hinge", max_iter=20_000,
                    tol=1e-4, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm.fit(phi_r, y)
    v = svm.coef_.ravel().astype(np.float64)

    margins = y * (phi_r @ v)
    # Computed BEFORE any rescaling, so it is a property of the fitted direction
    # rather than of the normalisation applied to it.
    frac_correct = float(np.mean(margins > 0))

    # Rescale so that the working margin level is exactly 1 (Def. r-separability).
    level = np.quantile(margins, quantile)
    separable_at_quantile = bool(level > 0)
    if not separable_at_quantile:
        warnings.warn(
            "The r-block is not separable at the requested quantile "
            f"(quantile-{quantile} margin = {level:.4g} <= 0; only "
            f"{frac_correct:.2%} of points are correctly classified). alpha is "
            "only meaningful under r-separability; treat downstream numbers as "
            "diagnostic, not as a verification of the theory.",
            RuntimeWarning,
        )
        # The clamp keeps the arithmetic finite, but note what it does: dividing
        # by 1e-9 inflates every margin by 10^9, which is why `gam_tilde_raw`
        # reads in the billions on non-separable data. Those magnitudes are an
        # artefact of this clamp and carry no information.
        level = max(level, 1e-9)
    v_svm = v / level
    margins_scaled = y * (phi_r @ v_svm)

    raw = {}
    for gg in (0, 1):
        m = margins_scaled[g == gg]
        raw[gg] = float(np.quantile(m, quantile)) if m.size else float("nan")

    base = min(raw[0], raw[1])
    if not np.isfinite(base) or base <= 0:
        # No positive common scale exists, so the WLOG min_g gamma-tilde_g = 1
        # cannot be imposed and the group r-margins are undefined.
        gam = {0: float("nan"), 1: float("nan")}
        # Orientation must be reported as undefined rather than computed. With
        # gam = nan the comparison `gam[1] >= gam[0]` is False (every ordering
        # test against nan is), so the old code fell through to "mirror" and
        # printed a definite orientation that was purely a nan artefact -- and
        # "mirror" then selected mu_B over mu_A downstream.
        orientation = "undefined"
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
        quantile=quantile,
        orientation=orientation,
        separable_at_quantile=separable_at_quantile,
        frac_correct=frac_correct,
        separable_frac=float(np.mean(margins_scaled >= 1.0)),
    )


def refit_margins_at_quantile(
    mf: MarginFit,
    phi_r: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    quantile: float,
) -> MarginFit:
    """Recompute the group r-margins at a different ess-inf quantile.

    WHY THIS EXISTS
    ---------------
    The manuscript defines gamma-tilde_g as an essential infimum, i.e. it asks
    for  P(y v.r >= 1) = 1  exactly. The separability probe shows the best
    achievable held-out accuracy on this representation is about 0.94, so a
    proxy quantile of 1% -- which demands 99% of points on the correct side --
    can never be met in the population, and alpha is then refused for a reason
    that has nothing to do with the geometry it is meant to describe.

    Sweeping the quantile turns that binary failure into a reported sensitivity:
    at which q does the r-block become separable, and how does alpha move with
    q once it is? A q chosen above the population error rate gives a q-RELAXED
    r-separability under which alpha is estimable, with the approximation
    stated rather than hidden.

    WHY NO REFIT IS NEEDED
    ----------------------
    `fit_margin_direction` divides v twice, first by `level` and then by `base`,
    and both are positive scalars whenever the block separates. A positive
    rescaling does not move a direction, so `v_hat` is INDEPENDENT of the
    quantile. Moreover gamma-tilde is a RATIO of two per-group quantiles of the
    same margin vector, so the `level` division cancels out of it entirely.
    Everything that changes with q can therefore be recomputed from the margins
    of the existing direction, and the SVM -- the expensive part, minutes at
    d_r in the thousands -- is fitted once for the whole sweep.

    By the same argument `iso_diagnostics` is q-independent: it consumes only
    `v_hat`. So mu_A', mu', dim K and d* are shared across the sweep, and only
    gamma-tilde moves.

    Returns a MarginFit carrying the same direction with gamma-tilde,
    separability and orientation recomputed at `quantile`.
    """
    v = mf.v_hat
    margins = y * (phi_r @ v)
    frac_correct = float(np.mean(margins > 0))

    level = float(np.quantile(margins, quantile))
    separable = bool(level > 0)

    raw = {}
    for gg in (0, 1):
        m = margins[g == gg]
        raw[gg] = float(np.quantile(m, quantile)) if m.size else float("nan")

    base = min(raw[0], raw[1])
    if not np.isfinite(base) or base <= 0:
        gam = {0: float("nan"), 1: float("nan")}
        orientation = "undefined"
    else:
        # The `level` scaling cancels from the ratio, so dividing the raw
        # per-group quantiles by their minimum reproduces exactly the
        # gamma-tilde that fit_margin_direction would return at this quantile.
        gam = {gg: raw[gg] / base for gg in (0, 1)}
        orientation = "standard" if gam[1] >= gam[0] else "mirror"

    return MarginFit(
        v_svm=mf.v_svm,
        v_hat=v,
        gam_tilde=gam,
        gam_tilde_raw={gg: float(np.min(margins[g == gg])) for gg in (0, 1)},
        quantile=quantile,
        orientation=orientation,
        separable_at_quantile=separable,
        frac_correct=frac_correct,
        separable_frac=float(np.mean(margins >= level)) if separable else frac_correct,
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
    # -- RAW fitted operators. Diagnostic only. --------------------------------
    # These are ||A_hat v||^2 etc. for the ridge-fitted A_hat, B_hat. Section C
    # never claims the FITTED pair is isotropic -- its whole point is that a
    # generic (A, B) is not, and that one recovers isotropy by reparametrising.
    # So these are the right numbers for asking "how far from isotropic is the
    # raw fit?" and the WRONG ones to put into alpha.
    mu_A: float
    mu_B: float
    mu: float

    # -- PRIMED operators: mu_A' = ||A'v||^2, mu_B' = ||B'v||^2, mu' = A'v.B'v -
    # Section C, immediately after Eq (40). These are Theorem D.4's inputs and
    # the ones alpha (Eq 44) must be computed from.
    mu_A_prime: float
    mu_B_prime: float
    mu_prime: float
    u_A_norm: float          # ||u_A|| = ||Av - A'v||, the absorbed displacement
    u_B_norm: float          # ||u_B|| = ||Bv - B'v||
    reparam_exists: bool     # dim K > 0: is there any isotropic (A', B') at all?
    attractive_satisfiable: bool   # dim K >= 2, Section C's condition for
                                   # breaking positive collinearity within K
    attractive_prime: bool   # -1 <= mu' < min(mu_A', mu_B') at the d*-minimal
                             # choice specifically (see note in iso_diagnostics)
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
    d_star_informative: bool # False when dim K == 0 pins d_star_relative at 1;
                             # see the note in iso_diagnostics' docstring


def _kperp_residual(A, B, v, rtol):
    """The isotropic reparametrisation of (A, B): K, the projections, and d*.

    Implements manuscript Section C, "Scope of the Isotropic Regime", Eqs
    (37)-(40). Writing

        A = A' + u_A v^T,   B = B' + u_B v^T,                            (37)

    the pair (A', B') is isotropic exactly when

        A'v, B'v  in  K := ker(Pi_bar_v . A^T) cap ker(Pi_bar_v . B^T),  (38)

    with Pi_bar_v the projector onto Span{v}^perp of Eq (22). Note K is defined
    from the UNPRIMED operators, and that is not a slip in the manuscript: since
    A'^T = A^T - v u_A^T, the extra term is parallel to v and is annihilated by
    Pi_bar_v, so ker(Pi_bar_v . A'^T) = ker(Pi_bar_v . A^T). The kernel does not
    move when we reparametrise.

    Minimising the displacement (39) over A'v, B'v in K is then an orthogonal
    projection onto K, giving

        A'v = Pi_K (A v),   B'v = Pi_K (B v),
        d*  = K_sup ( ||Pi_{K^perp} (A v)|| + ||Pi_{K^perp} (B v)|| ).    (40)

    We return the projections themselves and not merely the residual norm: they
    are what Section C's eigenvalues mu_A' = ||A'v||^2, mu_B' = ||B'v||^2 and
    mu' = A'v . B'v are built from, and those primed eigenvalues -- NOT the raw
    ||A_hat v||^2 -- are what Theorem D.4's alpha (Eq 44) takes as input.

    The rank cutoff matters: with ESTIMATED operators the "zero" singular values
    sit at the noise floor, not at machine epsilon, so a cutoff of `eps` would
    classify noise directions as part of the range and shrink K to nothing. We
    cut relative to the largest singular value and expose `rtol` so its
    influence can be reported rather than hidden. This tolerance is load
    bearing: it alone decides dim K, and hence whether a reparametrisation is
    judged to exist at all.

    Returns
    -------
    resid, scale : float
        Numerator and normaliser of d_star_relative.
    dim_K : int
    Avp, Bvp : (d_s,) arrays
        A'v and B'v, the projections onto K. Zero vectors when dim K == 0.
    Av, Bv : (d_s,) arrays
        The unprojected A v and B v, for reporting u_A = Av - A'v.
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

    Av, Bv = A @ v, B @ v
    Avp, Bvp = proj_K @ Av, proj_K @ Bv          # A'v and B'v of Eq (38)
    resid = np.linalg.norm(Av - Avp) + np.linalg.norm(Bv - Bvp)
    scale = np.linalg.norm(Av) + np.linalg.norm(Bv)
    dim_K = int(K_basis.shape[1]) if K_basis.size else 0
    return resid, scale, dim_K, Avp, Bvp, Av, Bv


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

    WHEN d* IS VACUOUS, AND WHY IT MUST BE FLAGGED
    ----------------------------------------------
    d* is only a measurement when the subspace K is non-trivial, and on
    ESTIMATED operators it very often cannot be. `_kperp_residual` builds

        M_stack = [Pi_perp A^T ; Pi_perp B^T]   of shape (2 d_r, d_s),

    and K is its null space, so

        dim K = d_s - rank(M_stack),    rank(M_stack) <= min(2 d_r, d_s).

    If d_s <= 2 d_r, the rank bound is d_s, and ridge-estimated A and B are
    full rank for any positive ridge penalty -- the estimate has no exact
    linear dependencies even when the population operators do. The rank then
    saturates at d_s, giving dim K = 0, K_basis empty, Pi_{K^perp} = I, and
    hence

        resid = ||A v|| + ||B v|| = scale   =>   d_star_relative == 1 EXACTLY.

    So on any run with d_s <= 2 d_r, d_star_relative is 1.0000 as a matter of
    arithmetic, whatever the operators look like. It is then reporting the
    `d_s >= 2 d_r` flag in disguise, not a distance to an isotropic
    reparametrisation, and the `rank_rtol` sweep cannot rescue it because the
    deficiency is in the SHAPE of M_stack rather than in where its singular
    values are cut. `d_star_informative` records this so a reader is never
    invited to interpret the pinned value.
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
    resid, scale, dim_K, Avp, Bvp, Av_, Bv_ = _kperp_residual(A, B, v, rank_rtol)
    sens = {}
    for rt in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        r_, s_, *_ = _kperp_residual(A, B, v, rt)
        sens[f"{rt:g}"] = float(r_ / s_) if s_ > 1e-15 else 0.0

    # Section C's eigenvalues, from the projected (isotropic) operators.
    #
    # When dim K == 0 the projector is the zero map, so A'v = B'v = 0 and all
    # three primed eigenvalues are 0. That is not a measurement of an isotropic
    # pair with small eigenvalues -- it is the statement that NO isotropic
    # reparametrisation exists, so Theorem D.4 has no hypothesis to stand on.
    # `reparam_exists` records the distinction, and compute_alpha refuses to
    # produce a number in that case rather than returning the degenerate
    # alpha = gamma-tilde_min that 0/0-style eigenvalues would give.
    mu_A_p = float(Avp @ Avp)
    mu_B_p = float(Bvp @ Bvp)
    mu_p = float(Avp @ Bvp)

    # Caveat on `attractive_prime`. The orthogonal projection is the d*-MINIMAL
    # choice of (A', B'), not the only admissible one: Section C notes that with
    # dim K >= 2 one may perturb A'v, B'v within K to break positive
    # collinearity and satisfy the attractive condition, at the cost of a
    # slightly larger d*. So attractive_prime == False does NOT mean the
    # attractive regime is unreachable; it means the cheapest reparametrisation
    # does not reach it. `attractive_satisfiable` (dim K >= 2) is the condition
    # for a perturbation to exist at all.
    attractive_p = bool(-1.0 <= mu_p < min(mu_A_p, mu_B_p)) if dim_K > 0 else False

    return IsoDiagnostics(
        mu_A=mu_A, mu_B=mu_B, mu=mu,
        mu_A_prime=mu_A_p, mu_B_prime=mu_B_p, mu_prime=mu_p,
        u_A_norm=float(np.linalg.norm(Av_ - Avp)),
        u_B_norm=float(np.linalg.norm(Bv_ - Bvp)),
        reparam_exists=bool(dim_K > 0),
        attractive_satisfiable=bool(dim_K >= 2),
        attractive_prime=attractive_p,
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
        d_star_informative=bool(dim_K > 0),
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
    # Theorem D.4 (Eq 44) reads alpha := gamma-tilde_min (1 + mu) / (1 + mu_A)
    # where, per Section C, the eigenvalues are those of the REPARAMETRISED
    # pair: mu_A' = ||A'v||^2, mu_B' = ||B'v||^2, mu' = A'v . B'v. An earlier
    # version fed the raw fitted ||A_hat v||^2 into this formula instead. That
    # is a different quantity: Section C exists precisely because a generic
    # fitted (A, B) is NOT isotropic, so the raw eigenvalues are not the ones
    # the theorem's hypothesis supplies. The unprimed values are still returned
    # below, as a diagnostic of how far the fit sits from isotropy, but alpha is
    # computed from the primed ones.
    if mf.orientation == "standard":
        gam_adv, mu_diag = mf.gam_tilde[1], iso.mu_A_prime
        mu_diag_raw = iso.mu_A
        advantaged = "minority"
    elif mf.orientation == "mirror":
        gam_adv, mu_diag = mf.gam_tilde[0], iso.mu_B_prime
        mu_diag_raw = iso.mu_B
        advantaged = "majority"
    else:  # "undefined": the margin fit failed, so there is no advantaged group
        gam_adv, mu_diag, mu_diag_raw = float("nan"), float("nan"), float("nan")
        advantaged = "undefined"

    denom = 1.0 + mu_diag
    if not iso.reparam_exists:
        # dim K == 0: no isotropic (A', B') exists, so there is nothing for
        # Eq (44) to be evaluated at. Computing it anyway would silently return
        # gamma-tilde_min, since all three primed eigenvalues are then 0.
        alpha = float("nan")
    else:
        alpha = float(gam_adv * (1.0 + iso.mu_prime) / denom) if denom != 0 \
            else float("nan")

    # Reported alongside, for comparison only: what the old code would have
    # produced from the raw fitted operators.
    denom_raw = 1.0 + mu_diag_raw
    alpha_raw_operators = float(gam_adv * (1.0 + iso.mu) / denom_raw) \
        if denom_raw != 0 else float("nan")

    # A non-finite alpha must be reported as a failure, never as a regime.
    #
    # The comparison `alpha < 1` evaluates to False when alpha is nan, because
    # every ordering comparison against nan is False. A bare
    #
    #     "alpha < 1 (...)" if alpha < 1 else "alpha >= 1 (...)"
    #
    # therefore routed EVERY failed estimate into the "alpha >= 1 (geometry
    # dominates)" branch -- turning a missing number into a definite claim about
    # the manuscript's central dichotomy, printed in bold next to a nan. The
    # finiteness of alpha is checked first so that the two-way regime split is
    # only ever reached by an alpha that actually exists.
    estimable = bool(np.isfinite(alpha))
    if estimable:
        regime = ("alpha < 1 (balancing helps)" if alpha < 1
                  else "alpha >= 1 (geometry dominates)")
        failure_reason = ""
        predicted = float(max(alpha, 1.0))
    else:
        regime = "UNDEFINED -- alpha was not estimable; see failure_reason"
        predicted = float("nan")
        if not iso.reparam_exists:
            failure_reason = (
                "dim K = 0, so no isotropic reparametrisation (A', B') exists "
                "(Section C, Eq 38): the intersection of the two kernels is "
                "trivial, there is no admissible A'v, and Theorem D.4's "
                "hypothesis is unavailable. Section C guarantees dim K > 0 only "
                "when d_s >= 2 d_r - 1, which this split does not satisfy; note "
                "that is a SUFFICIENT condition, so dim K can be positive "
                "without it when the fitted operators are rank deficient"
            )
        elif not mf.separable_at_quantile:
            failure_reason = (
                "the r-block is not separable at the "
                f"{mf.quantile:.0%} quantile -- only {mf.frac_correct:.2%} of "
                "points are correctly classified by v, so the group r-margins "
                "gamma-tilde_g have no positive common scale and the WLOG "
                "min_g gamma-tilde_g = 1 cannot be imposed. alpha is a "
                "statement about a separable r-block and there is nothing here "
                "for it to describe"
            )
        elif not np.isfinite(iso.mu) or not np.isfinite(mu_diag):
            failure_reason = ("the alignment eigenvalues are not finite, so the "
                              "isotropic-regime quantities feeding alpha do not exist")
        else:
            failure_reason = ("1 + mu_diagonal == 0, so alpha's denominator "
                              "vanishes")

    return {
        "alpha": alpha,
        "estimable": estimable,
        "failure_reason": failure_reason,
        "orientation": mf.orientation,
        "advantaged_group": advantaged,
        "gamma_tilde_advantaged": float(gam_adv),
        # Section C / Theorem D.4 inputs -- what alpha is actually built from.
        "mu_diagonal_used": float(mu_diag),
        "mu_prime": float(iso.mu_prime),
        "reparam_exists": bool(iso.reparam_exists),
        "attractive_satisfiable": bool(iso.attractive_satisfiable),
        # Raw fitted operators, for comparison only. `alpha_raw_operators` is
        # what the previous version of this function returned; it is kept so the
        # size of the correction is visible rather than silently absorbed.
        "mu_diagonal_raw": float(mu_diag_raw),
        "mu": float(iso.mu),
        "alpha_raw_operators": alpha_raw_operators,
        "regime": regime,
        "predicted_minority_exponent": predicted,
        "critical_gamma": float(denom / (1.0 + iso.mu_prime))
                          if (1.0 + iso.mu_prime) != 0 else float("nan"),
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
    min_cell: int = 200,
    reliable_cell: int = 400,
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

    Cell size is the binding constraint, and it is why this test should be run
    on the TEST split. On the Waterbirds *train* split the four cells are
    3498 / 184 / 56 / 1057: fitting a multi-output ridge from ~20 r-features
    inside a 56-sample cell yields an estimate whose error bar swamps its value.
    Cross-validation prevents that number from being spuriously HIGH, but it
    cannot create information -- the result simply becomes unstable across
    seeds while still printing to three decimals next to a p-value. The test
    split is constructed roughly balanced (~2255/2255/642/642), so the smallest
    cell is ~642 rather than 56.

    Cells below `min_cell` are therefore refused outright rather than reported,
    and cells between `min_cell` and `reliable_cell` are reported with an
    explicit warning attached.

    Returns per-cell held-out R^2, the null mean, an empirical p-value, and the
    pooled (marginal) R^2 for contrast.
    """
    rng = np.random.default_rng(seed)
    cells = []
    for yy in (-1, 1):
        for gg in (0, 1):
            m = (y == yy) & (g == gg)
            n = int(m.sum())
            if n < min_cell:
                cells.append({
                    "y": yy, "g": gg, "n": n, "r2": float("nan"),
                    "null_mean": float("nan"), "p": float("nan"),
                    "note": f"REFUSED: n={n} < min_cell={min_cell}; too small "
                            "for a meaningful fit (use the test split)",
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
                "p": p,
                "note": "" if n >= reliable_cell else
                        f"WARNING: n={n} < {reliable_cell}; estimate is unstable",
            })

    pooled = _cv_r2(phi_r, phi_s, ridge_alpha, n_splits, seed)
    finite = [c["r2"] for c in cells if np.isfinite(c["r2"])]
    sizes = [c["n"] for c in cells]
    return {
        "cells": cells,
        "pooled_r2": pooled,
        "mean_within_cell_r2": float(np.mean(finite)) if finite else float("nan"),
        "n_perm": n_perm,
        "min_cell_size": int(min(sizes)) if sizes else 0,
        "n_cells_refused": sum(1 for c in cells if not np.isfinite(c["r2"])),
        "n_cells_unreliable": sum(1 for c in cells
                                  if np.isfinite(c["r2"]) and c["n"] < reliable_cell),
        "p_value_floor": 1.0 / (1.0 + n_perm),
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
