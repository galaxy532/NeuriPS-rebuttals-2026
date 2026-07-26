"""
identify_rs.py -- Splitting a frozen representation into Phi_r and Phi_s.

BASIS AND RULE ARE SEPARATE
===========================
Identification involves two decisions that are easily confused.

  (1) THE BASIS -- which columns are the "features"?
        raw : the coordinates of Phi itself. Cheap, nothing to fit.
        SAE : the activations of a sparse autoencoder fitted to Phi
              (`fit_sae`). Slower, needs torch, but the columns are more
              monosemantic. Use when raw coordinates are polysemantic --
              i.e. when the raw run classifies most features as "weak".

  (2) THE RULE -- how is a column classified as r-type or s-type?
        two-concept : `two_concept_identify(cols, y, place)`
        sign-flip   : `sign_flip_identify(cols, concept, g)`

Neither rule requires an SAE: `two_concept_identify` works perfectly well on
raw coordinates, and `fit_sae` commits you to nothing. `analyze.py` therefore
exposes the basis as its only switch (`--sae`) and always runs BOTH rules on
whichever basis was chosen, reporting their agreement.

Whichever basis is chosen, the columns handed to a rule are written c_k below.
For the raw basis c_k is simply the k-th coordinate of Phi.

RULE 1 -- two-concept (preferred where the attribute is annotated)
==================================================================
Waterbirds annotates the spurious attribute directly: `metadata.csv` carries
`place` (land/water) next to `y` (landbird/waterbird). Each feature is
decomposed over the four (y, place) cells as a 2x2 factorial and classified by
which factor it responds to. This uses the annotation rather than inferring
around it. Full detail in `two_concept_identify`.

Requires: y and place. Not usable when the spurious attribute is unlabelled.

RULE 2 -- sign-flip (the Colored-MNIST rule; cross-check, or when `place` is absent)
===================================================================================
Let `concept` be the causal concept (digit class in Colored-MNIST; the bird
label y in Waterbirds). Define

    rho_0(k) = corr(c_k, concept | g = 0),
    rho_1(k) = corr(c_k, concept | g = 1).

A causal (r-type) feature tracks the concept itself, so its correlation has the
SAME sign in both groups. A spurious (s-type) feature tracks the group-dependent
attribute, whose relation to the concept REVERSES between groups, so its
correlation flips sign. Hence

    f_k is r-type  if rho_0 rho_1 > 0 and min(|rho_0|, |rho_1|) >= tau,
    f_k is s-type  if rho_0 rho_1 < 0 and min(|rho_0|, |rho_1|) >= tau,
    f_k is weak    otherwise.

This is the operational content of A != B in the manuscript's Definition
(features-mediated linear spurious correlation).

Requires: only y and the group index g -- so it still applies when the spurious
attribute is not annotated. `agreement()` compares the two rules' splits.

Why sign-flip still works on a binary concept
---------------------------------------------
A natural objection: Colored-MNIST had a quantitative concept (digit class)
with an explicitly inverted intensity map, whereas Waterbirds has a binary
label and nothing is "inverted" during image generation. Two answers:

  * With binary y, corr(c_k, y | g) is the point-biserial correlation, i.e. the
    standardised difference of class means within group g. Correlation needs
    variation, not ordinality, so the statistic is well defined.
  * The inversion IS present, but it comes from the group DEFINITION rather
    than from image generation. Waterbirds groups are g := 1[place != y], so
    conditioning on g fixes the background-label relation: place = y within
    g = 0 and place = -y within g = 1. A pure background feature therefore
    correlates positively with y in one group and negatively in the other. In
    Colored-MNIST the flip was engineered through the intensity map; here it is
    inherited from the standard majority/minority partition.

NOT GRAD-CAM, AND WHY
=====================
Grad-CAM produces a spatial saliency map over convolutional feature maps;
applied to a flat penultimate vector feeding a linear head it degenerates to
gradient-times-activation, and more importantly it answers the wrong question.
Attribution through the trained head identifies the coordinates the HEAD RELIES
ON, whereas the r/s split is defined by which factor a coordinate responds to.
A coordinate can be heavily relied upon and perfectly causal.

USAGE
=====
A library, called by analyze.py; no CLI of its own. From the repository root:

    python -c "
    from common import FeatureBundle, standardize
    from identify_rs import two_concept_identify, sign_flip_identify, agreement
    b = FeatureBundle.load('features_waterbirds_test.npz')
    phi = standardize(b.phi)
    two  = two_concept_identify(phi, b.y, b.place)   # raw basis, two-concept
    flip = sign_flip_identify(phi, b.y, b.g)         # raw basis, sign-flip
    print(two['n_r'], two['n_s'], '|', flip['n_r'], flip['n_s'])
    print(agreement(two, flip, phi.shape[1]))
    "
"""

from __future__ import annotations

import numpy as np


def two_concept_identify(
    phi: np.ndarray,
    y: np.ndarray,
    place: np.ndarray,
    tau: float = 0.15,
    purity: float = 0.60,
) -> dict:
    """Split Phi using BOTH annotations directly -- the preferred rule.

    Waterbirds records the spurious attribute explicitly: `metadata.csv` carries
    `place` (0 = land, 1 = water) alongside `y` (0 = landbird, 1 = waterbird).
    The sign-flip rule below infers spuriousness indirectly, through the group
    definition; when the attribute is annotated there is no reason to do that.

    With both recoded to +/-1, each feature is decomposed over the four
    (y, place) cells as a 2x2 factorial:

        c_k = beta_0 + beta_y * y + beta_p * place + beta_int * (y * place).

    Four parameters for four cell means, so this is saturated -- an exact
    reparametrisation of the cell means, not an approximation. The coefficients
    are the standard contrasts, computed from UNWEIGHTED cell means m[a][b]:

        beta_0   = ( m[+][+] + m[+][-] + m[-][+] + m[-][-] ) / 4
        beta_y   = ( m[+][+] + m[+][-] - m[-][+] - m[-][-] ) / 4
        beta_p   = ( m[+][+] - m[+][-] + m[-][+] - m[-][-] ) / 4
        beta_int = ( m[+][+] - m[+][-] - m[-][+] + m[-][-] ) / 4

    Unweighted means matter: ordinary least squares would weight each cell by
    its size, and the waterbird-on-land cell (n = 56 in train) would be
    effectively ignored -- yet it is precisely the cell where y and place
    disagree, which is what makes the two effects separable at all.

    Classification, after scaling each feature by its pooled within-cell
    standard deviation so the coefficients are comparable across features:

        strength = sqrt(beta_y^2 + beta_p^2)          -- does the feature respond at all
        purity_r = |beta_y| / (|beta_y| + |beta_p|)   -- to which factor

        r-type if strength >= tau and purity_r >= purity
        s-type if strength >= tau and purity_r <= 1 - purity
        weak   otherwise

    `beta_int` is reported but not used for classification: a large |beta_int|
    marks a conjunction feature (responding to bird-on-mismatched-background),
    which is neither cleanly causal nor cleanly spurious and which the sign-flip
    rule would silently misfile.

    A caution the caller should propagate: y and place are strongly correlated
    on the train split (corr = 0.867), so beta_y and beta_p are identifiable but
    noisy -- variance inflation factor ~4 relative to a balanced design. The
    returned `se_approx` gives the approximate standard error shared by the
    contrasts, which is dominated by the smallest cell.
    """
    ys = np.where(np.asarray(y) > 0, 1.0, -1.0)
    ps = np.where(np.asarray(place) > 0, 1.0, -1.0)

    cells, sizes, var_acc = {}, {}, []
    for a in (1.0, -1.0):
        for b in (1.0, -1.0):
            m = (ys == a) & (ps == b)
            n = int(m.sum())
            sizes[(a, b)] = n
            if n == 0:
                raise ValueError(
                    f"cell (y={a:+.0f}, place={b:+.0f}) is empty; the two "
                    "effects are not separable without all four cells."
                )
            cells[(a, b)] = phi[m].mean(axis=0)
            if n > 1:
                var_acc.append(phi[m].var(axis=0, ddof=1) * (n - 1))
    dof = sum(sizes.values()) - 4
    pooled_sd = np.sqrt(np.sum(var_acc, axis=0) / max(dof, 1))
    pooled_sd[pooled_sd < 1e-12] = 1.0

    pp, pm = cells[(1.0, 1.0)], cells[(1.0, -1.0)]
    mp, mm = cells[(-1.0, 1.0)], cells[(-1.0, -1.0)]
    beta_y = (pp + pm - mp - mm) / 4.0 / pooled_sd
    beta_p = (pp - pm + mp - mm) / 4.0 / pooled_sd
    beta_int = (pp - pm - mp + mm) / 4.0 / pooled_sd

    # SE of an unweighted contrast: (1/4) * sqrt(sum_cells 1/n_cell), in units
    # of the pooled within-cell SD. Dominated by the smallest cell.
    se = 0.25 * float(np.sqrt(sum(1.0 / n for n in sizes.values())))

    strength = np.sqrt(beta_y ** 2 + beta_p ** 2)
    denom = np.abs(beta_y) + np.abs(beta_p)
    purity_r = np.divide(np.abs(beta_y), denom, out=np.zeros_like(denom),
                         where=denom > 1e-12)
    strong = strength >= tau
    idx_r = np.flatnonzero(strong & (purity_r >= purity))
    idx_s = np.flatnonzero(strong & (purity_r <= 1.0 - purity))

    return {
        "idx_r": idx_r,
        "idx_s": idx_s,
        "beta_y": beta_y,
        "beta_p": beta_p,
        "beta_int": beta_int,
        "strength": strength,
        "purity_r": purity_r,
        "se_approx": se,
        "cell_sizes": {f"y={int(a):+d},place={int(b):+d}": n for (a, b), n in sizes.items()},
        "min_cell": int(min(sizes.values())),
        "n_r": int(idx_r.size),
        "n_s": int(idx_s.size),
        "n_weak": int(phi.shape[1] - idx_r.size - idx_s.size),
        "n_conjunction": int((np.abs(beta_int) > np.maximum(np.abs(beta_y), np.abs(beta_p))).sum()),
        "tau": tau,
        "purity": purity,
    }


def agreement(res_two: dict, res_flip: dict, d: int) -> dict:
    """Cross-check: how far do the two identification rules agree?

    The sign-flip rule and the two-concept regression are independent routes to
    the same split, so their agreement is evidence that neither is an artefact.
    Reported as the overlap of the two r-sets and the two s-sets.
    """
    r2, s2 = set(res_two["idx_r"].tolist()), set(res_two["idx_s"].tolist())
    rf, sf = set(res_flip["idx_r"].tolist()), set(res_flip["idx_s"].tolist())

    def jac(a: set, b: set) -> float:
        return len(a & b) / len(a | b) if (a | b) else float("nan")

    labelled = (r2 | s2) & (rf | sf)
    concord = sum(1 for k in labelled
                  if (k in r2) == (k in rf) and (k in s2) == (k in sf))
    return {
        "jaccard_r": jac(r2, rf),
        "jaccard_s": jac(s2, sf),
        "n_labelled_by_both": len(labelled),
        "concordance_on_shared": concord / len(labelled) if labelled else float("nan"),
    }


def _group_corr(phi: np.ndarray, concept: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of phi with `concept`, on `mask` rows."""
    X = phi[mask]
    c = concept[mask].astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    cc = c - c.mean()
    sx = np.sqrt((Xc ** 2).sum(axis=0))
    sc = np.sqrt((cc ** 2).sum())
    denom = sx * sc
    out = np.zeros(X.shape[1])
    good = denom > 1e-12
    out[good] = (Xc[:, good] * cc[:, None]).sum(axis=0) / denom[good]
    return out


def sign_flip_identify(
    phi: np.ndarray,
    concept: np.ndarray,
    g: np.ndarray,
    tau: float = 0.2,
) -> dict:
    """Split the columns of phi into r-type and s-type by the sign-flip rule.

    Parameters
    ----------
    phi : (N, d)
        The representation (or SAE activations).
    concept : (N,)
        The causal concept. For Waterbirds/CelebA this is the label y in the
        +/-1 convention; for Colored-MNIST it was the digit class.
    g : (N,)
        Group index in {0, 1}.
    tau : float
        Minimum absolute correlation in BOTH groups for a feature to be
        classified at all. Features below tau in either group are "weak" and are
        discarded, exactly as in the manuscript's Colored-MNIST protocol.

    Returns a dict with idx_r, idx_s, the two correlation vectors, and counts.
    """
    rho0 = _group_corr(phi, concept, g == 0)
    rho1 = _group_corr(phi, concept, g == 1)
    strong = np.minimum(np.abs(rho0), np.abs(rho1)) >= tau
    idx_r = np.flatnonzero(strong & (rho0 * rho1 > 0))
    idx_s = np.flatnonzero(strong & (rho0 * rho1 < 0))
    return {
        "idx_r": idx_r,
        "idx_s": idx_s,
        "rho0": rho0,
        "rho1": rho1,
        "n_r": int(idx_r.size),
        "n_s": int(idx_s.size),
        "n_weak": int(phi.shape[1] - idx_r.size - idx_s.size),
        "tau": tau,
    }


def tau_sensitivity(
    phi: np.ndarray, concept: np.ndarray, g: np.ndarray,
    taus=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4),
) -> list[dict]:
    """Report how the r/s split changes with the threshold tau.

    tau is the one free knob in the identification step, so its influence must
    be reported rather than fixed silently at a convenient value. A conclusion
    that only holds at one tau is not a conclusion.
    """
    rho0 = _group_corr(phi, concept, g == 0)
    rho1 = _group_corr(phi, concept, g == 1)
    rows = []
    for t in taus:
        strong = np.minimum(np.abs(rho0), np.abs(rho1)) >= t
        rows.append({
            "tau": t,
            "n_r": int((strong & (rho0 * rho1 > 0)).sum()),
            "n_s": int((strong & (rho0 * rho1 < 0)).sum()),
        })
    return rows


def fit_sae(
    phi: np.ndarray,
    d_hidden_mult: int = 4,
    l1: float = 0.03,
    epochs: int = 60,
    lr: float = 1e-3,
    batch: int = 512,
    seed: int = 0,
    device: str | None = None,
) -> dict:
    """Fit a sparse autoencoder to phi and return its activations.

    This is a BASIS CHOICE ONLY -- it classifies nothing. Feed the returned
    `activations` to whichever rule you want:

        acts = fit_sae(phi)["activations"]
        two_concept_identify(acts, y, place)     # SAE basis + two-concept
        sign_flip_identify(acts, y, g)           # SAE basis + sign-flip

    Separating this from classification is deliberate: an earlier version fused
    the two, so asking for an SAE silently forced the sign-flip rule and made
    the SAE basis unavailable to the two-concept rule.

    Mirrors the manuscript's Colored-MNIST protocol: a dictionary
    M in R^{d_hidden x d} is learned with an L1 penalty on the activations,

        c = ReLU(M phi + b),   phi_hat = M^T c,
        loss = || phi - phi_hat ||^2 + l1 * || c ||_1,

    so that c_k is the activation of dictionary feature k. Working in this basis
    is the right move when raw coordinates are polysemantic (several concepts
    sharing one coordinate through superposition), which shows up as most raw
    features being classified "weak".

    Requires torch. Without it, use the raw basis -- pass phi to a rule directly.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fit_sae requires torch. Without torch, use the raw basis: pass "
            "phi straight to two_concept_identify or sign_flip_identify."
        ) from e

    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(phi, dtype=torch.float32, device=dev)
    Xm, Xs = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp_min(1e-6)
    Xn = (X - Xm) / Xs

    d = phi.shape[1]
    d_h = d_hidden_mult * d
    M = torch.nn.Parameter(torch.randn(d_h, d, device=dev) * (1.0 / np.sqrt(d)))
    b = torch.nn.Parameter(torch.zeros(d_h, device=dev))
    opt = torch.optim.Adam([M, b], lr=lr)

    n = Xn.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            xb = Xn[perm[i:i + batch]]
            c = torch.relu(xb @ M.T + b)
            rec = c @ M
            loss = ((xb - rec) ** 2).sum(dim=1).mean() + l1 * c.abs().sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        C = torch.relu(Xn @ M.T + b)
        rec = C @ M
        ss_res = ((Xn - rec) ** 2).sum().item()
        ss_tot = ((Xn - Xn.mean(0, keepdim=True)) ** 2).sum().item()
        var_explained = 1.0 - ss_res / max(ss_tot, 1e-12)
        avg_active = (C > 0).float().sum(dim=1).mean().item()
        acts = C.cpu().numpy().astype(np.float64)

    return {
        "activations": acts,
        "var_explained": float(var_explained),
        "avg_active": float(avg_active),
        "d_hidden": d_h,
    }
