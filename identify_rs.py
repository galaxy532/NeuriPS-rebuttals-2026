"""
identify_rs.py -- Splitting a frozen representation into Phi_r and Phi_s.

This deliberately does NOT use Grad-CAM. Grad-CAM produces a spatial saliency
map over convolutional feature maps; applied to a flat penultimate vector
feeding a linear head it degenerates to gradient-times-activation, and more
importantly it answers the wrong question. Attribution through the trained head
identifies the coordinates the HEAD RELIES ON, whereas the manuscript's
r/s split is defined by whether the coordinate's relationship to the label is
GROUP-DEPENDENT. A coordinate can be heavily relied upon and perfectly causal.

We therefore reuse the identification rule already validated in the
manuscript's controlled Colored-MNIST experiment: the group-conditional
correlation sign-flip rule.

The rule
--------
Let c_k be the activation of feature k and let `concept` be the causal concept
(the digit class in Colored-MNIST; the bird label y in Waterbirds, where the
group is defined by whether the background agrees with the label). Define

    rho_0(k) = corr(c_k, concept | g = 0),
    rho_1(k) = corr(c_k, concept | g = 1).

A causal (r-type) feature tracks the concept itself, so its correlation has the
SAME sign in both groups. A spurious (s-type) feature tracks the group-dependent
attribute, whose relationship to the concept REVERSES between groups, so its
correlation flips sign. Hence

    f_k is r-type  if rho_0 rho_1 > 0 and min(|rho_0|, |rho_1|) >= tau,
    f_k is s-type  if rho_0 rho_1 < 0 and min(|rho_0|, |rho_1|) >= tau,
    f_k is weak    otherwise.

This is exactly the operational content of A != B in the manuscript's
Definition (features-mediated linear spurious correlation): the two groups are
distinguished by the sign of the alignment between the spurious block and the
causal signal.

Two routes are provided:

  * `sign_flip_identify` applies the rule directly to the coordinates of Phi.
    Cheap, no extra training, and adequate for post-ReLU backbone features.
  * `sae_identify` first fits a sparse autoencoder to Phi and applies the rule
    to the dictionary activations. This is the exact analogue of the
    manuscript's Colored-MNIST protocol and is preferable when raw coordinates
    are polysemantic (feature superposition), but it requires torch.
"""

from __future__ import annotations

import numpy as np


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


def sae_identify(
    phi: np.ndarray,
    concept: np.ndarray,
    g: np.ndarray,
    d_hidden_mult: int = 4,
    l1: float = 0.03,
    epochs: int = 60,
    lr: float = 1e-3,
    batch: int = 512,
    tau: float = 0.2,
    seed: int = 0,
    device: str | None = None,
) -> dict:
    """Fit a sparse autoencoder to phi, then apply the sign-flip rule to it.

    Mirrors the manuscript's Colored-MNIST protocol: a dictionary
    M in R^{d_hidden x d} is learned with an L1 penalty on the activations,

        c = ReLU(M phi + b),   phi_hat = M^T c,
        loss = || phi - phi_hat ||^2 + l1 * || c ||_1,

    so that c_k is the activation of dictionary feature k. The sign-flip rule is
    then applied to c rather than to the raw coordinates of phi, which is the
    right move when raw coordinates are polysemantic (several concepts sharing
    one coordinate through superposition).

    Requires torch. If torch is unavailable, use `sign_flip_identify` directly
    on phi instead.
    """
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "sae_identify requires torch. Either install torch or use "
            "sign_flip_identify(phi, concept, g) on the raw coordinates."
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

    res = sign_flip_identify(acts, concept, g, tau=tau)
    res.update({
        "activations": acts,
        "var_explained": float(var_explained),
        "avg_active": float(avg_active),
        "d_hidden": d_h,
    })
    return res
