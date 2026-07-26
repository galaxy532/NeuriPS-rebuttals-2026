# Rebuttal experiments — representation-level spuriousness

Standalone experiment code for the author-response period. Nothing here touches
the manuscript sources; this folder is self-contained.

**Goal.** Test, on a standard benchmark, the claim that what matters is the
coupling between `Phi_r` and `Phi_s` in the *learned representation*, not
between `r` and `s` in the raw data — and therefore that the theory's rates
apply to benchmarks that are label-mediated by construction.

---

## ⚠️ Read this before sharing anything

The NeurIPS response rules say: **no links**, with a single exception — *if
reviewers asked for code*, an anonymised link may be sent **to the AC** in an
Official Comment.

Re-reading the three reviews: **none of them asked for code.** GCa5 asked for
"at least one real-data or semi-realistic last-layer retraining experiment",
Nz4w asked for "a simulation ... the last layer of a real network", oDsv asked
nothing about code. Both are requests for *results*, not for source.

So posting a GitHub link in the rebuttal would not be covered by the exception.
The safe sequence is:

1. Put the **numbers** in the response. Everything in this folder is designed to
   emit markdown tables that paste directly into a 10,000-character reply.
   No figure is ever required.
2. **Offer** code rather than linking it: "we are happy to provide anonymised
   code to the AC if that would be helpful." Offering is safe; an unprompted
   link is not.
3. Only if a reviewer or the AC then asks, share an anonymised link **to the AC**
   via Official Comment.

If it does come to that, a personal GitHub account — even a second one — is a
poor anonymisation channel: commit author name and email, repository creation
time, fork graph and the account's other activity are all visible, and a
determined reviewer can deanonymise from any one of them. Use
`anonymous.4open.science` (the community-standard anonymiser for double-blind
venues) instead, and scrub author names, institution paths, absolute home
directory paths, and dataset paths from the code first.

Note also that the original submission already shipped code in the
supplementary material, so reviewers are not without code today.

---

## Contents

Every file here is on the path to a number that goes into the response.
The calibration harness that checks these estimators against known ground truth
lives separately, in `../Estimator_Validation/` — it produces no rebuttal
numbers and is not part of this pipeline.

| File | Role | Runs where |
|---|---|---|
| `common.py` | Core estimators: margin direction `v`, group margins, operators `A`/`B`, isotropy diagnostics, `alpha`, within-cell coupling test, logistic GD | CPU |
| `extract_features.py` | ERM training + frozen `Phi` extraction (Waterbirds) | **GPU** |
| `identify_rs.py` | `Phi_r` / `Phi_s` split by the group-conditional sign-flip rule (+ optional SAE) | CPU / GPU |
| `analyze.py` | Driver: identification → coupling test → isotropy defect → `alpha` | CPU |
| `eps_sweep.py` | Group-proportion sweep testing the predicted decay law | CPU |

## How to run

```bash
# 1. on the GPU box: ERM + freeze Phi   (the only GPU step, ~1-2 h)
python extract_features.py --root /path/to/waterbird_complete95_forest2water2 \
                           --out features_waterbirds.npz --epochs 10

# 2. identification + coupling test + isotropy defect + alpha
python analyze.py --bundle features_waterbirds.npz --tau 0.2 --n-perm 200

# 3. the group-proportion sweep, using alpha-hat from step 2
python eps_sweep.py --bundle features_waterbirds.npz --alpha <alpha_hat> \
                    --T 2000000 --h 0.05
```

Steps 2 and 3 are CPU-only and cheap: they operate on cached features. Each
writes both a `.json` and a `.md`; the `.md` is the paste-ready table.

`analyze.py` also carries two flags used only by the validation harness —
`--use-stored-split` and `--no-standardize`. Leave both off for real data.

## What each reviewer ask maps onto

| Ask | Answered by |
|---|---|
| GCa5 Q3, Nz4w Q2 — real-data / last-layer experiment | `extract_features.py` → `analyze.py` → `eps_sweep.py` |
| GCa5 Q2 — robustness of the transition outside isotropy | `iso_diagnostics`: measured eigenvector defects and `d*` on real features |
| Nz4w — "explain and extend Def 4.2", does it force `A`,`B` to commute | `iso_diagnostics` gives the *measured* defect; `d*` quantifies the reparametrisation displacement |
| Nz4w — Figure 3 unconvincing | `eps_sweep.py` emits the invariant-collapse table numerically |
| All three — is the setting real? | `within_cell_coupling`: coupling that survives conditioning on `(y, g)` |

---

## Estimator status

These estimators have been checked against synthetic data with a prescribed `α`;
the harness and its full results are in **`../Estimator_Validation/`**. Summary:
`α` recovery passes to 0.02%, the coupling test passes on both power and
calibration, and the isotropy diagnostic passes after two bugs it caught were
fixed.

One result from that harness changes how you run **step 3** and is repeated here
because it is easy to get wrong: the minority exponent `beta_min` converges to
`max(alpha, 1)` **from below, slowly** (0.716 at `z_T = 1,500`, 0.752 at
`z_T = 6,000`, against a target of 1.0), while the majority control sits at
0.94–0.98 throughout. A short run will systematically under-report `beta_min`
and look like a refutation. Budget `--T 2000000`, and report `beta_min` with its
drift across windows rather than as a converged number — `eps_sweep.py` prints
both windows by default.

## Honest limitations to keep in view

- **`alpha` is estimated, not controlled.** On Waterbirds it is whatever the
  representation gives. Report it with the isotropy defect beside it: if the
  defect is large, `alpha` is a heuristic summary, not a theorem input.
- **r-separability is not guaranteed.** The `Phi_r` block may not be linearly
  separable; `analyze.py` reports the separable fraction and warns. The margins
  use a 1% lower quantile as the `ess inf` proxy, since the sample minimum is
  driven by a single point. Both the quantile and the raw minimum are reported.
- **Standardisation is a modelling choice.** `alpha` is invariant under a common
  rescaling of `r` and `s` but *not* under per-coordinate standardisation, which
  is a diagonal transform. Real features are standardised, and the output records
  that it was applied.
- **CelebA does not map cleanly.** Defining groups by whether `Male` agrees with
  the majority blond-female pairing gives `eps ≈ 0.45` — no imbalance at all,
  because CelebA's imbalance lives *inside* the blond class. Waterbirds is the
  primary target; revisit the CelebA group definition before using its numbers.
- **A partial result is still worth reporting.** The coupling measurement alone
  answers "is this setting real?", which is the significance complaint. The rate
  verification is upside. Do not let the two stand or fall together.
