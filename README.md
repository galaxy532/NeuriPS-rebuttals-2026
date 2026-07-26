# Rebuttal experiments — representation-level spuriousness

Standalone experiment code for the author-response period. Nothing here touches
the manuscript sources; this folder is self-contained.

Assumed layout — commands are run from the repository root, and the dataset
lives beside the repo, never inside it:

```
<paperspace root>/
├── NeuriPS-rebuttals-2026/     <- you are here; run all commands from here
│   ├── download_waterbirds.py
│   ├── extract_features.py
│   └── ...
└── data/
    └── waterbird_complete95_forest2water2/
```

**Goal.** Test, on a standard benchmark, the claim that what matters is the
coupling between `Phi_r` and `Phi_s` in the *learned representation*, not
between `r` and `s` in the raw data — and therefore that the theory's rates
apply to benchmarks that are label-mediated by construction.

---

## Contents

Every file here is on the path to a number that goes into the response.
The calibration harness that checks these estimators against known ground truth
lives separately, in `../Estimator_Validation/` — it produces no rebuttal
numbers and is not part of this pipeline.

| File | Role | Runs where |
|---|---|---|
| `common.py` | Core estimators: margin direction `v`, group margins, operators `A`/`B`, isotropy diagnostics, `alpha`, within-cell coupling test, logistic GD | CPU |
| `download_waterbirds.py` | Fetch + extract + verify the dataset (no credentials needed) | CPU |
| `extract_features.py` | ERM training + frozen `Phi` extraction (Waterbirds) | **GPU** |
| `identify_rs.py` | `Phi_r` / `Phi_s` split by the group-conditional sign-flip rule (+ optional SAE) | CPU / GPU |
| `analyze.py` | Driver: identification → coupling test → isotropy defect → `alpha` | CPU |
| `eps_sweep.py` | Group-proportion sweep testing the predicted decay law | CPU |

## How to run

Run every command **from the repository root**. All defaults are relative, so
there are no paths to type:

```bash
# 0. fetch the dataset (~1.2 GB, public, no token) into ../data
python download_waterbirds.py

# 1. ERM + freeze Phi, caching train and test in one pass  (only GPU step, ~1-2 h)
python extract_features.py --cleanup

# 2. identification + coupling test + isotropy defect + alpha
python analyze.py

# 3. the group-proportion sweep, using the alpha-hat printed by step 2
python eps_sweep.py --alpha <alpha_hat>
```

That is the whole run. Steps 0 and 1 can be collapsed with
`python extract_features.py --download --cleanup`.

Where things live:

| | path |
|---|---|
| dataset | `../data/waterbird_complete95_forest2water2` (sibling of the repo) |
| cached features | `features_waterbirds_{train,test}.npz` in the repo (git-ignored) |
| results | `results.{json,md}`, `eps_sweep.{json,md}` (git-ignored) |

Steps 2 and 3 are CPU-only and operate purely on cached features. Each writes a
`.json` and a `.md`; the `.md` is the paste-ready table.

`analyze.py` also carries two flags used only by the validation harness —
`--use-stored-split` and `--no-standardize`. Leave both off for real data.

### Do you need to keep the 1.2 GB?

No — **once the features are cached, the raw dataset is disposable.** Steps 2
and 3 never open an image. `extract_features.py` trains once and extracts all
requested splits in the same pass precisely so that one visit to the raw data is
enough, and `--cleanup` deletes it afterwards.

The trade is roughly 1.2 GB of images for ~40–50 MB of cached features per
split. The only reason to keep the raw data is if you expect to **re-extract**:
a different backbone, more epochs, another seed, or a split you did not ask for
the first time. If there is any chance of that, run without `--cleanup` — a
re-download costs ~10 minutes, which is worse than it sounds on the day of a
deadline. Safe middle ground: keep it until `analyze.py` has produced a coupling
result you are happy with, then delete.

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
