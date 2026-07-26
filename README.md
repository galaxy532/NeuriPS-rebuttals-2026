# Rebuttal experiments — representation-level spuriousness

**Goal.** Show on Waterbirds that what matters is the coupling between `Phi_r`
and `Phi_s` in the *learned representation*, not between `r` and `s` in the raw
data — so the theory's rates apply to benchmarks that are label-mediated by
construction.

---

# What to run

## Setup

```bash
pip install numpy scipy scikit-learn pandas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## The four commands

Run all of them **from the repository root**. Every path is relative — there is
nothing to configure.

```bash
python download_waterbirds.py          # 1. get the data      (~10 min)
python extract_features.py             # 2. train + freeze Phi (~1-2 h, GPU)
python analyze.py                      # 3. the coupling result (~5 min)
python eps_sweep.py --alpha <A>        # 4. the rate result    (~1 h)
```

`<A>` is the alpha printed by step 3. That's it.

Before step 2 finishes you can sanity-check the estimators — no data or GPU
needed:

```bash
cd Estimator_Validation && python validate.py && cd ..
```

## What each step gives you, and what to look for

| Step | Produces | The number that matters |
|---|---|---|
| 3 `analyze.py` | `results.md` | **within-cell R²** and its p-value |
| 4 `eps_sweep.py` | `eps_sweep.md` | **minority invariant CV** (0 = perfect collapse) |

**Step 3 is the one that carries the rebuttal.** Read the within-cell R² table:

- *Within-cell R² clearly above the null, p at the floor* → the representation
  entangles bird and background even after conditioning on both. This is the
  result. It answers "is this setting real?", which is the significance
  complaint from GCa5 and Nz4w.
- *Within-cell R² at null level* → Φ preserved conditional independence here.
  Report it honestly; the label/feature-mediated distinction stays load-bearing
  and the second manuscript should not be reframed.

Also in `results.md`: the isotropy defects and `d*` answer GCa5 Q2 and Nz4w's
"explain Def 4.2" with *measured numbers on real features*, which is stronger
than any purely theoretical reply.

**Step 4 is upside, not a prerequisite.** If it doesn't converge in time, step 3
still stands on its own. Do not let the two stand or fall together.

## If something looks wrong

| Symptom | Cause | Fix |
|---|---|---|
| `needs the place annotation` | bundle predates the two-concept rule | rerun step 2 (`--from-checkpoint` if you already trained) |
| `n_r` or `n_s` < 2 | threshold too strict | `--tau 0.10` |
| cells "REFUSED" | you used the train bundle | drop `--bundle`; the default is test |
| `beta_min` far below `max(alpha,1)` | run too short | expected — see the note below |
| `separable fraction` well under 1 | `Phi_r` isn't separable | α is diagnostic only; say so |

**Expect `beta_min` to look too low.** It converges to `max(alpha, 1)` from
*below*, slowly: 0.716 at `z_T = 1,500`, 0.752 at `z_T = 6,000`, against a
target of 1.0. The majority control sits at 0.94–0.98 throughout — that is what
tells you the machinery is right rather than broken. Report `beta_min` with its
drift across windows, never as a converged number. `eps_sweep.py` prints both
windows by default.

---

# ⚠️ Before sharing anything

The NeurIPS rules say **no links**, with one exception: *if reviewers asked for
code*, an anonymised link may go **to the AC** in an Official Comment.

**None of the three reviews asked for code.** They asked for *results*. So a
GitHub link in the rebuttal is not covered by the exception. Instead:

1. Put the **numbers** in the response — every script emits paste-ready markdown
   tables, and no figure is ever required.
2. **Offer** rather than link: "we are happy to provide anonymised code to the
   AC if that would be helpful." Offering is safe.
3. Only if asked, share via `anonymous.4open.science` — not a second GitHub
   account, whose commit metadata, creation time and other activity all
   deanonymise you.

The original submission already shipped code in the supplementary material, so
reviewers are not empty-handed today.

---

# Reference

## Layout

```
<paperspace root>/
├── NeuriPS-rebuttals-2026/     <- run all commands from here
└── data/
    └── waterbird_complete95_forest2water2/
```

| | path |
|---|---|
| dataset | `../data/...` (sibling of the repo, never inside it) |
| cached features | `features_waterbirds_{train,test}.npz` (git-ignored) |
| results | `results.{json,md}`, `eps_sweep.{json,md}` (git-ignored) |

## Files

| File | Role | Runs where |
|---|---|---|
| `download_waterbirds.py` | fetch + extract + verify | CPU |
| `extract_features.py` | ERM + freeze `Phi`, all splits in one pass | **GPU** |
| `identify_rs.py` | `Phi_r`/`Phi_s` split: two-concept rule + sign-flip cross-check | CPU |
| `analyze.py` | coupling test → isotropy defect → `alpha` | CPU |
| `eps_sweep.py` | group-proportion sweep, the predicted decay law | CPU |
| `common.py` | shared estimators | CPU |

Calibration checks live in `../Estimator_Validation/` and produce no rebuttal
numbers. Every script has runnable examples in its docstring.

## Why test split for step 3 and train split for step 4

Step 3 conditions on the four `(y, place)` cells. On the **train** split those
are 3498 / 184 / 56 / 1057 — a 56-sample cell cannot support a multi-output
ridge, and cross-validation stops the estimate being spuriously high but cannot
make it stable. The **test** split is built roughly balanced
(~2255/2255/642/642), so the smallest cell is ~642. Cells under 200 are now
refused outright rather than reported.

This is principled, not convenient: step 3 asks about the frozen representation
Φ, which is the same function whichever images pass through it. Step 4 asks
about training dynamics under group imbalance, so it needs the train split where
the imbalance lives.

## How `Phi_r` / `Phi_s` are identified

Waterbirds annotates the spurious attribute directly — `metadata.csv` has
`place` (land/water) next to `y` (landbird/waterbird). The default
**two-concept** rule decomposes each feature over the four `(y, place)` cells as
a 2×2 factorial and classifies it by which factor it responds to, using
unweighted cell means so the small disagreeing cells still count.

The **sign-flip** rule from the Colored-MNIST experiment runs alongside as an
independent cross-check; `results.md` reports the agreement between them.

On a Waterbirds-shaped toy with a known split, mean F1 across seeds in a
weak-signal regime:

| regime | smallest cell | two-concept | sign-flip |
|---|---|---|---|
| train-like | 110 | **0.971** | 0.932 |
| test-like | 1224 | **0.978** | 0.963 |

Two-concept wins in both, and by more when cells are small.

## Known limitations

- **`alpha` is estimated, not controlled.** Report it next to the isotropy
  defect: if the defect is large, `alpha` is a heuristic summary, not a theorem
  input.
- **r-separability isn't guaranteed.** `analyze.py` reports the separable
  fraction and warns. Margins use a 1% lower quantile as the `ess inf` proxy;
  the raw minimum is reported too.
- **Standardisation is a modelling choice.** `alpha` is invariant under a common
  rescaling of `r` and `s` but not under per-coordinate standardisation. Real
  features are standardised and the output records it.
- **CelebA does not map cleanly.** Grouping by whether `Male` agrees with the
  majority blond-female pairing gives `eps ≈ 0.45` — no imbalance, because
  CelebA's imbalance lives *inside* the blond class. Waterbirds is the target.
- **The download path is untested** — it is the one step I could not execute.
  Run step 1 early rather than discovering a dead URL on deadline day.
