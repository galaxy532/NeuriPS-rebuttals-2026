### Frozen-representation analysis

- n = 5794, d = 2048, empirical eps = 0.5000
- basis: **sae**, tau = 0.05
- SAE: d_hidden = 8192, variance explained = 0.9905, avg active features = 438.7
  - dead units (never active): 0; near-constant: 0; activations standardised: True
  - activations are standardised AFTER the autoencoder: the encoder's ReLU destroys the centring of its input, and the margin and operator fits carry no intercept, so they need centred columns to be well posed.
- agreement between the two rules: Jaccard r = 0.290, s = 0.253; concordance on the 152 features both label = 1.000 (0 discordant)
  - zero discordant features, so by the rule of three the 95% upper bound on the true discordance rate is 2.0%

**Threshold sensitivity (sign-flip rule)**

| tau | n_r | n_s |
|---|---|---|
| 0.05 | 108 | 47 |
| 0.1 | 33 | 16 |
| 0.15 | 15 | 5 |
| 0.2 | 7 | 3 |
| 0.3 | 2 | 0 |
| 0.4 | 1 | 0 |

---

## Rule: two-concept

- split: n_r = 363, n_s = 181, weak = 7648
- (y, place) cell sizes: y=+1,place=+1: 642, y=+1,place=-1: 642, y=-1,place=+1: 2255, y=-1,place=-1: 2255  (smallest 642)
- approx. SE of each contrast: 0.0158
- conjunction-dominated features (|beta_int| the largest of the three contrasts). Since y*place = 1 - 2g in Waterbirds, beta_int is the group contrast, so these respond to GROUP membership more strongly than to bird type or background -- features the Phi = (r, s) partition has no slot for.
  - among the 731 features passing tau: **29** (4.0%)
  - over all columns, including those too weak to classify: 2066 (25.2%), against a chance rate of 33.3% under exchangeable contrasts
  - the unrestricted rate is at or below chance, i.e. consistent with noise, and is NOT evidence that the (r, s) partition is violated
  - the two rates are not comparable to each other: passing tau selects on beta_y and beta_p being large, which depresses the restricted rate well below 1/3 even on pure noise (~0.11 in testing). Only the unrestricted rate has 1/3 as its null; the restricted one needs an estimated reference.

**Within-(y,g)-cell coupling, Phi_r -> Phi_s (cross-validated R^2, block-permutation null)**

p-value floor with 50 permutations: 0.0196. Cells below the minimum are refused, not reported. Refused: 0; reported but unreliable: 0.

| y | g | n | R2_cv | null mean | null q95 | p | note |
|---|---|---|---|---|---|---|---|
| -1 | 0 | 2255 | -0.186 | -0.2772 | -0.2727 | 0.0196 |  |
| -1 | 1 | 2255 | -0.195 | -0.2770 | -0.2731 | 0.0196 |  |
| +1 | 0 | 642 | -2.243 | -2.8946 | -2.8009 | 0.0196 |  |
| +1 | 1 | 642 | -2.272 | -2.9473 | -2.8259 | 0.0196 |  |

- pooled (marginal) R2 = -0.032  vs mean within-cell R2 = -1.224
- The pooled figure is inflated by label-mediation; the within-cell figure conditions on (y, g) and is the one that speaks to feature-mediation.

**Distance from the isotropic regime**

| product | defect (angle) | defect (scaled) | degenerate |
|---|---|---|---|
| A^T A | 0.9134 | 0.1692 |  |
| A^T B | 0.9989 | 0.1765 |  |
| B^T B | 0.8317 | 0.1396 |  |
| B^T A | 0.9992 | 0.2086 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- RAW fitted operators (diagnostic only): mu_A = 0.1447, mu_B = 0.1794, mu = -0.0127; attractive -1 <= mu < min(mu_A, mu_B): True

**Isotropic reparametrisation (Section C, Eqs 37-40)**

- **No isotropic (A', B') exists: dim K = 0.** The two kernels of Eq (38) intersect trivially, so there is no admissible A'v, the primed eigenvalues are all 0 by degeneracy rather than by measurement, and Theorem D.4 has no hypothesis to stand on. Section C guarantees dim K > 0 when d_s >= 2 d_r - 1; that is sufficient, not necessary, so a rank-deficient fit can still give dim K > 0 without it.
- d_r = 363, d_s = 181, d_s >= 2 d_r: False, dim K = 0
- d* = 59.2566 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 1.000
- **d\* is VACUOUS on this run.** dim K = 0, which is forced whenever d_s <= 2 d_r and the estimated operators are full rank: M_stack is (2 d_r, d_s), so its rank saturates at d_s, K = {0}, Pi_{K^perp} = I, and d*_relative = 1.0000 exactly as arithmetic. It is restating the `d_s >= 2 d_r` flag, not measuring a distance, and the rank-cutoff sweep above cannot change that. Do not quote it.

**Margins and the exponent alpha**

- gamma-tilde_maj = nan, gamma-tilde_min = nan (orientation: undefined, ess-inf proxy: 1% quantile)
- **r-block is NOT separable at the 1% quantile.** Only 84.47% of points are correctly classified by v, so gamma-tilde_g above are nan and the orientation is undefined rather than 'mirror'. The billions in `raw_min_margin` are an artefact of the 1e-9 clamp, not measurements.
- **alpha: NOT ESTIMABLE.** dim K = 0, so no isotropic reparametrisation (A', B') exists (Section C, Eq 38): the intersection of the two kernels is trivial, there is no admissible A'v, and Theorem D.4's hypothesis is unavailable. Section C guarantees dim K > 0 only when d_s >= 2 d_r - 1, which this split does not satisfy; note that is a SUFFICIENT condition, so dim K can be positive without it when the fitted operators are rank deficient. No regime is claimed for this rule.

**alpha under q-relaxed r-separability**

The manuscript's gamma-tilde_g is an essential infimum, i.e. q -> 0. The separability probe puts the population error rate of the best tuned linear rule near 5-6% on this representation, so no q below that can be met and refusing alpha there says nothing about the geometry. Reading across rows shows where separability becomes attainable and how much alpha depends on that choice.

| q | separable | gamma-tilde_maj | gamma-tilde_min | orientation | alpha |
|---|---|---|---|---|---|
| 0.005 | False | - | - | undefined | not estimable |
| 0.01 | False | - | - | undefined | not estimable |
| 0.02 | False | - | - | undefined | not estimable |
| 0.05 | False | - | - | undefined | not estimable |
| 0.08 | False | - | - | undefined | not estimable |
| 0.1 | False | - | - | undefined | not estimable |
| 0.15 | False | - | - | undefined | not estimable |

- smallest q at which the r-block separates: None; smallest q at which alpha is estimable: None

---

## Rule: sign-flip

- split: n_r = 108, n_s = 47, weak = 8037

**Within-(y,g)-cell coupling, Phi_r -> Phi_s (cross-validated R^2, block-permutation null)**

p-value floor with 50 permutations: 0.0196. Cells below the minimum are refused, not reported. Refused: 0; reported but unreliable: 0.

| y | g | n | R2_cv | null mean | null q95 | p | note |
|---|---|---|---|---|---|---|---|
| -1 | 0 | 2255 | 0.016 | -0.0661 | -0.0626 | 0.0196 |  |
| -1 | 1 | 2255 | 0.026 | -0.0661 | -0.0625 | 0.0196 |  |
| +1 | 0 | 642 | -0.087 | -0.2952 | -0.2741 | 0.0196 |  |
| +1 | 1 | 642 | -0.128 | -0.3034 | -0.2790 | 0.0196 |  |

- pooled (marginal) R2 = 0.061  vs mean within-cell R2 = -0.043
- The pooled figure is inflated by label-mediation; the within-cell figure conditions on (y, g) and is the one that speaks to feature-mediation.

**Distance from the isotropic regime**

| product | defect (angle) | defect (scaled) | degenerate |
|---|---|---|---|
| A^T A | 0.8693 | 0.2304 |  |
| A^T B | 0.8997 | 0.1384 |  |
| B^T B | 0.8719 | 0.1197 |  |
| B^T A | 0.9330 | 0.1742 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- RAW fitted operators (diagnostic only): mu_A = 0.1713, mu_B = 0.1403, mu = +0.0981; attractive -1 <= mu < min(mu_A, mu_B): True

**Isotropic reparametrisation (Section C, Eqs 37-40)**

- **No isotropic (A', B') exists: dim K = 0.** The two kernels of Eq (38) intersect trivially, so there is no admissible A'v, the primed eigenvalues are all 0 by degeneracy rather than by measurement, and Theorem D.4 has no hypothesis to stand on. Section C guarantees dim K > 0 when d_s >= 2 d_r - 1; that is sufficient, not necessary, so a rank-deficient fit can still give dim K > 0 without it.
- d_r = 108, d_s = 47, d_s >= 2 d_r: False, dim K = 0
- d* = 31.6850 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 1.000
- **d\* is VACUOUS on this run.** dim K = 0, which is forced whenever d_s <= 2 d_r and the estimated operators are full rank: M_stack is (2 d_r, d_s), so its rank saturates at d_s, K = {0}, Pi_{K^perp} = I, and d*_relative = 1.0000 exactly as arithmetic. It is restating the `d_s >= 2 d_r` flag, not measuring a distance, and the rank-cutoff sweep above cannot change that. Do not quote it.

**Margins and the exponent alpha**

- gamma-tilde_maj = nan, gamma-tilde_min = nan (orientation: undefined, ess-inf proxy: 1% quantile)
- **r-block is NOT separable at the 1% quantile.** Only 86.54% of points are correctly classified by v, so gamma-tilde_g above are nan and the orientation is undefined rather than 'mirror'. The billions in `raw_min_margin` are an artefact of the 1e-9 clamp, not measurements.
- **alpha: NOT ESTIMABLE.** dim K = 0, so no isotropic reparametrisation (A', B') exists (Section C, Eq 38): the intersection of the two kernels is trivial, there is no admissible A'v, and Theorem D.4's hypothesis is unavailable. Section C guarantees dim K > 0 only when d_s >= 2 d_r - 1, which this split does not satisfy; note that is a SUFFICIENT condition, so dim K can be positive without it when the fitted operators are rank deficient. No regime is claimed for this rule.

**alpha under q-relaxed r-separability**

The manuscript's gamma-tilde_g is an essential infimum, i.e. q -> 0. The separability probe puts the population error rate of the best tuned linear rule near 5-6% on this representation, so no q below that can be met and refusing alpha there says nothing about the geometry. Reading across rows shows where separability becomes attainable and how much alpha depends on that choice.

| q | separable | gamma-tilde_maj | gamma-tilde_min | orientation | alpha |
|---|---|---|---|---|---|
| 0.005 | False | - | - | undefined | not estimable |
| 0.01 | False | - | - | undefined | not estimable |
| 0.02 | False | - | - | undefined | not estimable |
| 0.05 | False | - | - | undefined | not estimable |
| 0.08 | False | - | - | undefined | not estimable |
| 0.1 | False | - | - | undefined | not estimable |
| 0.15 | True | - | - | undefined | not estimable |

- smallest q at which the r-block separates: 0.15; smallest q at which alpha is estimable: None
