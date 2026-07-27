### Frozen-representation analysis

- n = 5794, d = 2048, empirical eps = 0.5000
- basis: **sae**, tau = 0.05
- SAE: d_hidden = 8192, variance explained = 0.9905, avg active features = 438.7
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
| -1 | 0 | 2255 | -0.004 | -0.2510 | -0.2420 | 0.0196 |  |
| -1 | 1 | 2255 | -0.018 | -0.2480 | -0.2407 | 0.0196 |  |
| +1 | 0 | 642 | -0.576 | -1.2749 | -1.2080 | 0.0196 |  |
| +1 | 1 | 642 | -0.595 | -1.3022 | -1.2443 | 0.0196 |  |

- pooled (marginal) R2 = 0.087  vs mean within-cell R2 = -0.298
- The pooled figure is inflated by label-mediation; the within-cell figure conditions on (y, g) and is the one that speaks to feature-mediation.

**Distance from the isotropic regime**

| product | defect (angle) | defect (scaled) | degenerate |
|---|---|---|---|
| A^T A | 0.9507 | 0.1274 |  |
| A^T B | 0.9992 | 0.2274 |  |
| B^T B | 0.9051 | 0.2017 |  |
| B^T A | 0.9969 | 0.1144 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 7.6818, mu_B = 17.8482, mu = -1.5991
- attractive condition -1 <= mu < min(mu_A, mu_B): False
- d_r = 363, d_s = 181, d_s >= 2 d_r: False, dim K = 4
- d* = 416.6809 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 0.991

**Margins and the exponent alpha**

- gamma-tilde_maj = nan, gamma-tilde_min = nan (orientation: undefined, ess-inf proxy: 1% quantile)
- **r-block is NOT separable at the 1% quantile.** Only 93.63% of points are correctly classified by v, so gamma-tilde_g above are nan and the orientation is undefined rather than 'mirror'. The billions in `raw_min_margin` are an artefact of the 1e-9 clamp, not measurements.
- **alpha: NOT ESTIMABLE.** the r-block is not separable at the 1% quantile -- only 93.63% of points are correctly classified by v, so the group r-margins gamma-tilde_g have no positive common scale and the WLOG min_g gamma-tilde_g = 1 cannot be imposed. alpha is a statement about a separable r-block and there is nothing here for it to describe. No regime is claimed for this rule.

---

## Rule: sign-flip

- split: n_r = 108, n_s = 47, weak = 8037

**Within-(y,g)-cell coupling, Phi_r -> Phi_s (cross-validated R^2, block-permutation null)**

p-value floor with 50 permutations: 0.0196. Cells below the minimum are refused, not reported. Refused: 0; reported but unreliable: 0.

| y | g | n | R2_cv | null mean | null q95 | p | note |
|---|---|---|---|---|---|---|---|
| -1 | 0 | 2255 | 0.105 | -0.0635 | -0.0575 | 0.0196 |  |
| -1 | 1 | 2255 | 0.127 | -0.0637 | -0.0592 | 0.0196 |  |
| +1 | 0 | 642 | 0.019 | -0.2680 | -0.2494 | 0.0196 |  |
| +1 | 1 | 642 | 0.011 | -0.2656 | -0.2435 | 0.0196 |  |

- pooled (marginal) R2 = 0.174  vs mean within-cell R2 = 0.066
- The pooled figure is inflated by label-mediation; the within-cell figure conditions on (y, g) and is the one that speaks to feature-mediation.

**Distance from the isotropic regime**

| product | defect (angle) | defect (scaled) | degenerate |
|---|---|---|---|
| A^T A | 0.7570 | 0.1040 |  |
| A^T B | 0.9979 | 0.2815 |  |
| B^T B | 0.7560 | 0.1991 |  |
| B^T A | 0.9925 | 0.1492 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 2.8658, mu_B = 5.9598, mu = +0.4392
- attractive condition -1 <= mu < min(mu_A, mu_B): True
- d_r = 108, d_s = 47, d_s >= 2 d_r: False, dim K = 0
- d* = 236.0572 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 0.985
- **d\* is VACUOUS on this run.** dim K = 0, which is forced whenever d_s <= 2 d_r and the estimated operators are full rank: M_stack is (2 d_r, d_s), so its rank saturates at d_s, K = {0}, Pi_{K^perp} = I, and d*_relative = 1.0000 exactly as arithmetic. It is restating the `d_s >= 2 d_r` flag, not measuring a distance, and the rank-cutoff sweep above cannot change that. Do not quote it.

**Margins and the exponent alpha**

- gamma-tilde_maj = nan, gamma-tilde_min = nan (orientation: undefined, ess-inf proxy: 1% quantile)
- **r-block is NOT separable at the 1% quantile.** Only 93.87% of points are correctly classified by v, so gamma-tilde_g above are nan and the orientation is undefined rather than 'mirror'. The billions in `raw_min_margin` are an artefact of the 1e-9 clamp, not measurements.
- **alpha: NOT ESTIMABLE.** the r-block is not separable at the 1% quantile -- only 93.87% of points are correctly classified by v, so the group r-margins gamma-tilde_g have no positive common scale and the WLOG min_g gamma-tilde_g = 1 cannot be imposed. alpha is a statement about a separable r-block and there is nothing here for it to describe. No regime is claimed for this rule.
