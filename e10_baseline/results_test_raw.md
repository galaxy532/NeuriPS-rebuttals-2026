### Frozen-representation analysis

- n = 5794, d = 2048, empirical eps = 0.5000
- basis: **raw**, tau = 0.15
- agreement between the two rules: Jaccard r = 0.722, s = 0.564; concordance on the 949 features both label = 1.000 (0 discordant)
  - zero discordant features, so by the rule of three the 95% upper bound on the true discordance rate is 0.3%

**Threshold sensitivity (sign-flip rule)**

| tau | n_r | n_s |
|---|---|---|
| 0.05 | 1205 | 419 |
| 0.1 | 975 | 277 |
| 0.15 | 769 | 182 |
| 0.2 | 576 | 119 |
| 0.3 | 281 | 45 |
| 0.4 | 96 | 8 |

---

## Rule: two-concept

- split: n_r = 1063, n_s = 320, weak = 665
- (y, place) cell sizes: y=+1,place=+1: 642, y=+1,place=-1: 642, y=-1,place=+1: 2255, y=-1,place=-1: 2255  (smallest 642)
- approx. SE of each contrast: 0.0158
- conjunction-dominated features (|beta_int| the largest of the three contrasts). Since y*place = 1 - 2g in Waterbirds, beta_int is the group contrast, so these respond to GROUP membership more strongly than to bird type or background -- features the Phi = (r, s) partition has no slot for.
  - among the 1724 features passing tau: **0** (0.0%)
  - over all columns, including those too weak to classify: 19 (0.9%), against a chance rate of 33.3% under exchangeable contrasts
  - the unrestricted rate is at or below chance, i.e. consistent with noise, and is NOT evidence that the (r, s) partition is violated
  - the two rates are not comparable to each other: passing tau selects on beta_y and beta_p being large, which depresses the restricted rate well below 1/3 even on pure noise (~0.11 in testing). Only the unrestricted rate has 1/3 as its null; the restricted one needs an estimated reference.

**Within-(y,g)-cell coupling, Phi_r -> Phi_s (cross-validated R^2, block-permutation null)**

p-value floor with 50 permutations: 0.0196. Cells below the minimum are refused, not reported. Refused: 0; reported but unreliable: 0.

| y | g | n | R2_cv | null mean | null q95 | p | note |
|---|---|---|---|---|---|---|---|
| -1 | 0 | 2255 | 0.729 | -1.3982 | -1.3784 | 0.0196 |  |
| -1 | 1 | 2255 | 0.747 | -1.3945 | -1.3727 | 0.0196 |  |
| +1 | 0 | 642 | 0.689 | -1.7989 | -1.7414 | 0.0196 |  |
| +1 | 1 | 642 | 0.668 | -1.8211 | -1.7574 | 0.0196 |  |

- pooled (marginal) R2 = 0.860  vs mean within-cell R2 = 0.708
- The pooled figure is inflated by label-mediation; the within-cell figure conditions on (y, g) and is the one that speaks to feature-mediation.

**Distance from the isotropic regime**

| product | defect (angle) | defect (scaled) | degenerate |
|---|---|---|---|
| A^T A | 0.8845 | 0.0974 |  |
| A^T B | 0.9919 | 0.1141 |  |
| B^T B | 0.9225 | 0.1090 |  |
| B^T A | 0.9900 | 0.1023 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 0.7231, mu_B = 0.6792, mu = +0.2017
- attractive condition -1 <= mu < min(mu_A, mu_B): True
- d_r = 1063, d_s = 320, d_s >= 2 d_r: False, dim K = 0
- d* = 94.8349 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 1.000
- **d\* is VACUOUS on this run.** dim K = 0, which is forced whenever d_s <= 2 d_r and the estimated operators are full rank: M_stack is (2 d_r, d_s), so its rank saturates at d_s, K = {0}, Pi_{K^perp} = I, and d*_relative = 1.0000 exactly as arithmetic. It is restating the `d_s >= 2 d_r` flag, not measuring a distance, and the rank-cutoff sweep above cannot change that. Do not quote it.

**Margins and the exponent alpha**

- gamma-tilde_maj = 1.0000, gamma-tilde_min = 1.6403 (orientation: standard, ess-inf proxy: 1% quantile)
- r-block IS separable at the 1% quantile; fraction correctly classified = 0.9934
- **alpha = 1.1439** -> alpha >= 1 (geometry dominates); predicted minority exponent max(alpha,1) = 1.1439

---

## Rule: sign-flip

- split: n_r = 769, n_s = 182, weak = 1097

**Within-(y,g)-cell coupling, Phi_r -> Phi_s (cross-validated R^2, block-permutation null)**

p-value floor with 50 permutations: 0.0196. Cells below the minimum are refused, not reported. Refused: 0; reported but unreliable: 0.

| y | g | n | R2_cv | null mean | null q95 | p | note |
|---|---|---|---|---|---|---|---|
| -1 | 0 | 2255 | 0.748 | -0.7464 | -0.7365 | 0.0196 |  |
| -1 | 1 | 2255 | 0.780 | -0.7459 | -0.7305 | 0.0196 |  |
| +1 | 0 | 642 | 0.585 | -2.5205 | -2.4062 | 0.0196 |  |
| +1 | 1 | 642 | 0.504 | -2.5800 | -2.5075 | 0.0196 |  |

- pooled (marginal) R2 = 0.852  vs mean within-cell R2 = 0.654
- The pooled figure is inflated by label-mediation; the within-cell figure conditions on (y, g) and is the one that speaks to feature-mediation.

**Distance from the isotropic regime**

| product | defect (angle) | defect (scaled) | degenerate |
|---|---|---|---|
| A^T A | 0.9484 | 0.1184 |  |
| A^T B | 0.9843 | 0.0889 |  |
| B^T B | 0.9364 | 0.0867 |  |
| B^T A | 0.9906 | 0.1157 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 0.5255, mu_B = 0.4841, mu = +0.2197
- attractive condition -1 <= mu < min(mu_A, mu_B): True
- d_r = 769, d_s = 182, d_s >= 2 d_r: False, dim K = 0
- d* = 74.9177 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 0.999
- **d\* is VACUOUS on this run.** dim K = 0, which is forced whenever d_s <= 2 d_r and the estimated operators are full rank: M_stack is (2 d_r, d_s), so its rank saturates at d_s, K = {0}, Pi_{K^perp} = I, and d*_relative = 1.0000 exactly as arithmetic. It is restating the `d_s >= 2 d_r` flag, not measuring a distance, and the rank-cutoff sweep above cannot change that. Do not quote it.

**Margins and the exponent alpha**

- gamma-tilde_maj = nan, gamma-tilde_min = nan (orientation: undefined, ess-inf proxy: 1% quantile)
- **r-block is NOT separable at the 1% quantile.** Only 90.02% of points are correctly classified by v, so gamma-tilde_g above are nan and the orientation is undefined rather than 'mirror'. The billions in `raw_min_margin` are an artefact of the 1e-9 clamp, not measurements.
- **alpha: NOT ESTIMABLE.** the r-block is not separable at the 1% quantile -- only 90.02% of points are correctly classified by v, so the group r-margins gamma-tilde_g have no positive common scale and the WLOG min_g gamma-tilde_g = 1 cannot be imposed. alpha is a statement about a separable r-block and there is nothing here for it to describe. No regime is claimed for this rule.
