### Frozen-representation analysis

- n = 5794, d = 2048, empirical eps = 0.5000
- basis: **sae**, tau = 0.15
- SAE: d_hidden = 8192, variance explained = 0.9905, avg active features = 438.7
- agreement between the two rules: Jaccard r = 0.394, s = 0.312; concordance on the 18 features both label = 1.000 (0 discordant)
  - **Caution:** only 18 features are labelled by both rules; the concordance is a small-sample proportion whose 95% upper bound on discordance is still 16.7%. Do not quote it as agreement between the rules.

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

- split: n_r = 31, n_s = 16, weak = 8145
- (y, place) cell sizes: y=+1,place=+1: 642, y=+1,place=-1: 642, y=-1,place=+1: 2255, y=-1,place=-1: 2255  (smallest 642)
- approx. SE of each contrast: 0.0158
- conjunction-dominated features (|beta_int| the largest of the three contrasts). Since y*place = 1 - 2g in Waterbirds, beta_int is the group contrast, so these respond to GROUP membership more strongly than to bird type or background -- features the Phi = (r, s) partition has no slot for.
  - among the 54 features passing tau: **0** (0.0%)
  - over all columns, including those too weak to classify: 2066 (25.2%), against a chance rate of 33.3% under exchangeable contrasts
  - the unrestricted rate is at or below chance, i.e. consistent with noise, and is NOT evidence that the (r, s) partition is violated
  - the two rates are not comparable to each other: passing tau selects on beta_y and beta_p being large, which depresses the restricted rate well below 1/3 even on pure noise (~0.11 in testing). Only the unrestricted rate has 1/3 as its null; the restricted one needs an estimated reference.

**Within-(y,g)-cell coupling, Phi_r -> Phi_s (cross-validated R^2, block-permutation null)**

p-value floor with 50 permutations: 0.0196. Cells below the minimum are refused, not reported. Refused: 0; reported but unreliable: 0.

| y | g | n | R2_cv | null mean | null q95 | p | note |
|---|---|---|---|---|---|---|---|
| -1 | 0 | 2255 | 0.016 | -0.0171 | -0.0141 | 0.0196 |  |
| -1 | 1 | 2255 | 0.024 | -0.0173 | -0.0142 | 0.0196 |  |
| +1 | 0 | 642 | 0.023 | -0.0681 | -0.0570 | 0.0196 |  |
| +1 | 1 | 642 | 0.051 | -0.0652 | -0.0556 | 0.0196 |  |

- pooled (marginal) R2 = 0.036  vs mean within-cell R2 = 0.029
- The pooled figure is inflated by label-mediation; the within-cell figure conditions on (y, g) and is the one that speaks to feature-mediation.

**Distance from the isotropic regime**

| product | defect (angle) | defect (scaled) | degenerate |
|---|---|---|---|
| A^T A | 0.7756 | 0.2185 |  |
| A^T B | 0.9703 | 0.6524 |  |
| B^T B | 0.7284 | 0.4729 |  |
| B^T A | 0.8014 | 0.2179 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 0.6650, mu_B = 5.1128, mu = -0.6716
- attractive condition -1 <= mu < min(mu_A, mu_B): True
- d_r = 31, d_s = 16, d_s >= 2 d_r: False, dim K = 0
- d* = 168.5583 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 0.994
- **d\* is VACUOUS on this run.** dim K = 0, which is forced whenever d_s <= 2 d_r and the estimated operators are full rank: M_stack is (2 d_r, d_s), so its rank saturates at d_s, K = {0}, Pi_{K^perp} = I, and d*_relative = 1.0000 exactly as arithmetic. It is restating the `d_s >= 2 d_r` flag, not measuring a distance, and the rank-cutoff sweep above cannot change that. Do not quote it.

**Margins and the exponent alpha**

- gamma-tilde_maj = nan, gamma-tilde_min = nan (orientation: undefined, ess-inf proxy: 1% quantile)
- **r-block is NOT separable at the 1% quantile.** Only 89.44% of points are correctly classified by v, so gamma-tilde_g above are nan and the orientation is undefined rather than 'mirror'. The billions in `raw_min_margin` are an artefact of the 1e-9 clamp, not measurements.
- **alpha: NOT ESTIMABLE.** the r-block is not separable at the 1% quantile -- only 89.44% of points are correctly classified by v, so the group r-margins gamma-tilde_g have no positive common scale and the WLOG min_g gamma-tilde_g = 1 cannot be imposed. alpha is a statement about a separable r-block and there is nothing here for it to describe. No regime is claimed for this rule.

---

## Rule: sign-flip

- split: n_r = 15, n_s = 5, weak = 8172

**Within-(y,g)-cell coupling, Phi_r -> Phi_s (cross-validated R^2, block-permutation null)**

p-value floor with 50 permutations: 0.0196. Cells below the minimum are refused, not reported. Refused: 0; reported but unreliable: 0.

| y | g | n | R2_cv | null mean | null q95 | p | note |
|---|---|---|---|---|---|---|---|
| -1 | 0 | 2255 | 0.023 | -0.0084 | -0.0047 | 0.0196 |  |
| -1 | 1 | 2255 | 0.021 | -0.0082 | -0.0058 | 0.0196 |  |
| +1 | 0 | 642 | 0.039 | -0.0304 | -0.0206 | 0.0196 |  |
| +1 | 1 | 642 | 0.050 | -0.0312 | -0.0198 | 0.0196 |  |

- pooled (marginal) R2 = 0.064  vs mean within-cell R2 = 0.033
- The pooled figure is inflated by label-mediation; the within-cell figure conditions on (y, g) and is the one that speaks to feature-mediation.

**Distance from the isotropic regime**

| product | defect (angle) | defect (scaled) | degenerate |
|---|---|---|---|
| A^T A | 0.6704 | 0.3998 |  |
| A^T B | 0.8769 | 0.7471 |  |
| B^T B | 0.4844 | 0.4222 |  |
| B^T A | 0.5469 | 0.2676 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 0.1296, mu_B = 1.4827, mu = +0.2890
- attractive condition -1 <= mu < min(mu_A, mu_B): False
- d_r = 15, d_s = 5, d_s >= 2 d_r: False, dim K = 0
- d* = 86.2850 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 1.000
- **d\* is VACUOUS on this run.** dim K = 0, which is forced whenever d_s <= 2 d_r and the estimated operators are full rank: M_stack is (2 d_r, d_s), so its rank saturates at d_s, K = {0}, Pi_{K^perp} = I, and d*_relative = 1.0000 exactly as arithmetic. It is restating the `d_s >= 2 d_r` flag, not measuring a distance, and the rank-cutoff sweep above cannot change that. Do not quote it.

**Margins and the exponent alpha**

- gamma-tilde_maj = nan, gamma-tilde_min = nan (orientation: undefined, ess-inf proxy: 1% quantile)
- **r-block is NOT separable at the 1% quantile.** Only 79.24% of points are correctly classified by v, so gamma-tilde_g above are nan and the orientation is undefined rather than 'mirror'. The billions in `raw_min_margin` are an artefact of the 1e-9 clamp, not measurements.
- **alpha: NOT ESTIMABLE.** the r-block is not separable at the 1% quantile -- only 79.24% of points are correctly classified by v, so the group r-margins gamma-tilde_g have no positive common scale and the WLOG min_g gamma-tilde_g = 1 cannot be imposed. alpha is a statement about a separable r-block and there is nothing here for it to describe. No regime is claimed for this rule.
