### Frozen-representation analysis

- n = 5794, d = 2048, empirical eps = 0.5000
- basis: **raw**, tau = 0.15
- agreement between the two rules: Jaccard r = 0.722, s = 0.564; concordance on the 949 features both label = 1.000

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
- approx. SE of each contrast: 0.0158; conjunction-dominated features: 19

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
| A^T A | 0.8884 | 0.0986 |  |
| A^T B | 0.9920 | 0.1153 |  |
| B^T B | 0.9258 | 0.1100 |  |
| B^T A | 0.9900 | 0.1034 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 0.7166, mu_B = 0.6689, mu = +0.2032
- attractive condition -1 <= mu < min(mu_A, mu_B): True
- d_r = 1063, d_s = 320, d_s >= 2 d_r: False, dim K = 0
- d* = 94.2670 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 1.000

**Margins and the exponent alpha**

- gamma-tilde_maj = 1.0000, gamma-tilde_min = 1.1571 (orientation: standard, ess-inf proxy: 1% quantile)
- r-block separable fraction = 0.9900
- **alpha = 0.8111** -> alpha < 1 (balancing helps); predicted minority exponent max(alpha,1) = 1.0000

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
| A^T A | 0.9481 | 0.1165 |  |
| A^T B | 0.9857 | 0.0897 |  |
| B^T B | 0.9355 | 0.0868 |  |
| B^T A | 0.9911 | 0.1140 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 0.5184, mu_B = 0.4885, mu = +0.2109
- attractive condition -1 <= mu < min(mu_A, mu_B): True
- d_r = 769, d_s = 182, d_s >= 2 d_r: False, dim K = 0
- d* = 74.8263 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 0.999

**Margins and the exponent alpha**

- gamma-tilde_maj = nan, gamma-tilde_min = nan (orientation: mirror, ess-inf proxy: 1% quantile)
- r-block separable fraction = 0.9071
- **alpha = nan** -> alpha >= 1 (geometry dominates); predicted minority exponent max(alpha,1) = nan
