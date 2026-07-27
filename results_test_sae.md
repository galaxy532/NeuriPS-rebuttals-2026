### Frozen-representation analysis

- n = 5794, d = 2048, empirical eps = 0.5000
- basis: **sae**, tau = 0.15
- SAE: d_hidden = 8192, variance explained = 0.9881, avg active features = 3806.0
- agreement between the two rules: Jaccard r = 0.404, s = 0.347; concordance on the 1351 features both label = 1.000

**Threshold sensitivity (sign-flip rule)**

| tau | n_r | n_s |
|---|---|---|
| 0.05 | 2850 | 2067 |
| 0.1 | 1583 | 1078 |
| 0.15 | 834 | 518 |
| 0.2 | 446 | 218 |
| 0.3 | 168 | 21 |
| 0.4 | 74 | 4 |

---

## Rule: two-concept

- split: n_r = 2060, n_s = 1491, weak = 4641
- (y, place) cell sizes: y=+1,place=+1: 642, y=+1,place=-1: 642, y=-1,place=+1: 2255, y=-1,place=-1: 2255  (smallest 642)
- approx. SE of each contrast: 0.0158; conjunction-dominated features: 216

**Within-(y,g)-cell coupling, Phi_r -> Phi_s (cross-validated R^2, block-permutation null)**

p-value floor with 200 permutations: 0.0050. Cells below the minimum are refused, not reported. Refused: 0; reported but unreliable: 0.

| y | g | n | R2_cv | null mean | null q95 | p | note |
|---|---|---|---|---|---|---|---|
| -1 | 0 | 2255 | -0.164 | -3.7906 | -3.7459 | 0.0050 |  |
| -1 | 1 | 2255 | -0.176 | -3.7057 | -3.6581 | 0.0050 |  |
| +1 | 0 | 642 | 0.524 | -0.7604 | -0.7301 | 0.0050 |  |
| +1 | 1 | 642 | 0.500 | -0.7375 | -0.7160 | 0.0050 |  |

- pooled (marginal) R2 = 0.519  vs mean within-cell R2 = 0.171
- The pooled figure is inflated by label-mediation; the within-cell figure conditions on (y, g) and is the one that speaks to feature-mediation.

**Distance from the isotropic regime**

| product | defect (angle) | defect (scaled) | degenerate |
|---|---|---|---|
| A^T A | 0.8879 | 0.0704 |  |
| A^T B | 0.9986 | 0.0904 |  |
| B^T B | 0.9357 | 0.0848 |  |
| B^T A | 0.9951 | 0.0489 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 2.2113, mu_B = 2.4330, mu = +0.2900
- attractive condition -1 <= mu < min(mu_A, mu_B): True
- d_r = 2060, d_s = 1491, d_s >= 2 d_r: False, dim K = 0
- d* = 163.3210 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 0.961

**Margins and the exponent alpha**

- gamma-tilde_maj = 1.0000, gamma-tilde_min = 1.0000 (orientation: mirror, ess-inf proxy: 1% quantile)
- r-block separable fraction = 0.9900
- **alpha = 0.3758** -> alpha < 1 (balancing helps); predicted minority exponent max(alpha,1) = 1.0000

---

## Rule: sign-flip

- split: n_r = 834, n_s = 518, weak = 6840

**Within-(y,g)-cell coupling, Phi_r -> Phi_s (cross-validated R^2, block-permutation null)**

p-value floor with 200 permutations: 0.0050. Cells below the minimum are refused, not reported. Refused: 0; reported but unreliable: 0.

| y | g | n | R2_cv | null mean | null q95 | p | note |
|---|---|---|---|---|---|---|---|
| -1 | 0 | 2255 | 0.416 | -0.8438 | -0.8314 | 0.0050 |  |
| -1 | 1 | 2255 | 0.437 | -0.8338 | -0.8112 | 0.0050 |  |
| +1 | 0 | 642 | 0.303 | -1.7294 | -1.6365 | 0.0050 |  |
| +1 | 1 | 642 | 0.211 | -1.7429 | -1.6838 | 0.0050 |  |

- pooled (marginal) R2 = 0.626  vs mean within-cell R2 = 0.342
- The pooled figure is inflated by label-mediation; the within-cell figure conditions on (y, g) and is the one that speaks to feature-mediation.

**Distance from the isotropic regime**

| product | defect (angle) | defect (scaled) | degenerate |
|---|---|---|---|
| A^T A | 0.8892 | 0.0579 |  |
| A^T B | 0.9963 | 0.1076 |  |
| B^T B | 0.9500 | 0.0787 |  |
| B^T A | 0.9810 | 0.0472 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 0.6162, mu_B = 0.9370, mu = +0.2504
- attractive condition -1 <= mu < min(mu_A, mu_B): True
- d_r = 834, d_s = 518, d_s >= 2 d_r: False, dim K = 0
- d* = 89.5349 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 0.946

**Margins and the exponent alpha**

- gamma-tilde_maj = 1.0000, gamma-tilde_min = 1.0000 (orientation: standard, ess-inf proxy: 1% quantile)
- r-block separable fraction = 0.9900
- **alpha = 0.7736** -> alpha < 1 (balancing helps); predicted minority exponent max(alpha,1) = 1.0000
