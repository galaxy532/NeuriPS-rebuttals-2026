### Frozen-representation analysis

- n = 5794, d = 2048, empirical eps = 0.5000
- basis: **sae**, tau = 0.15
- SAE: d_hidden = 8192, variance explained = 0.9905, avg active features = 438.7
- agreement between the two rules: Jaccard r = 0.394, s = 0.312; concordance on the 18 features both label = 1.000

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
- approx. SE of each contrast: 0.0158; conjunction-dominated features: 2066

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
| A^T A | 0.8122 | 0.2198 |  |
| A^T B | 0.9617 | 0.5461 |  |
| B^T B | 0.7900 | 0.4412 |  |
| B^T A | 0.8417 | 0.2427 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 0.5905, mu_B = 3.9354, mu = -0.6431
- attractive condition -1 <= mu < min(mu_A, mu_B): True
- d_r = 31, d_s = 16, d_s >= 2 d_r: False, dim K = 0
- d* = 150.7848 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 0.993

**Margins and the exponent alpha**

- gamma-tilde_maj = nan, gamma-tilde_min = nan (orientation: mirror, ess-inf proxy: 1% quantile)
- r-block separable fraction = 0.8940
- **alpha = nan** -> alpha >= 1 (geometry dominates); predicted minority exponent max(alpha,1) = nan

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
| A^T A | 0.8315 | 0.3640 |  |
| A^T B | 0.9496 | 0.7876 |  |
| B^T B | 0.5511 | 0.4586 |  |
| B^T A | 0.6520 | 0.2234 |  |

- 'degenerate' marks products whose M v is negligible, where the angle-based defect is 0/0 and only the scaled column is meaningful.
- mu_A = 0.0712, mu_B = 1.3503, mu = +0.1833
- attractive condition -1 <= mu < min(mu_A, mu_B): False
- d_r = 15, d_s = 5, d_s >= 2 d_r: False, dim K = 0
- d* = 78.1472 (relative 1.0000); sensitivity to the rank cutoff: 1e-06: 1.000, 0.0001: 1.000, 0.001: 1.000, 0.01: 1.000, 0.1: 1.000

**Margins and the exponent alpha**

- gamma-tilde_maj = nan, gamma-tilde_min = nan (orientation: mirror, ess-inf proxy: 1% quantile)
- r-block separable fraction = 0.8550
- **alpha = nan** -> alpha >= 1 (geometry dominates); predicted minority exponent max(alpha,1) = nan
