
### Separability probe  basis=raw  n=5794  d=2048  tau=0.15

Tuned L2 logistic probe WITH intercept, C chosen by inner CV on training rows only. Splits are stratified on (y, g); mean +/- sd over seeds.

| block | cols | median C | train acc | **held-out** | worst group | held-out, C=1e6 no-intercept SVM |
|---|---|---|---|---|---|---|
| full | 2048 | 0.55 | 0.9998 | **0.9434** +/- 0.0035 | 0.8676 | 0.8875 |
| two-concept:r | 1063 | 0.1 | 0.9957 | **0.9427** +/- 0.0017 | 0.8489 | 0.8746 |
| two-concept:s | 320 | 0.01 | 0.9598 | **0.9382** +/- 0.0041 | 0.8193 | 0.7593 |
| two-concept:weak | 665 | 0.055 | 0.9798 | **0.9396** +/- 0.0024 | 0.8567 | 0.7919 |
| sign-flip:r | 769 | 0.055 | 0.9798 | **0.9420** +/- 0.0003 | 0.8598 | 0.8608 |
| sign-flip:s | 182 | 0.1 | 0.9551 | **0.9372** +/- 0.0007 | 0.7726 | 0.7528 |
| sign-flip:weak | 1097 | 0.055 | 0.9850 | **0.9396** +/- 0.0003 | 0.8707 | 0.8383 |

- the last column is the estimator the rest of the pipeline uses. Where it sits far below the tuned probe, the pipeline's separability numbers reflect the solver, not the representation.

**Learning curve on the full representation** (evaluation set held fixed; C retuned at each size)

| train n | C | train acc | held-out acc |
|---|---|---|---|
| 289 | 0.001 | 0.9758 | 0.8696 |
| 724 | 0.01 | 0.9862 | 0.8968 |
| 1448 | 0.1 | 1.0000 | 0.9241 |
| 2172 | 0.01 | 0.9820 | 0.9327 |
| 2896 | 100 | 1.0000 | 0.9365 |

- accuracy gained over the last doubling of data: +0.0038
- a curve that has flattened well below 1.0 means the ceiling is a property of the representation, so r-separability fails in the population and alpha is not estimable on this data. A curve still climbing means we are sample-limited and the question is open.
