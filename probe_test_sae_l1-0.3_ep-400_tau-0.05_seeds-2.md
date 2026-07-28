
### Separability probe  basis=sae  n=5794  d=8192  tau=0.05

SAE: L0 = 438.7 of 8192, var explained = 0.9905

Tuned L2 logistic probe WITH intercept, C chosen by inner CV on training rows only. Splits are stratified on (y, g); mean +/- sd over seeds.

| block | cols | median C | train acc | **held-out** | worst group | held-out, C=1e6 no-intercept SVM |
|---|---|---|---|---|---|---|
| full | 8192 | 0.0001 | 0.9824 | **0.7947** +/- 0.0048 | 0.7097 | 0.6203 |
| two-concept:r | 363 | 0.01 | 0.9656 | **0.9362** +/- 0.0017 | 0.7773 | 0.8314 |
| two-concept:s | 181 | 0.001 | 0.7887 | **0.7717** +/- 0.0002 | 0.0343 | 0.5307 |
| two-concept:weak | 7648 | 0.0505 | 1.0000 | **0.6950** +/- 0.0014 | 0.6392 | 0.5038 |
| sign-flip:r | 108 | 0.01 | 0.9637 | **0.9474** +/- 0.0009 | 0.8271 | 0.8513 |
| sign-flip:s | 47 | 10 | 0.8177 | **0.8071** +/- 0.0003 | 0.1371 | 0.5757 |
| sign-flip:weak | 8037 | 0.0055 | 1.0000 | **0.6687** +/- 0.0066 | 0.5847 | 0.4950 |

- the last column is the estimator the rest of the pipeline uses. Where it sits far below the tuned probe, the pipeline's separability numbers reflect the solver, not the representation.

**Learning curve on the full representation** (evaluation set held fixed; C retuned at each size)

| train n | C | train acc | held-out acc |
|---|---|---|---|
| 289 | 0.0001 | 1.0000 | 0.6691 |
| 724 | 0.001 | 1.0000 | 0.7312 |
| 1448 | 0.0001 | 0.9945 | 0.7671 |
| 2172 | 0.0001 | 0.9894 | 0.7878 |
| 2896 | 0.0001 | 0.9834 | 0.7995 |

- accuracy gained over the last doubling of data: +0.0117
- a curve that has flattened well below 1.0 means the ceiling is a property of the representation, so r-separability fails in the population and alpha is not estimable on this data. A curve still climbing means we are sample-limited and the question is open.
