
### Where is the signal?  basis=sae  n=5794  d=8192  tau=0.05

SAE: L0 = 438.7 of 8192, var explained = 0.9905, dead = 0, near-constant = 0, standardised = True

FULL Phi: separable = True, frac_correct = 1.0000, held-out acc = 0.6142
  (held-out acc is the honest number: in-sample separability with chance held-out accuracy would mean memorisation)

How concentrated is the signal (coordinates ranked by |v_full|):
| top m | separable | frac_correct | held-out acc |
|---|---|---|---|
| 10 | False | 0.8240 | 0.8813 |
| 40 | False | 0.8031 | 0.8813 |
| 160 | False | 0.8747 | 0.8433 |
| 640 | False | 0.9893 | 0.8761 |

-> smallest separating m on this ladder: None
   No rung separated. The signal is not concentrated in any small coordinate set, so explanation (ii) holds and no per-feature rule can recover it.

--- rule: two-concept  (n_r=363, n_s=181, weak=7648) ---
| block | cols | mass share | enrichment | separable | frac_correct | held-out |
|---|---|---|---|---|---|---|
| r | 363 | 0.388 | 8.76 | False | 0.8447 | 0.8264 |
| s | 181 | 0.026 | 1.19 | False | 0.5378 | 0.5138 |
| weak | 7648 | 0.586 | 0.63 | True | 1.0000 | 0.4990 |
- enrichment = mass share / column share. 1.00 means the block holds exactly the share of v_full its size would predict, i.e. it is not special.
- of the top 640 coordinates of v_full, this rule calls 175 r, 26 s, 439 weak (27.3% labelled r)
  -> a low r-fraction here is direct evidence for explanation (i): the signal sits where the rule is not looking.

--- rule: sign-flip  (n_r=108, n_s=47, weak=8037) ---
| block | cols | mass share | enrichment | separable | frac_correct | held-out |
|---|---|---|---|---|---|---|
| r | 108 | 0.404 | 30.67 | False | 0.8654 | 0.8696 |
| s | 47 | 0.028 | 4.84 | False | 0.5383 | 0.5618 |
| weak | 8037 | 0.568 | 0.58 | True | 1.0000 | 0.4872 |
- enrichment = mass share / column share. 1.00 means the block holds exactly the share of v_full its size would predict, i.e. it is not special.
- of the top 640 coordinates of v_full, this rule calls 80 r, 12 s, 548 weak (12.5% labelled r)
  -> a low r-fraction here is direct evidence for explanation (i): the signal sits where the rule is not looking.
