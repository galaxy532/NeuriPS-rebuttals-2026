
### Where is the signal?  basis=sae  n=5794  d=8192  tau=0.05

SAE: L0 = 419.0 of 8192, var explained = 0.9907, dead = 0, near-constant = 0, standardised = True

FULL Phi: separable = True, frac_correct = 1.0000, held-out acc = 0.6473
  (held-out acc is the honest number: in-sample separability with chance held-out accuracy would mean memorisation)

How concentrated is the signal (coordinates ranked by |v_full|):
| top m | separable | frac_correct | held-out acc |
|---|---|---|---|
| 10 | False | 0.9253 | 0.8758 |
| 40 | False | 0.8956 | 0.8440 |
| 160 | False | 0.8692 | 0.9023 |
| 640 | True | 0.9971 | 0.8961 |

-> smallest separating m on this ladder: 640
   If 640 is far above the number of features an identification rule keeps, explanation (ii) holds: the block is too small to separate no matter which columns it picks.

--- rule: two-concept  (n_r=615, n_s=176, weak=7401) ---
| block | cols | mass share | enrichment | separable | frac_correct | held-out |
|---|---|---|---|---|---|---|
| r | 615 | 0.426 | 5.68 | False | 0.8823 | 0.7823 |
| s | 176 | 0.060 | 2.79 | False | 0.5761 | 0.5390 |
| weak | 7401 | 0.514 | 0.57 | True | 1.0000 | 0.4855 |
- enrichment = mass share / column share. 1.00 means the block holds exactly the share of v_full its size would predict, i.e. it is not special.
- of the top 640 coordinates of v_full, this rule calls 219 r, 14 s, 407 weak (34.2% labelled r)
  -> a low r-fraction here is direct evidence for explanation (i): the signal sits where the rule is not looking.

--- rule: sign-flip  (n_r=196, n_s=41, weak=7955) ---
| block | cols | mass share | enrichment | separable | frac_correct | held-out |
|---|---|---|---|---|---|---|
| r | 196 | 0.365 | 15.24 | False | 0.8757 | 0.8240 |
| s | 41 | 0.076 | 15.28 | False | 0.6838 | 0.6280 |
| weak | 7955 | 0.559 | 0.58 | True | 1.0000 | 0.5003 |
- enrichment = mass share / column share. 1.00 means the block holds exactly the share of v_full its size would predict, i.e. it is not special.
- of the top 640 coordinates of v_full, this rule calls 132 r, 8 s, 500 weak (20.6% labelled r)
  -> a low r-fraction here is direct evidence for explanation (i): the signal sits where the rule is not looking.
