# Experiment 16: leaf_l2=1.0 validation on Exp 15 losers

**Date:** 2026-04-25
**Config:** Exp-15 defaults + leaf_l2=1.0 (XGBoost reg_lambda match).
**Datasets re-fit:** 20 of 20 loser datasets from Exp 15.

## Headline

- 1/20 are now outright wins (ratio < 1.00).
- 6/20 are within 10% (ratio < 1.10).
- 3/20 are still catastrophic (ratio > 2.00).
- Mean ratio change: **+0.006** (negative = improvement).

## Per-dataset comparison

| dataset | Exp 15 ratio | Exp 16 ratio | Δ | verdict |
|---|---|---|---|---|
| 527_analcatdata_election2000 | 2.355 | 2.364 | +0.009 | still_catastrophic |
| 663_rabe_266 | 2.341 | 2.366 | +0.026 | still_catastrophic |
| 561_cpu | 2.149 | 2.186 | +0.037 | still_catastrophic |
| 659_sleuth_ex1714 | 1.742 | 1.742 | -0.000 | still_clear_loss |
| 1089_USCrime | 1.608 | 1.608 | +0.000 | still_clear_loss |
| 485_analcatdata_vehicle | 1.484 | 1.484 | -0.000 | improved_but_still_loss |
| 230_machine_cpu | 1.479 | 1.426 | -0.052 | improved_but_still_loss |
| 505_tecator | 1.365 | 1.470 | +0.105 | improved_but_still_loss |
| 1096_FacultySalaries | 1.253 | 1.317 | +0.064 | improved_but_still_loss |
| 542_pollution | 1.146 | 1.141 | -0.004 | improved_but_still_loss |
| 666_rmftsa_ladata | 1.140 | 1.141 | +0.002 | improved_but_still_loss |
| 228_elusage | 1.139 | 1.158 | +0.019 | improved_but_still_loss |
| 656_fri_c1_100_5 | 1.112 | 1.116 | +0.004 | improved_but_still_loss |
| 687_sleuth_ex1605 | 1.094 | 1.045 | -0.049 | in_band |
| 591_fri_c1_100_10 | 1.090 | 1.093 | +0.003 | in_band |
| 594_fri_c2_100_5 | 1.066 | 1.045 | -0.021 | in_band |
| 201_pol | 1.025 | 1.020 | -0.005 | in_band |
| 537_houses | 1.009 | 1.002 | -0.007 | in_band |
| 657_fri_c2_250_10 | 1.006 | 0.994 | -0.012 | now_a_win |
| 1030_ERA | 1.002 | 1.003 | +0.001 | in_band |
