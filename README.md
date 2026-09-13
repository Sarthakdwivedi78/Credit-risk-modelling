# LendingClub Loan Default Prediction & Profit-Optimized Lending Policy

Predicting loan default on **395,754 real LendingClub loans** and converting that prediction into a profit-maximizing approval policy — not just a probability score, but an actual lending decision backed by loan-level economics.

A modernized, extended rewrite of [*Lending Club Loan Defaulters Prediction*](https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction).

---

## Table of contents

- [Problem](#problem)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Beating the incumbent: LendingClub's own grades](#beating-the-incumbent-lendingclubs-own-grades)
- [Business impact: a profit-optimized policy](#business-impact-a-profit-optimized-policy)
- [Sensitivity analysis](#sensitivity-analysis)
- [Explainability](#explainability)
- [Engineering notes: bugs found and fixed](#engineering-notes-bugs-found-and-fixed)
- [Limitations](#limitations)
- [Repository structure](#repository-structure)
- [How to run](#how-to-run)
- [Tech stack](#tech-stack)

---

## Problem

A lender deciding whether to approve a loan faces two asymmetric errors:

- **Approve a borrower who defaults** → lose most of the principal
- **Reject a borrower who would have repaid** → lose the interest margin

Because these costs differ by an order of magnitude, the interesting question isn't just "how accurate is the model" but **"where should the approval line be drawn, and is that line more profitable than what's already in use?"** This project treats both questions as first-class, using only information available at application time.

## Dataset

- **395,754 funded loans** (396,030 raw rows, 276 dropped for residual missing values), from LendingClub via Kaggle
- **19.61% default rate** (`Charged Off` vs `Fully Paid`)
- 28 features after engineering, 135 after one-hot encoding
- `issue_d` (loan issue date) is deliberately excluded as a model feature — it isn't known at application time and would leak information

## Approach

**Feature engineering.** Parsed loan term and credit-history start into numerics, binarized sparse public-record counts, and replaced raw ZIP codes — which the original notebook one-hot encoded into hundreds of sparse columns — with a compact two-letter `state` field. Added domain ratio features (`loan_to_income`, `installment_to_loan`, `revol_bal_to_income`, `open_to_total_acc_ratio`, `annual_installment_to_income`) since tree models can't easily synthesize ratios from raw columns, and "obligation relative to capacity" is the core question in underwriting. Several of these ratios land in the **top 10 most important features** (see [Explainability](#explainability)).

**Leakage control.** Three-way stratified train / validation / test split. The preprocessing pipeline and the `mort_acc` imputation lookup are fit on training data only. Validation drives early stopping, threshold selection, and calibration. The test set is scored exactly once.

**Models.** XGBoost and LightGBM, with class imbalance handled via `scale_pos_weight` rather than resampling (no synthetic rows, no discarded data). Benchmarked against three baselines: approve-everyone, **LendingClub's own `sub_grade`** (the incumbent system already in production), and logistic regression.

**Calibration.** `scale_pos_weight` deliberately inflates predicted probabilities to counteract class imbalance — useful for ranking applicants, invalid for reading the output as a real probability of default. Isotonic regression fitted on the validation set corrected the scale.

**Decision policy.** Rather than defaulting to a 0.5 cut-off, the operating threshold was chosen by maximizing **modeled portfolio profit** — interest earned on repaid loans minus principal lost on defaults (at an assumed loss-given-default, LGD) — swept on validation and applied once to test.

**Validation.** Beyond the random test split, the model was also validated **out-of-time** (trained on earlier loan vintages, tested on later ones) and broken out by **grade, term, and purpose** to check where it's weak.

## Results

| Metric | Value |
|---|---|
| **Best model** | LightGBM (calibrated) |
| ROC-AUC | **0.7233** |
| PR-AUC | 0.3839 (1.96× the 19.6% base rate) |
| KS statistic | **0.3263** (above the 0.30 "usable scorecard" convention) |
| Brier score, raw → calibrated | 0.2084 → **0.1409** |
| Decile risk spread (D10 / D1) | **11.5×** |
| Defaults captured in riskiest 3 deciles | **54.6%** |
| Out-of-time ROC-AUC | 0.7105 (vs 0.7228 random split — a modest, honest 0.012 degradation) |

**Full model comparison** (test set, LightGBM/XGBoost raw rows at a neutral 0.5 threshold; the calibrated row is the actual deployed policy at its profit-optimal threshold):

| Model | ROC-AUC | PR-AUC | KS | Brier | Precision | Recall | Approval rate |
|---|---|---|---|---|---|---|---|
| LightGBM | 0.7234 | 0.3919 | 0.3255 | 0.2084 | 0.326 | 0.655 | 60.7% |
| **LightGBM (calibrated) — deployed** | **0.7233** | 0.3839 | **0.3263** | **0.1409** | 0.565 | 0.097 | 96.7% |
| XGBoost | 0.7228 | 0.3911 | 0.3246 | 0.2097 | 0.324 | 0.655 | 60.4% |
| Logistic regression | 0.7134 | 0.3707 | 0.3122 | 0.2158 | 0.314 | 0.662 | 58.7% |
| LC `sub_grade` (incumbent) | 0.6867 | 0.3266 | 0.2764 | 0.1468 | 0.385 | 0.002 | 99.9% |
| Approve everyone | 0.5000 | 0.1961 | 0.0113 | 0.1577 | — | — | 100% |

The deployed policy's low recall/high approval rate isn't a bug — it reflects the profit-optimal answer under the assumed loss economics (see [Business impact](#business-impact-a-profit-optimized-policy)), not a threshold picked for classification metrics.

**Risk deciles** — applicants sorted by predicted risk, split into ten equal buckets. A healthy model shows the default rate climbing monotonically from D1 to D10:

| Decile | n | Default rate | Lift vs base rate |
|---|---|---|---|
| D1 (safest) | 5,937 | 4.13% | 0.21× |
| D2 | 5,936 | 6.59% | 0.34× |
| D3 | 5,936 | 9.79% | 0.50× |
| D4 | 5,937 | 12.63% | 0.64× |
| D5 | 5,936 | 15.33% | 0.78× |
| D6 | 5,936 | 17.99% | 0.92× |
| D7 | 5,937 | 22.64% | 1.15× |
| D8 | 5,936 | 26.20% | 1.34× |
| D9 | 5,936 | 33.22% | 1.69× |
| D10 (riskiest) | 5,937 | **47.60%** | **2.43×** |

Monotonic across all ten buckets with no inversions — the strongest single piece of evidence that the model's risk ordering is trustworthy.

## Beating the incumbent: LendingClub's own grades

The comparison that actually matters commercially isn't against a synthetic baseline — it's against the grading system LendingClub already uses to price loans. The model beats it by a clear margin:

| Model | ROC-AUC |
|---|---|
| LendingClub `sub_grade` | 0.6867 |
| Logistic regression | 0.7134 |
| **LightGBM (calibrated)** | **0.7233** (+0.037 over sub_grade) |

**Segment breakdown** (AUC within each slice — checks the model isn't just re-deriving grade):

| Segment | n | Default rate | ROC-AUC |
|---|---|---|---|
| Grade A | 9,666 | 6.2% | 0.661 |
| Grade B | 17,333 | 12.8% | 0.630 (weakest segment) |
| Grade C | 15,965 | 21.1% | 0.639 |
| Grade D | 9,620 | 29.4% | 0.646 |
| Grade E | 4,566 | 36.1% | 0.646 |
| Grade F | 1,747 | 42.6% | 0.638 |
| Term = 36 months | 45,258 | 15.8% | 0.704 |
| Term = 60 months | 14,106 | 31.9% | 0.678 |
| Purpose = major_purchase | 1,288 | 16.2% | 0.747 (strongest segment) |
| Purpose = credit_card | 12,683 | 16.5% | 0.725 |
| Purpose = debt_consolidation | 35,080 | 20.9% | 0.721 |
| Purpose = small_business | 830 | 29.5% | 0.675 |

AUC *within* a single grade band is naturally lower than the overall 0.72 — this is expected (restriction of range: applicants inside one grade are already pre-selected to be similar in risk), not a sign the model is failing there.

## Business impact: a profit-optimized policy

Rather than an abstract cost ratio, the decision threshold was chosen from actual loan economics:

- **Approve, loan repays** → profit = total scheduled payments − principal
- **Approve, loan defaults** → loss = principal × loss-given-default (LGD)
- **Reject** → $0

At **LGD = 0.6** (a moderate assumption — see [Sensitivity analysis](#sensitivity-analysis) for the full range):

| Policy | Test-portfolio profit (59,364 loans) |
|---|---|
| Approve everyone | $79,823,240 |
| Best possible `sub_grade`-based cutoff | $79,823,240 *(identical — see note below)* |
| **Model policy @ threshold 0.480** | **$82,429,549** |

**Uplift: $2,606,309 over both approve-everyone and the best sub_grade policy (~$44 per loan), while approving 96.6% of applicants** and cutting the default rate among approved loans from 19.6% to 18.3%.

> **Why "best sub_grade policy" exactly equals "approve everyone":** this is a real finding, not a bug. At LGD = 0.6, no fixed sub-grade cutoff produces more expected profit than approving everyone — LendingClub's interest-rate pricing already compensates for grade-level risk well enough that grade alone isn't a profitable rejection signal. The gradient-boosted model *does* beat that line, meaning its value comes from finding **risk that grade-based pricing hasn't already priced in** (adverse selection within grade) — a more sophisticated result than simply "higher AUC."

## Sensitivity analysis

Every dollar figure above scales with the LGD assumption, so the policy was re-optimized across LGD = 0.3–0.9 to check the $2.6M figure isn't an artifact of one convenient number:

| LGD | Approval rate | Profit uplift vs approve-all |
|---|---|---|
| 0.3 | 100.0% | ~$0.1M |
| 0.4 | 99.5% | ~$0.3M |
| 0.5 | 97.2% | ~$0.8M |
| **0.6 (headline)** | **96.7%** | **$2.6M** |
| 0.7 | 93.9% | ~$5.4M |
| 0.8 | 93.0% | ~$8.3M |
| 0.9 | 86.0% | ~$16.1M |

*(Values above LGD = 0.6 read from the sensitivity chart to 1 decimal place — swap in the exact figures from `reports/lgd_sensitivity.csv` before citing them precisely.)*

Both curves move monotonically in the expected direction: the more expensive a default is, the more valuable it becomes to screen for one, and that value compounds rather than growing linearly. The model adds value across the *entire* plausible range, not just at the specific LGD chosen for the headline number — and LendingClub's own historically low recovery rates on charged-off loans suggest the realistic LGD may sit closer to the 0.7–0.9 end of this range than to 0.6.

## Explainability

**Top features by gain** (LightGBM): `int_rate`, `dti`, `revol_util`, `revol_bal_to_income`, `open_to_total_acc_ratio`, `installment_to_loan`, `annual_inc`, `revol_bal`, `loan_to_income`, `annual_installment_to_income`.

Five of the top ten are the **engineered ratio features** — direct evidence they were worth building rather than relying on raw columns alone.

**SHAP direction** confirms intuitive, defensible relationships: higher interest rate, higher DTI, higher revolving utilization, and higher loan-to-income all push predicted risk up; a 36-month term (vs 60) pushes it down. No counter-intuitive or suspicious feature effects were found.

*(See `reports/figures/feature_importance.png` and `reports/figures/shap_summary.png`.)*

## Engineering notes: bugs found and fixed

Two real defects were caught during development, each of which had silently produced misleading results before the fix:

1. **Threshold invalidated by calibration.** An operating threshold tuned on raw (uncalibrated) probabilities was reused after isotonic calibration rescaled the entire distribution, collapsing recall from 0.75 to 0.15 in an earlier iteration. **Fix:** a threshold is only valid on the probability scale it was optimized for — recompute it after any rescaling. The same bug pattern resurfaced in a comparison table later and was fixed by scoring each probability scale at its own appropriate threshold.
2. **LightGBM was silently undertrained**, stopping after only 6 boosting iterations in an early run (vs 917 in the final one). Three compounding causes: early stopping watched the default `binary_logloss`, which `scale_pos_weight` distorts; LightGBM kept watching that default metric even when another was supplied; and `subsample=0.8` had **no effect at all**, because LightGBM silently ignores `subsample` unless `subsample_freq > 0`. **Fix:** `eval_metric="average_precision"` + `first_metric_only=True` + `subsample_freq=1`.

Both are documented here because they're easy mistakes to make again, and the before/after numbers (recall 0.75→0.15, LightGBM iterations 6→917) are a useful illustration of why validating intermediate outputs — not just final metrics — matters.

## Limitations

- **Survivorship bias.** The dataset contains only *funded* loans. Applicants LendingClub rejected outright never appear, so the model learns "who defaults among the already-approved," not "who defaults among all applicants." This is the single biggest caveat for any real deployment.
- **LGD is an assumption, not a measurement.** Every dollar figure in this project scales linearly with it. The *shape* of the profit curve (monotonic, model beats baselines everywhere) is robust; the specific dollar amount at LGD = 0.6 is not.
- **Modest temporal degradation.** Out-of-time ROC-AUC (0.7105) is meaningfully below the random-split figure (0.7228), meaning some of the random-split performance reflects vintage-specific patterns that don't fully generalize forward.
- **Fair lending.** `state` is used as a feature, and geography can proxy for protected characteristics. Any real lending use would require disparate-impact testing, and this feature would likely need to be dropped or constrained.

## Repository structure

```
lending-club-risk/
├── notebooks/
│   └── lending_club_default_prediction.ipynb   # full analysis, self-contained
├── reports/
│   ├── model_comparison.csv
│   ├── decile_table.csv
│   ├── segment_analysis.csv
│   ├── lgd_sensitivity.csv
│   ├── results_summary.json
│   └── figures/
│       ├── roc_pr_comparison.png
│       ├── calibration.png
│       ├── profit_curve.png
│       ├── risk_deciles.png
│       ├── feature_importance.png
│       ├── shap_summary.png
│       ├── shap_dependence.png
│       └── lgd_sensitivity.png
├── models/
│   ├── preprocessor.joblib
│   ├── best_model_calibrated.joblib
│   ├── xgboost_model.joblib
│   └── lightgbm_model.joblib
├── PROJECT_DOCUMENTATION.md
└── README.md
```

## How to run

1. Download `lending_club_loan_two.csv` from the [source Kaggle notebook](https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction) (or any LendingClub dataset containing that file).
2. Open `notebooks/lending_club_default_prediction.ipynb` on Kaggle (it auto-detects the file under `/kaggle/input/`) or locally with the CSV in the working directory.
3. Run all cells. Everything is self-contained — no external package required.
4. Outputs land in `reports/` and `models/`; on Kaggle, click **Save Version** to make them downloadable from the Output tab.

## Tech stack

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · matplotlib / seaborn · (optional: Optuna for hyperparameter tuning)

---

*Based on the exploratory analysis and problem framing from [faressayah's LendingClub notebook](https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction) on Kaggle. Data originally from LendingClub.*
