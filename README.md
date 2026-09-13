# LendingClub Credit Risk Modelling & Profit-Optimized Lending Policy

## Project Overview

This project develops an end-to-end **credit risk modelling and lending decision framework** using real LendingClub loan data. Rather than focusing only on predicting loan defaults, the project converts model predictions into a **profit-optimized approval policy** using loan-level financial economics.

The analysis covers **395,754 funded loans** and combines feature engineering, gradient-boosting models, probability calibration, explainability, risk segmentation, temporal validation, and decision-threshold optimization.

## Objective

The primary objective is to identify borrowers with higher default risk and determine an approval threshold that balances **credit risk with lending profitability**.

The project addresses two key lending outcomes:

* Approving a borrower who defaults → potential loss of principal
* Rejecting a borrower who would repay → loss of potential interest income

Therefore, the final decision is optimized around expected portfolio profit rather than classification accuracy alone.

## Dataset

The dataset contains LendingClub loan information obtained through Kaggle.

* **395,754 funded loans**
* **19.61% default rate**
* **28 engineered features**
* **135 features after one-hot encoding**
* Target: `Charged Off` vs `Fully Paid`

The `issue_d` variable is excluded from modelling because it would not be available at application time and could introduce information leakage.

## Methodology

### 1. Data Preprocessing & Feature Engineering

The preprocessing workflow includes:

* Converting loan-term information into numerical variables
* Processing credit-history features
* Binarizing sparse public-record variables
* Converting ZIP-code information into a compact state representation
* Handling missing values using training-data-only transformations
* Creating domain-specific financial ratios

Important engineered variables include:

* `loan_to_income`
* `installment_to_loan`
* `revol_bal_to_income`
* `open_to_total_acc_ratio`
* `annual_installment_to_income`

These features capture the relationship between a borrower's financial obligations and repayment capacity.

### 2. Leakage-Controlled Modelling

A **three-way stratified train/validation/test split** was used.

* Training data → model fitting and preprocessing
* Validation data → early stopping, threshold selection, and calibration
* Test data → final unbiased evaluation

All preprocessing and imputation operations are fitted using training data only.

### 3. Machine Learning Models

The project evaluates:

* **LightGBM**
* **XGBoost**
* **Logistic Regression**
* LendingClub's existing `sub_grade` system
* Approve-everyone baseline

Class imbalance is handled using `scale_pos_weight` rather than synthetic oversampling or data removal.

### 4. Probability Calibration

Because class weighting can distort predicted probabilities, **isotonic regression** is applied to the validation predictions.

This improves the reliability of predicted risk probabilities before they are used for business decisions.

### 5. Profit-Optimized Decision Policy

Instead of using the conventional **0.5 classification threshold**, the approval threshold is selected by maximizing estimated portfolio profit.

The framework considers:

* Interest earned from repaid loans
* Principal losses from defaulted loans
* Assumed **Loss Given Default (LGD)**

The selected threshold is optimized on validation data and then applied to the test set.

### 6. Risk Segmentation

Applicants are ranked by predicted risk and divided into **10 risk deciles**.

This helps evaluate whether predicted risk increases consistently across borrower groups and provides a practical view of portfolio risk concentration.

### 7. Model Validation

Performance is evaluated using:

* ROC-AUC
* PR-AUC
* KS Statistic
* Brier Score
* Precision
* Recall
* Approval Rate

Additional validation is performed across:

* Credit grades
* Loan terms
* Loan purposes
* Different loan vintages

An **out-of-time validation** is also performed to assess temporal generalization.

## Results

The best-performing model is **calibrated LightGBM**.

| Metric                               |     Result |
| ------------------------------------ | ---------: |
| ROC-AUC                              | **0.7233** |
| PR-AUC                               | **0.3839** |
| KS Statistic                         | **0.3263** |
| Brier Score                          | **0.1409** |
| Out-of-Time ROC-AUC                  | **0.7105** |
| Riskiest 3 Deciles Defaults Captured |  **54.6%** |
| Risk Spread (D10/D1)                 |  **11.5×** |

The model achieves a **0.7233 ROC-AUC**, outperforming LendingClub's `sub_grade` benchmark at **0.6867**.

## Business Impact

At an assumed **LGD of 0.6**, the profit-optimized model policy generates:

* Test portfolio profit: **$82.43M**
* Profit uplift: **$2.61M**
* Approval rate: **96.7%**
* Approved-loan default rate: **18.3%**, compared with **19.6%** for the overall portfolio

This demonstrates that model value is not limited to better predictive performance—the predictions can be translated into a financially informed lending strategy.

## Explainability

**SHAP** is used to understand model behaviour and identify the features driving credit risk.

Top features include:

* `int_rate`
* `dti`
* `revol_util`
* `revol_bal_to_income`
* `open_to_total_acc_ratio`
* `installment_to_loan`
* `annual_inc`
* `revol_bal`
* `loan_to_income`
* `annual_installment_to_income`

The engineered financial-ratio variables account for **five of the top ten features**, demonstrating their value in capturing borrower repayment capacity.

## Key Insights

* Risk deciles show a monotonic increase in default rates from the safest to riskiest borrowers.
* The riskiest decile has a **47.60% default rate**, compared with **4.13%** in the safest decile.
* Calibrated LightGBM outperforms both Logistic Regression and LendingClub's existing grade-based benchmark.
* Profit-based threshold selection can produce greater business value than optimizing classification metrics alone.
* Feature engineering around borrower obligations and financial capacity significantly contributes to model explainability.
* Out-of-time validation reveals modest performance degradation, highlighting the importance of temporal robustness.

## Model Development Lessons

Two important modelling issues were identified and corrected during development:

1. **Threshold and calibration mismatch:** A threshold optimized on raw probabilities became invalid after isotonic calibration. The threshold must be re-optimized on the calibrated probability scale.

2. **LightGBM undertraining:** Early stopping was affected by the default evaluation metric under class weighting. The configuration was corrected using `average_precision`, `first_metric_only=True`, and `subsample_freq=1`.

These checks highlight the importance of validating intermediate modelling outputs rather than relying solely on final metrics.

## Limitations

* The dataset contains only funded loans, creating potential **survivorship bias**.
* LGD is an assumed parameter rather than a directly measured value.
* Model performance decreases slightly under out-of-time validation.
* Geographic information may introduce fairness concerns and would require additional testing before real-world deployment.

## Repository Structure

```text
Credit-risk-modelling/
├── notebooks/
│   └── lending_club_default_prediction.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── decile_table.csv
│   ├── segment_analysis.csv
│   ├── lgd_sensitivity.csv
│   ├── results_summary.json
│   └── figures/
├── models/
│   ├── preprocessor.joblib
│   ├── best_model_calibrated.joblib
│   ├── xgboost_model.joblib
│   └── lightgbm_model.joblib
├── PROJECT_DOCUMENTATION.md
├── requirements.txt
└── README.md
```

## Tech Stack

**Python · Pandas · Scikit-learn · LightGBM · XGBoost · SHAP · Matplotlib · Seaborn · Optuna**

## Conclusion

This project demonstrates an end-to-end approach to credit risk analytics, moving from borrower-level data and feature engineering to machine learning, calibrated risk estimation, explainability, and economically optimized lending decisions.

The key takeaway is that an effective credit risk model should not only **rank borrowers by default risk**, but also connect those predictions to **business decisions and portfolio economics**.
