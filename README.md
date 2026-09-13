# Credit Risk Modelling

## Project Overview

This project presents an end-to-end **credit risk modelling and lending decision framework** using real-world LendingClub loan data. Beyond predicting loan defaults, the analysis translates model predictions into a **profit-optimized approval strategy** based on loan-level financial outcomes.

The project covers **395,754 funded loans** and combines feature engineering, gradient-boosting models, probability calibration, model explainability, risk segmentation, temporal validation, and decision-threshold optimization.

## Objective

The primary goal is to identify borrowers with elevated default risk and determine an approval threshold that balances **credit risk with lending profitability**.

The framework considers two key lending outcomes:

* Approving a borrower who defaults → potential loss of principal
* Rejecting a borrower who would repay → missed interest income

Therefore, the final lending decision is optimized around expected portfolio profit rather than classification accuracy alone.

## Dataset

The dataset contains LendingClub loan information sourced through Kaggle.

* **395,754 funded loans**
* **19.61% default rate**
* **28 engineered features**
* **135 features after one-hot encoding**
* Target: `Charged Off` vs `Fully Paid`

The `issue_d` variable is excluded from modelling because it would not be available at application time and could introduce information leakage.

## Methodology

### 1. Data Preprocessing & Feature Engineering

The preprocessing pipeline includes:

* Converting loan-term information into numerical variables
* Processing credit-history features
* Binarizing sparse public-record variables
* Converting ZIP-code information into a compact state representation
* Handling missing values using training-data-only transformations
* Creating domain-specific financial ratios

Key engineered variables include:

* `loan_to_income`
* `installment_to_loan`
* `revol_bal_to_income`
* `open_to_total_acc_ratio`
* `annual_installment_to_income`

These features capture relationships between borrower income, financial obligations, credit utilization, and repayment capacity.

### 2. Leakage-Controlled Modelling

A **three-way stratified train/validation/test split** is used.

* Training data → model fitting and preprocessing
* Validation data → early stopping, calibration, and threshold selection
* Test data → final unbiased evaluation

All preprocessing and imputation steps are fitted using training data only to prevent information leakage.

### 3. Machine Learning Models

The project evaluates:

* **LightGBM**
* **XGBoost**
* **Logistic Regression**
* LendingClub's existing `sub_grade` system
* Approve-everyone baseline

Class imbalance is handled using `scale_pos_weight` rather than synthetic oversampling or removing observations.

### 4. Probability Calibration

Since class weighting can distort predicted probabilities, **isotonic regression** is applied to validation predictions.

This improves the reliability of estimated default probabilities before they are used for lending decisions.

### 5. Profit-Optimized Decision Policy

Instead of applying the conventional **0.5 classification threshold**, the approval threshold is selected by maximizing estimated portfolio profit.

The framework considers:

* Interest earned from repaid loans
* Principal losses from defaulted loans
* Assumed **Loss Given Default (LGD)**

The optimal threshold is determined using validation data and then evaluated on the test set.

### 6. Risk Segmentation

Borrowers are ranked according to predicted default probability and divided into **10 risk deciles**.

This provides a practical view of risk concentration and helps assess whether predicted risk increases consistently across borrower groups.

### 7. Model Validation

Performance is evaluated using:

* ROC-AUC
* PR-AUC
* KS Statistic
* Brier Score
* Precision
* Recall
* Approval Rate

Additional validation is conducted across:

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

The model achieves a **0.7233 ROC-AUC**, outperforming LendingClub's `sub_grade` benchmark of **0.6867**.

## Business Impact

At an assumed **LGD of 0.6**, the profit-optimized model policy generates:

* Test portfolio profit: **$82.43M**
* Profit uplift: **$2.61M**
* Approval rate: **96.7%**
* Approved-loan default rate: **18.3%**, compared with **19.6%** for the overall portfolio

These results demonstrate how credit risk predictions can be translated into a financially informed lending strategy.

## Explainability

**SHAP** is used to interpret model predictions and identify the features contributing most strongly to credit risk.

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

Five of the top ten features are engineered financial ratios, highlighting their importance in capturing borrower repayment capacity.

## Key Insights

* Default rates increase consistently across risk deciles from the safest to riskiest borrowers.
* The riskiest decile has a **47.60% default rate**, compared with **4.13%** in the safest decile.
* Calibrated LightGBM outperforms both Logistic Regression and LendingClub's existing grade-based benchmark.
* Profit-based threshold optimization can create greater business value than optimizing classification metrics alone.
* Financial-ratio features contribute significantly to model interpretability and risk assessment.
* Out-of-time validation shows modest performance degradation, highlighting the importance of temporal robustness.

## Model Development Lessons

Two important modelling issues were identified and addressed:

1. **Threshold and calibration mismatch:** A threshold optimized on raw probabilities became invalid after isotonic calibration. The threshold must therefore be re-optimized using calibrated probabilities.

2. **LightGBM undertraining:** Early stopping was affected by the default evaluation metric under class weighting. This was corrected using `average_precision`, `first_metric_only=True`, and `subsample_freq=1`.

These checks highlight the importance of validating intermediate modelling stages rather than relying solely on final model metrics.

## Limitations

* The dataset contains only funded loans, which may introduce **survivorship bias**.
* LGD is based on an assumed value rather than directly measured recovery data.
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

This project demonstrates an end-to-end approach to credit risk analytics, progressing from borrower-level data and feature engineering to machine learning, probability calibration, explainability, risk segmentation, and profit-driven lending decisions.

The key takeaway is that an effective credit risk model should not only **rank borrowers by default risk**, but also translate those predictions into **data-driven lending decisions that balance portfolio risk and financial returns**.
