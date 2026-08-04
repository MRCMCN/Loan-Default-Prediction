# Loan-Default-Prediction

Loan default prediction for credit risk management: a Logistic Regression benchmark against a Gradient Boosting challenger on ~149K mortgage applications, with an explicit analysis of data leakage in the source dataset.

## Project overview

- **Question.** Can loan default be predicted from application data, and how much does a non-linear challenger model add over an interpretable benchmark?
- **Models.** Logistic Regression as the interpretable benchmark, Histogram-based Gradient Boosting as the challenger. Both wrapped in the same preprocessing pipeline (median imputation for numerical inputs, most-frequent for categorical, one-hot encoding).
- **Partitions.** Stratified 60% train / 30% validation / 10% test. The champion model and the operating cutoff are both selected on validation; the test set is used only for the final assessment.
- **Metrics.** AUC, KS statistic and cumulative lift at 5/10/20% depth — the standard toolkit in credit scoring — alongside accuracy and F1.
- **Leakage analysis.** Several inputs are missing almost exclusively for defaulted loans, or contain categories made entirely of defaults. These are identified, quantified, and removed before modelling.

## Data

The dataset is not included in the repository. Download it from Kaggle and place it as `data/Loan_Default.csv`.

Source: Kaggle dataset "Loan Default Dataset" by M Yasser H (CC0: Public Domain)

- 148,670 loan applications, 34 variables
- Target `Status`: 24.6% default, 75.4% non-default

## Key metrics

Test partition, after removing the contaminated variables:

| Model | AUC | KS | Accuracy | F1 | Lift @5% |
|---|---|---|---|---|---|
| Logistic Regression | 0.730 | 0.352 | 0.784 | 0.366 | 3.17 |
| **Gradient Boosting** (champion) | **0.788** | **0.430** | **0.812** | **0.488** | **3.71** |

Effect of leaving the contaminated variables in:

| Model | AUC (clean) | AUC (all inputs) |
|---|---|---|
| Logistic Regression | 0.730 | 0.863 |
| Gradient Boosting | 0.788 | **1.000** |

## Results

**Data leakage**

- Several fields are populated only after origination, so their *missingness* encodes the outcome: `rate_of_interest` is missing for 36,439 loans, of which 100% defaulted. The same pattern holds for `Interest_rate_spread`, `property_value` and `LTV`.
- One category of `credit_type` (EQUI, 15,298 loans) consists entirely of defaults.
- With these inputs left in, Gradient Boosting reaches a perfect AUC of 1.000 — it is not learning credit risk, it is learning which applications were already known to have failed. This is the single most important finding of the project.

**Model comparison**

- On the clean input set, Gradient Boosting outperforms the Logistic Regression benchmark on every metric (AUC 0.788 vs 0.730, KS 0.430 vs 0.352).
- The gap is meaningful but not dramatic: the interpretable benchmark retains most of the signal, which matters in a regulatory context where explainability is a requirement rather than a preference.
- Train-to-test degradation is small for both models (GB: AUC 0.809 → 0.788), indicating no substantial overfitting.

**Operating cutoff**

- At the default 0.5 threshold the champion catches 36% of defaults.
- At the KS-optimal cutoff estimated on validation (0.276) it catches 62%, at the cost of flagging more good applications. F1 rises from 0.488 to 0.562.
- The choice between the two is a business decision, not a statistical one: it depends on the cost of a missed default relative to a rejected good customer.

**Drivers**

- Most important inputs by permutation importance: `Neg_ammortization`, `loan_amount`, `co-applicant_credit_type`, `income`.
- The logistic coefficients agree in direction: higher income lowers the log-odds of default, negative amortization and lump-sum payment structures raise them.

## Conclusion

- The headline result is negative and deliberate: a Gradient Boosting model on this dataset scores a *perfect* AUC, and that perfection is the symptom of a broken setup rather than a good model. Diagnosing why is the substance of the project.
- Once the contaminated inputs are removed, performance falls to a realistic AUC of 0.788 — consistent with what application-only data can support.
- Gradient Boosting beats the Logistic Regression benchmark, but the benchmark stays close enough that the interpretability trade-off is a genuine one.

**Note.** The dataset includes `Gender`, which ranks among the most important predictors. It is kept here because the aim is to model the data as given.
