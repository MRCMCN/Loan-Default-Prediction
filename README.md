# Loan-Default-Prediction

Loan default prediction for credit risk management: a Logistic Regression benchmark against a Gradient Boosting challenger on ~149K mortgage applications, with an explicit analysis of data leakage in the source dataset.

## Project overview

- **Question.** Can loan default be predicted from application data, and how much does a non-linear challenger model add over an interpretable benchmark?
- **Models.** Logistic Regression as the interpretable benchmark, Histogram-based Gradient Boosting as the challenger. Both share the same preprocessing (median imputation for numerical inputs, most-frequent for categorical, one-hot encoding), with standardisation added for the logistic.
- **Partitions.** Stratified 60% train / 30% validation / 10% test — 89,202 / 44,600 / 14,868 applications. The champion model and the operating cutoff are both selected on validation; the test set is used only for the final assessment.
- **Metrics.** AUC, KS statistic and cumulative lift at 5/10/20% depth — the standard toolkit in credit scoring — alongside accuracy and F1.
- **Leakage analysis.** Several inputs are missing almost exclusively for defaulted loans, or contain categories made entirely of defaults. These are identified, quantified, and the material ones removed before modelling.

## Data

The dataset is not included in the repository. Download it from Kaggle and place it as `data/Loan_Default.csv`.

Source: Kaggle dataset "Loan Default Dataset" by M Yasser H (CC0: Public Domain)

- 148,670 loan applications, 34 variables
- Target `Status`: 24.6% default, 75.4% non-default
- Twelve columns are dropped before modelling — five uninformative, seven contaminated — leaving 21 inputs

## Key metrics

Test partition. The last column refits each model on every available input, including the contaminated ones.

| Model | AUC | KS | Accuracy | F1 | Lift @5% | AUC (all inputs) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.730 | 0.352 | 0.784 | 0.366 | 3.17 | 0.863 |
| **Gradient Boosting** (champion) | **0.788** | **0.430** | **0.812** | **0.488** | **3.71** | **1.000** |

## Results

**Data leakage**

- Several fields are populated only after origination, so their *missingness* encodes the outcome: `rate_of_interest` is missing for 36,439 loans, of which 100% defaulted. `Interest_rate_spread` (36,639), `property_value` and `LTV` (15,098 each) show the same perfect pattern; `Upfront_charges` reaches a 92% default rate on 39,642 missing values and `dtir1` 68% on 24,121. One category of `credit_type` (EQUI, 15,298 loans) consists entirely of defaults.
- With these inputs left in, Gradient Boosting reaches AUC, KS, F1 and accuracy all *exactly* equal to 1. That is deterministic separation, not a good model: it is reading an indicator of the outcome rather than a signal about credit risk. This is the single most important finding of the project.
- Two further variables, `age` and `submission_of_application`, share the pattern on 200 observations (0.1% of the sample) and are left in as immaterial. The cutoff for removal is a judgement call — `dtir1` at 68% is not perfect leakage, and dropping it is a conservative choice rather than a mechanical rule.

**Model comparison and operating cutoff**

- On the clean input set, Gradient Boosting wins on every metric in the table above. In Gini terms, the convention in credit scoring, the gap is 0.576 against 0.460 — a 25% relative gain for the challenger. Real, but small enough that the interpretability trade-off is genuine in a regulatory context where explainability is a requirement rather than a preference.
- Train-to-test degradation is small for both models (GB: AUC 0.809 → 0.788), indicating no substantial overfitting.
- At the default 0.5 threshold the champion catches 36% of defaults (1,331 of 3,664). At the KS-optimal cutoff estimated on validation (0.264) it catches 62%, at the cost of flagging 2,188 good applications instead of 461: F1 rises from 0.488 to 0.562, accuracy falls from 0.812 to 0.760.
- The choice between the two is a business decision, not a statistical one: it depends on the cost of a missed default relative to a rejected good customer.

**Drivers**

- Most important inputs by permutation importance, in descending order: `income`, `co-applicant_credit_type`, `Gender`, `loan_amount`, `Neg_ammortization`.
- The logistic coefficients agree in direction: higher income lowers the log-odds of default (−0.388), while negative amortization (+0.169) and lump-sum payment structures (+0.202) raise them.

**Protected attributes**

`Gender` ranks third by permutation importance, and the logistic assigns it explicit weight (`Male` +0.254, `Female` +0.164, `Joint` −0.381). It is retained here because the aim is to model the data as given and to report what the data actually contains. A production scorecard could not use it.

## Conclusion

The headline result is negative and deliberate: a Gradient Boosting model on this dataset scores a *perfect* AUC, and that perfection is the symptom of a broken setup rather than a good model. Diagnosing why is the substance of the project — and once the contaminated inputs are removed, what remains is a realistic AUC of 0.788 and an interpretability trade-off that is genuine rather than rhetorical.
