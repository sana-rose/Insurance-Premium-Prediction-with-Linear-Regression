# Insurance Premium Prediction: Linear Regression

An end-to-end, interpretable regression project that predicts **annual medical insurance premiums** from demographic, lifestyle and health characteristics. The pipeline covers cleaning, EDA, hypothesis testing, assumption checks, model building and comparison against regularised models.

## Business Problem

A health insurance provider wants a transparent model to estimate each customer's annual premium and to understand how factors such as age, BMI, smoking status and chronic conditions relate to the premium. Interpretability matters as much as accuracy, which is why linear regression was chosen: its coefficients read directly as dollars per unit change.

## Dataset

`insurance_premium.csv`: 2,500 records, 12 columns.

| Type | Features |
| --- | --- |
| Demographic | `age`, `sex`, `region` |
| Lifestyle | `smoker`, `exercise_freq` |
| Health | `bmi`, `chronic_conditions`, `blood_pressure`, `cholesterol` |
| Financial / family | `annual_income`, `children` |
| **Target** | `annual_premium` (USD, continuous) |

Missing values in the raw data: `bmi` (6%), `annual_income` (5%), `blood_pressure` (4%), `cholesterol` (4%), `exercise_freq` (3%), `smoker` (2%).

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, Statsmodels, Scikit-learn

## Methodology

1. **Data audit**: shape, dtypes, summary statistics and a missing-value report.
2. **EDA**: target distribution (skewness 1.11), feature histograms, categorical counts, premium by category, and correlation analysis.
3. **Outlier treatment**: IQR-based detection, with values capped (winsorised) at the IQR bounds.
4. **Imputation**: mode for categorical columns, KNN (k=5, distance-weighted) for numeric columns.
5. **Hypothesis testing**: Welch t-tests, one-way ANOVA and a chi-square test.
6. **Encoding**: binary encoding for `sex` and `smoker`, one-hot encoding (drop-first) for `region` and `exercise_freq`.
7. **Split and scaling**: 60% train / 20% validation / 20% test (1,500 / 500 / 500 rows), with `StandardScaler` fitted on training data.
8. **Assumption checks**: VIF, residual plots, Durbin-Watson, Breusch-Pagan and Jarque-Bera.
9. **Modelling**: Linear Regression, then Ridge, Lasso and Elastic Net for comparison.
10. **Validation**: train/validation/test scoring, 5-fold cross-validation, RMSE and MAE, and adjusted R².

## Key Findings

### Hypothesis tests

| Test | Result |
| --- | --- |
| Smoker vs non-smoker (t-test) | Significant (p < 0.001); mean premium $39,914 vs $26,119 |
| Exercise frequency (ANOVA) | Significant (p < 0.001) |
| Sex (t-test) | Not significant (p = 0.91) |
| Region (ANOVA) | Not significant (p = 0.62) |
| Smoker vs region (chi-square) | Independent (p = 0.70) |

### Strongest drivers

Coefficients are on standardised features, so each shows the premium change (in USD) for a one-standard-deviation increase.

| Feature | Coefficient |
| --- | --- |
| `smoker` | +5,542 |
| `age` | +3,461 |
| `bmi` | +1,907 |
| `chronic_conditions` | +1,587 |
| `exercise_freq_regular` | -827 |
| `children` | +643 |

Smoking status is the largest driver, followed by age, BMI and chronic conditions. Sex, region and income have small coefficients.

### Assumption checks

| Check | Result |
| --- | --- |
| Multicollinearity (VIF) | Pass: all VIF < 1.6 |
| Linearity (residuals vs fitted) | Pass |
| Independence (Durbin-Watson = 2.109) | Pass |
| Homoscedasticity (Breusch-Pagan p = 0.534) | Pass |
| Normality of residuals (Jarque-Bera p < 0.001) | Fail: residuals deviate from normality |

### Model performance

| Split / metric | Linear Regression |
| --- | --- |
| Train R² | 0.8835 |
| Validation R² | 0.8483 |
| **Test R²** | **0.8682** |
| Test adjusted R² | 0.8643 |
| 5-fold CV R² | 0.8726 (± 0.0251) |
| Test RMSE | $2,658 |
| Test MAE | $1,878 |

Scores are high and close across splits, so there are no signs of overfitting.

### Model comparison (test set)

| Model | R² | Adjusted R² | RMSE | MAE |
| --- | --- | --- | --- | --- |
| Linear | 0.868 | 0.864 | 2,658 | 1,878 |
| Ridge | 0.868 | 0.864 | 2,658 | 1,878 |
| Lasso | 0.868 | 0.864 | 2,658 | 1,878 |
| Elastic Net | 0.773 | 0.767 | 3,487 | 2,675 |

Ridge and Lasso match plain linear regression, so the model is already stable. Elastic Net scores lower at the default settings (`alpha=1.0`, `l1_ratio=0.5`), most likely because of heavier shrinkage; its `alpha` was not tuned.



## Limitations

- The residuals are not normally distributed, so coefficient significance tests and prediction intervals should be read with caution.
- Outlier capping and imputation were applied to the full dataset before the train/validation/test split, which can leak small amounts of information from the test set into preprocessing.
- The target was also capped at its IQR upper bound ($47,523), so the model has not seen extreme premiums.
- Regularisation strengths were left at default values and not tuned.

## Next Steps

- Add a BMI × smoker interaction term
- Try a log transform of the target to reduce right skew
- Tune regularisation `alpha` with cross-validation
- Benchmark against a tree-based model to measure any non-linear gains
- Move imputation and scaling into a scikit-learn `Pipeline` fitted on training data only
