# Employee Retention Prediction

An exploratory data analysis (EDA) and baseline machine-learning project using an HR dataset to examine factors associated with employees leaving a company and to train a logistic-regression classifier.

> **Important:** The analysis describes associations in this dataset. It does not establish that any factor causes an employee to leave, and the model should not be used by itself to make employment decisions.

## Project contents

- `Salome_Kungu_HR_Assignment.ipynb` — EDA, visualizations, and logistic-regression training.
- `HR_comma_sep 1.csv` — input dataset. The target column is `left` (`1` = left, `0` = stayed).
- `employee retention 1.png` — salary-level retention chart.
- `employee retention 2.png` — department retention chart.

## Visualizations

### Employee turnover by salary level

![Employee turnover proportion by salary level](<employee retention 1.png>)

### Employee turnover by department

![Employee turnover proportion by department](<employee retention 2.png>)

## Analysis overview

The notebook checks dataset structure and missing values, compares feature summaries by the `left` target, calculates turnover proportions for salary and department groups, and trains a logistic-regression model using selected numeric features plus one-hot-encoded `Department` and `salary` features.

### Dataset observations

The notebook reports 14,999 records and no missing values. About 23.8% of records are labeled as employees who left. In the descriptive comparisons, employees who left had lower average satisfaction, more average monthly hours, and lower promotion rates. Turnover proportions also differ across salary and department groups. These are exploratory associations, not causal findings.

## Run the notebook

The notebook was authored for Google Colab and currently reads the CSV from a Google Drive-specific path. To run it:

1. Open `Salome_Kungu_HR_Assignment.ipynb` in Google Colab or Jupyter.
2. Install the required packages if they are not already available: `pandas`, `numpy`, `matplotlib`, and `scikit-learn`.
3. Update the CSV-loading cell to use the location of `HR_comma_sep 1.csv`. For a local notebook in the repository directory, for example:

   ```python
   df = pd.read_csv("HR_comma_sep 1.csv")
   ```

4. Run the notebook cells in order.

## Potential code and analysis issues

- **Environment-specific file loading:** The notebook mounts Google Drive and reads `/content/drive/MyDrive/HR_comma_sep 1.csv`. This will fail outside the author's Colab/Drive setup unless the path is changed. The Drive mount is unnecessary when using the CSV included in this repository.
- **No model evaluation is implemented:** The notebook trains the classifier, but does not calculate test-set predictions or report accuracy, precision, recall, F1-score, a confusion matrix, or ROC-AUC. Therefore, its predictive performance is unknown; the README should not imply the model has been validated.
- **Non-stratified split:** `train_test_split` does not specify `stratify=y`. With an imbalanced target, class proportions can differ between training and test sets. Use a stratified split for a more representative evaluation.
- **Preprocessing occurs before splitting:** One-hot encoding is applied to the full dataset before the train/test split. This exposes the test set's category vocabulary to preprocessing. Put encoding and the classifier in a scikit-learn `Pipeline`/`ColumnTransformer`, and fit the pipeline only on training data.
- **Class imbalance and metric choice:** Only about 24% of rows are labeled as having left. Accuracy alone could hide poor detection of employees who left. Report per-class precision/recall, F1, a confusion matrix, and ideally ROC-AUC or PR-AUC; consider class weighting or other imbalance strategies only after checking these metrics.
- **Causal language is too strong:** Group averages and turnover proportions show associations, not direct impact or causes. Avoid translating these observational results into causal HR recommendations without further analysis and domain review.
- **Warnings are globally hidden:** `warnings.filterwarnings("ignore")` can mask convergence or deprecation warnings. Remove it or suppress only known, understood warnings, and check model convergence.
- **Reproducibility and maintainability:** The notebook has unused imports (such as `csv` and `linear_model`), hard-coded paths, and saved outputs that may become stale when cells or data change. Remove unused imports and rerun/clear outputs before publishing changes.
- **Fairness and privacy:** This is employee data and predictive use in HR can have significant consequences. Review data provenance, privacy permissions, potential proxy variables, group-level error rates, and applicable employment laws before any real-world use. Keep the model as an educational baseline unless it has been properly validated and governed.

## Suggested next steps

1. Make data loading repository-relative and remove the Colab-only Drive mount (or document it as an optional setup).
2. Build preprocessing and logistic regression in a pipeline, and use a stratified train/test split.
3. Evaluate on held-out data with appropriate classification metrics and compare against a simple baseline.
4. Check calibration, error rates across relevant groups, and robustness before considering any practical application.
5. Record package versions and the exact dataset source to improve reproducibility.
