# Beyond Predictive Accuracy: GLM vs Tree Ensembles for Medical-Cost Prediction

Code and data for the IAMST 2027 paper *"Beyond Predictive Accuracy: An Actuarial Benchmark of Generalized Linear and Tree-Ensemble Models for Medical-Cost Prediction."*

The study compares OLS, log-link Gamma GLMs, a decision tree, Random Forest and gradient boosting for predicting recorded annual medical charges, with a held-out test set, nested cross-validation, exact Shapley values and interaction indices, explanation-stability checks, and a subgroup error audit by sex, region and smoking status.

## Files

| File | Contents |
|---|---|
| `analysis.py` | Full analysis: tuning, evaluation, Shapley explanations, subgroup audit, figures |
| `insurance.csv` | Medical Cost Personal data set (1,338 records) |
| `results.json` | All numerical results reported in the paper |
| `nested_folds.csv` | Fold-level results of the nested cross-validation |
| `results_xgboost.json` | Robustness check: the same analysis run with `--xgboost` |

## How to reproduce

Requires Python 3.10+. The published results were produced with scikit-learn 1.9.1.

```
pip install numpy pandas scikit-learn matplotlib
python analysis.py insurance.csv
```

The run takes about 15–30 minutes and writes `results.json` and four figures (`fig1_fit.png` to `fig4_local.png`) to an `out/` folder. Random seeds are fixed, so results should match the paper.

The script always uses scikit-learn's gradient boosting unless you add `--xgboost` (an optional variant whose results differ from the paper). Exact duplicate records are kept in the same fold in every cross-validation split. The nested cross-validation cache in `out/` is keyed to the data hash, configuration and scikit-learn version, so it is never reused for different data or settings.

## Data source and licence

`insurance.csv` is the Medical Cost Personal data set, distributed on Kaggle at https://www.kaggle.com/datasets/mirichoi0218/insurance and originally from the companion repository of Lantz, B. (2013), *Machine Learning with R*, Packt Publishing (https://github.com/stedy/Machine-Learning-with-R-datasets).

The data set is released under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/), with its contents under the [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/). It is redistributed here unchanged, under the same licence.

SHA-256 of the file used (Unix line endings, no trailing newline): `505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281`. A copy converted to Windows line endings has a different raw hash; the script prints a line-ending-independent content hash, which should match the value above.

## Citation

Soumya Wasnik, 2026. Beyond Predictive Accuracy: An Actuarial Benchmark of Generalized Linear and Tree-Ensemble Models for Medical-Cost Prediction. In: Proceedings of the 2027 UOK-AIMT International Conference on Innovation in Actuarial Science, Management, Science and Technology (IAMST 2027), University of Kelaniya, Sri Lanka.
