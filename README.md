# Taplend Loan Risk Review

A Streamlit interface for exploring the supplied XGBoost loan outcome model. The app reports a model estimate; it does not approve or decline applications.

## Run on Windows

From this project folder, create and activate a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, run the environment executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

The model files `best_loan_model.joblib` and `model_features.joblib` must remain beside `app.py`.

## Deploy to Streamlit Community Cloud

Push `app.py`, `requirements.txt`, `README.md`, and both model artifacts to a GitHub repository. In Streamlit Community Cloud, create an app using that repository and set the main file to `app.py`. Keep the serialized model artifacts private if their license or source data requires it.

## Model limitations

- The notebook reports 62% accuracy and 0.618 ROC-AUC on a single stratified random holdout split of 130 examples. These results are modest and do not establish production performance.
- The saved model expects approved amount and approved payment-term features. Those values are unavailable at initial application review, so this demo substitutes requested terms; this can create a material train/serve mismatch.
- The supplied notebook does not report probability calibration, a time-based validation, subgroup/fairness evaluation, or a decision-cost analysis. The displayed probability and adjustable threshold should not be treated as a validated policy.
- The model was trained from a small, selected set of historical loans. Selection effects, missing or stale inputs, changing economic conditions, and proxy variables such as ZIP code or bank routing may affect performance and fairness.
- Before any real use, retrain with features available at decision time, validate on later and representative data, assess calibration and subgroup performance, document a decision policy, and obtain appropriate legal and compliance review.