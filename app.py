 
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Taplend | Loan Risk Review",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #202a2b;
        --muted: #697575;
        --paper: #f5f6f2;
        --line: #dfe4de;
        --green: #197a5b;
        --amber: #bd6b32;
    }
    .block-container { max-width: 1180px; padding-top: 1.7rem; padding-bottom: 3rem; }
    header[data-testid="stHeader"] { background: var(--paper); }
    .eyebrow { color: var(--green); font-size: .76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    .page-title { font: 800 2.15rem/1.15 'Manrope', sans-serif; margin: .35rem 0 .5rem; }
    .lede { color: var(--muted); font-size: 1rem; max-width: 760px; }
    .section-label { font: 700 1.05rem 'Manrope', sans-serif; padding-bottom: .5rem; border-bottom: 1px solid var(--line); margin: 1rem 0 .8rem; }
    div[data-testid="stForm"] { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 1rem 1.2rem; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); padding: 1rem; border-radius: 8px; }
    .disclaimer { border-left: 4px solid var(--amber); background: #fff5e9; padding: .85rem 1rem; border-radius: 0 6px 6px 0; color: #664322; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    model = joblib.load(BASE_DIR / "best_loan_model.joblib")
    features = joblib.load(BASE_DIR / "model_features.joblib")
    return model, features


try:
    model, expected_features = load_assets()
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.info("Install the packages in requirements.txt and confirm both model files are beside app.py.")
    st.stop()

st.markdown('<div class="eyebrow">Taplend · Decision support prototype</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Loan risk review</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="lede">Explore the model’s estimated repayment outcome for a proposed applicant. '
    'This is a classroom demonstration, not a lending decision.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer"><strong>Important model limitation</strong><br>'
    'Prototype only: holdout performance is modest (62% accuracy, 0.618 ROC-AUC; n=130). '
    'Requested terms stand in for approved-term model inputs unavailable at application time. '
    'Do not use this estimate to make lending decisions.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Review settings")
    threshold = st.slider(
        "Good-outcome threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Used only to label the model estimate. It is not a validated lending cutoff.",
    )
    st.caption("Model: XGBoost · Target: Good loan = 1, Bad loan = 0")
    st.markdown("---")
    st.caption("Use the result as an analysis aid only. Review fairness, calibration, and policy before any operational use.")

with st.form("applicant_form"):
    st.markdown('<div class="section-label">Applicant and requested loan</div>', unsafe_allow_html=True)
    first, second, third = st.columns(3)
    with first:
        amount_requested = st.number_input("Amount requested ($)", min_value=0.0, value=500.0, step=50.0)
        monthly_income = st.number_input("Monthly income ($)", min_value=0.0, value=2500.0, step=100.0)
        monthly_rent = st.number_input("Monthly housing cost ($)", min_value=0.0, value=500.0, step=50.0)
        age = st.number_input("Age at application", min_value=18, max_value=100, value=30)
    with second:
        loan_duration = st.number_input("Loan duration (months)", min_value=1, max_value=120, value=6)
        num_payments = st.number_input("Scheduled payments", min_value=1, max_value=120, value=12)
        payment_frequency = st.selectbox("Payment frequency", ["Bi-weekly", "Monthly", "Semi-monthly", "Weekly"])
        payment_ach = st.checkbox("Repayment by ACH", value=True)
    with third:
        residence = st.segmented_control("Residence", ["Rent", "Own"], default="Rent")
        direct_deposit = st.checkbox("Direct deposit", value=True)
        how_use_money = st.selectbox(
            "Loan purpose",
            [
                "Other", "Bills (Auto)", "Bills (General)", "Bills (Home / Utilities)",
                "Bills (Medical)", "Car", "Gifts / Leisure", "Medical", "Pay off loans / overdrawn account",
                "Rent", "Rent / Mortgage", "School", "Unknown", "Won't say",
            ],
        )

    st.markdown('<div class="section-label">Credit and stability indicators</div>', unsafe_allow_html=True)
    first, second, third = st.columns(3)
    with first:
        raw_l2c_score = st.number_input("L2C score", min_value=0, max_value=1000, value=600)
        fico_bank = st.number_input("FICO bank-card score", min_value=0, max_value=1000, value=650)
        fico_telecom = st.number_input("FICO telecom score", min_value=0, max_value=1000, value=0, help="Use 0 when unavailable, consistent with a missing score placeholder.")
        fico_retail = st.number_input("FICO retail score", min_value=0, max_value=1000, value=0, help="Use 0 when unavailable, consistent with a missing score placeholder.")
        fico_money = st.number_input("FICO money score", min_value=0, max_value=1000, value=0, help="Use 0 when unavailable, consistent with a missing score placeholder.")
    with second:
        email_duration = st.selectbox("Email account age", ["More than 12 months", "3 months or less", "4-12 months"])
        residence_duration = st.selectbox("Time at residence", ["1-3 years", "6 months or less", "7-12 months", "3+ years"])
        bank_duration = st.selectbox("Bank account age", ["1-3 years", "3+ years", "6 months or less", "7-12 months", "Unknown"])
        home_phone = st.selectbox("Home phone type", ["Other / unknown", "Mobile", "Work"])
        other_phone = st.selectbox("Additional phone type", ["Other / unknown", "Mobile", "None", "Work"])
    with third:
        address_zip = st.number_input("ZIP code", min_value=0, max_value=99999, value=0, help="Enter 0 if unavailable.")
        bank_routing = st.number_input("Bank routing number", min_value=0, max_value=9999999999, value=0, help="Enter 0 if unavailable.")
        st.markdown("**Model-only term proxies**")
        st.caption("The training model also requires approved terms. Because those are unavailable before review, requested terms are substituted below.")
        use_requested_terms = st.checkbox("Use requested terms as proxies", value=True)

    submitted = st.form_submit_button("Run risk estimate", type="primary", width="stretch")

if submitted:
    if not use_requested_terms:
        st.error("The model requires approved-term fields. Enable the disclosed requested-term proxy to continue.")
        st.stop()

    numeric_values = {
        "amount_requested": amount_requested,
        "monthly_rent_amount": monthly_rent,
        "loan_duration": loan_duration,
        "num_payments": num_payments,
        "payment_amount": amount_requested / num_payments,
        "amount_approved": amount_requested,
        "duration_approved": loan_duration,
        "payment_amount_approved": amount_requested / num_payments,
        "address_zip": address_zip,
        "bank_routing_number": bank_routing,
        "monthly_income_amount": monthly_income,
        "raw_l2c_score": raw_l2c_score,
        "raw_FICO_telecom": fico_telecom,
        "raw_FICO_retail": fico_retail,
        "raw_FICO_bank_card": fico_bank,
        "raw_FICO_money": fico_money,
        "age_at_application": age,
    }
    input_row = pd.DataFrame(0, index=[0], columns=expected_features, dtype=float)
    for feature, value in numeric_values.items():
        if feature in input_row.columns:
            input_row.loc[0, feature] = value

    selected_one_hot = {
        "residence_rent_or_own_True": residence == "Rent",
        "bank_account_direct_deposit_True": direct_deposit,
        "payment_ach_True": payment_ach,
        "email_duration_3_months_or_less": email_duration == "3 months or less",
        "email_duration_4-12_months": email_duration == "4-12 months",
        "residence_duration_3+_years": residence_duration == "3+ years",
        "residence_duration_6_months_or_less": residence_duration == "6 months or less",
        "residence_duration_7-12_months": residence_duration == "7-12 months",
        "bank_account_duration_3+_years": bank_duration == "3+ years",
        "bank_account_duration_6_months_or_less": bank_duration == "6 months or less",
        "bank_account_duration_7-12_months": bank_duration == "7-12 months",
        "bank_account_duration_Unknown": bank_duration == "Unknown",
        "payment_frequency_Monthly": payment_frequency == "Monthly",
        "payment_frequency_Semi-monthly": payment_frequency == "Semi-monthly",
        "payment_frequency_Weekly": payment_frequency == "Weekly",
        "home_phone_type_Mobile": home_phone == "Mobile",
        "home_phone_type_Work": home_phone == "Work",
        "other_phone_type_Mobile": other_phone == "Mobile",
        "other_phone_type_None": other_phone == "None",
        "other_phone_type_Work": other_phone == "Work",
    }
    purpose_feature = {
        "Bills (Auto)": "how_use_money_Bills_(Auto)",
        "Bills (General)": "how_use_money_Bills_(General)",
        "Bills (Home / Utilities)": "how_use_money_Bills_(Home_/_Utilities)",
        "Bills (Medical)": "how_use_money_Bills_(Medical)",
        "Car": "how_use_money_Car",
        "Gifts / Leisure": "how_use_money_Gifts_/_Leisure",
        "Medical": "how_use_money_Medical",
        "Other": "how_use_money_Other",
        "Pay off loans / overdrawn account": "how_use_money_Pay_off_loans_/_overdrawn_acct",
        "Rent": "how_use_money_Rent",
        "Rent / Mortgage": "how_use_money_Rent_/_Mortgage",
        "School": "how_use_money_School",
        "Unknown": "how_use_money_Unknown",
        "Won't say": "how_use_money_Won't_say",
    }
    selected_one_hot[purpose_feature[how_use_money]] = True
    for feature, selected in selected_one_hot.items():
        if feature in input_row.columns:
            input_row.loc[0, feature] = int(selected)

    classes = list(model.classes_)
    good_class_index = classes.index(1)
    probability_good = float(model.predict_proba(input_row)[0][good_class_index])
    predicted_good = probability_good >= threshold

    st.markdown("---")
    st.markdown('<div class="section-label">Model estimate</div>', unsafe_allow_html=True)
    result, probability, cutoff = st.columns([1.4, 1, 1])
    with result:
        if predicted_good:
            st.success("Predicted class: Good loan")
        else:
            st.warning("Predicted class: Bad loan")
    probability.metric("Estimated good-outcome probability", f"{probability_good:.1%}")
    cutoff.metric("Review threshold", f"{threshold:.0%}")
    st.progress(probability_good, text="Estimated probability of a good loan outcome")
    st.caption("The class label changes with the selected threshold. The probability is not calibrated, and neither output is a recommendation to approve or decline.")
