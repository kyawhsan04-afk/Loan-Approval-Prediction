# ============================================================
# BANK LOAN APPROVAL PREDICTION
# Streamlit Machine Learning Application
# ============================================================

import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# DATASET PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "loan_dataset.csv"


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        return None

    return pd.read_csv(DATA_FILE)


loan_dataset = load_data()


# ============================================================
# CHECK DATASET
# ============================================================

if loan_dataset is None:

    st.error("Dataset file not found.")

    st.write("Please make sure your GitHub repository contains:")

    st.code(
        """
loan-approval-prediction/
│
├── app.py
├── loan_dataset.csv
└── requirements.txt
"""
    )

    st.stop()


# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource
def train_model(data):

    df = data.copy()

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df = df.dropna()

    # --------------------------------------------------------
    # Encode Loan Status
    # --------------------------------------------------------

    df["Loan_Status"] = df["Loan_Status"].map({
        "N": 0,
        "Y": 1
    })

    # --------------------------------------------------------
    # Encode Dependents
    # --------------------------------------------------------

    df["Dependents"] = df["Dependents"].replace(
        "3+",
        "4"
    )

    # --------------------------------------------------------
    # Encode Categorical Columns
    # --------------------------------------------------------

    df["Gender"] = df["Gender"].map({
        "Male": 1,
        "Female": 0
    })

    df["Married"] = df["Married"].map({
        "Yes": 1,
        "No": 0
    })

    df["Education"] = df["Education"].map({
        "Graduate": 1,
        "Not Graduate": 0
    })

    df["Self_Employed"] = df["Self_Employed"].map({
        "Yes": 1,
        "No": 0
    })

    df["Property_Area"] = df["Property_Area"].map({
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    })

    # --------------------------------------------------------
    # Convert Numeric Columns
    # --------------------------------------------------------

    df["Dependents"] = pd.to_numeric(
        df["Dependents"]
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area"
    ]

    # Check columns
    required_columns = feature_columns + ["Loan_Status"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in dataset: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # X and Y
    # --------------------------------------------------------

    X = df[feature_columns]

    y = df["Loan_Status"]

    # --------------------------------------------------------
    # Train Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        stratify=y,
        random_state=2
    )

    # --------------------------------------------------------
    # SVM Model
    # --------------------------------------------------------

    model = SVC(
        kernel="linear"
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Training Accuracy
    # --------------------------------------------------------

    train_prediction = model.predict(X_train)

    train_accuracy = accuracy_score(
        y_train,
        train_prediction
    )

    # --------------------------------------------------------
    # Testing Accuracy
    # --------------------------------------------------------

    test_prediction = model.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    return (
        model,
        feature_columns,
        train_accuracy,
        test_accuracy,
        len(df)
    )


# ============================================================
# TRAIN MODEL
# ============================================================

try:

    (
        model,
        feature_columns,
        train_accuracy,
        test_accuracy,
        training_records
    ) = train_model(loan_dataset)

except Exception as error:

    st.error("Model training failed.")

    st.exception(error)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🏦 Bank Loan Approval Prediction")

st.write(
    "Enter the applicant's information below to predict "
    "whether the loan is likely to be approved."
)

st.info(
    "This application is an educational machine-learning "
    "project. Predictions should not be used as actual "
    "banking or financial decisions."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Model Information")

st.sidebar.write(
    "**Algorithm:** Support Vector Machine"
)

st.sidebar.write(
    "**Kernel:** Linear"
)

st.sidebar.metric(
    "Training Accuracy",
    f"{train_accuracy * 100:.2f}%"
)

st.sidebar.metric(
    "Test Accuracy",
    f"{test_accuracy * 100:.2f}%"
)

st.sidebar.write(
    f"**Training Records:** {training_records}"
)


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.header("Applicant Information")

left, right = st.columns(2)


# ============================================================
# LEFT SIDE
# ============================================================

with left:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )


# ============================================================
# RIGHT SIDE
# ============================================================

with right:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=100
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    loan_term = st.selectbox(
        "Loan Amount Term",
        [360, 180, 120, 240, 300, 60, 480]
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda value:
        "Good Credit History"
        if value == 1.0
        else "No / Bad Credit History"
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )


# ============================================================
# CONVERT INPUTS
# ============================================================

gender_value = (
    1
    if gender == "Male"
    else 0
)

married_value = (
    1
    if married == "Yes"
    else 0
)

dependents_value = (
    4
    if dependents == "3+"
    else int(dependents)
)

education_value = (
    1
    if education == "Graduate"
    else 0
)

self_employed_value = (
    1
    if self_employed == "Yes"
    else 0
)

property_area_value = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}[property_area]


# ============================================================
# CREATE MODEL INPUT
# ============================================================

input_data = [[
    gender_value,
    married_value,
    dependents_value,
    education_value,
    self_employed_value,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area_value
]]


input_dataframe = pd.DataFrame(
    input_data,
    columns=feature_columns
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

st.header("Loan Prediction")

predict_button = st.button(
    "Predict Loan Status",
    type="primary",
    use_container_width=True
)


if predict_button:

    prediction = model.predict(
        input_dataframe
    )

    if prediction[0] == 1:

        st.success(
            "## Loan Approved"
        )

        st.write(
            "The machine-learning model predicts that "
            "this loan application is likely to be approved."
        )

    else:

        st.error(
            "## Loan Not Approved"
        )

        st.write(
            "The machine-learning model predicts that "
            "this loan application is likely to be rejected."
        )


# ============================================================
# APPLICANT SUMMARY
# ============================================================

with st.expander("View Applicant Information"):

    st.write(
        f"**Gender:** {gender}"
    )

    st.write(
        f"**Married:** {married}"
    )

    st.write(
        f"**Dependents:** {dependents}"
    )

    st.write(
        f"**Education:** {education}"
    )

    st.write(
        f"**Self Employed:** {self_employed}"
    )

    st.write(
        f"**Applicant Income:** {applicant_income}"
    )

    st.write(
        f"**Coapplicant Income:** {coapplicant_income}"
    )

    st.write(
        f"**Loan Amount:** {loan_amount}"
    )

    st.write(
        f"**Loan Amount Term:** {loan_term}"
    )

    st.write(
        f"**Credit History:** "
        f"{'Good' if credit_history == 1.0 else 'No / Bad'}"
    )

    st.write(
        f"**Property Area:** {property_area}"
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander("Dataset Information"):

    st.write(
        f"Original dataset: "
        f"{loan_dataset.shape[0]} rows × "
        f"{loan_dataset.shape[1]} columns"
    )

    st.write(
        f"Records used for training: "
        f"{training_records}"
    )

    st.write(
        "The model removes rows containing missing values "
        "before training."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Bank Loan Approval Prediction | "
    "Support Vector Machine | "
    "Streamlit"
)
