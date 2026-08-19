# ============================================================
# BANK LOAN APPROVAL PREDICTION
# Streamlit + Support Vector Machine
# ============================================================

import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# FIND DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_dataset():
    """
    Find the loan dataset inside the Streamlit project.
    """

    # Expected filename
    expected_file = BASE_DIR / "bankloandata.csv"

    if expected_file.is_file():
        return expected_file

    # Search the project directory
    csv_files = list(BASE_DIR.rglob("*.csv"))

    if csv_files:
        return csv_files[0]

    return None


DATA_FILE = find_dataset()


# ============================================================
# DATASET ERROR HANDLING
# ============================================================

if DATA_FILE is None:

    st.error("❌ Dataset file not found.")

    st.write("Streamlit is currently running from:")

    st.code(str(BASE_DIR))

    st.write("No CSV file was found in this project.")

    st.write("Your GitHub repository should contain:")

    st.code(
        """
loan-approval-prediction/
│
├── app.py
├── bankloandata.csv
└── requirements.txt
"""
    )

    st.stop()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data(file_path):

    return pd.read_csv(file_path)


try:

    loan_dataset = load_data(DATA_FILE)

except Exception as error:

    st.error("❌ Unable to read the dataset.")

    st.exception(error)

    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Loan_ID",
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
    "Property_Area",
    "Loan_Status"
]


missing_columns = [
    column
    for column in required_columns
    if column not in loan_dataset.columns
]


if missing_columns:

    st.error("❌ Dataset format is incorrect.")

    st.write("The following columns are missing:")

    for column in missing_columns:
        st.write(f"- {column}")

    st.write("Columns found in your dataset:")

    st.write(list(loan_dataset.columns))

    st.stop()


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    df = data.copy()

    # --------------------------------------------------------
    # Remove missing values
    # --------------------------------------------------------

    df = df.dropna()

    # --------------------------------------------------------
    # Encode target variable
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

    df["Dependents"] = pd.to_numeric(
        df["Dependents"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Encode Gender
    # --------------------------------------------------------

    df["Gender"] = df["Gender"].map({
        "Male": 1,
        "Female": 0
    })

    # --------------------------------------------------------
    # Encode Married
    # --------------------------------------------------------

    df["Married"] = df["Married"].map({
        "Yes": 1,
        "No": 0
    })

    # --------------------------------------------------------
    # Encode Education
    # --------------------------------------------------------

    df["Education"] = df["Education"].map({
        "Graduate": 1,
        "Not Graduate": 0
    })

    # --------------------------------------------------------
    # Encode Self Employed
    # --------------------------------------------------------

    df["Self_Employed"] = df["Self_Employed"].map({
        "Yes": 1,
        "No": 0
    })

    # --------------------------------------------------------
    # Encode Property Area
    # --------------------------------------------------------

    df["Property_Area"] = df["Property_Area"].map({
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    })

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

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=feature_columns + ["Loan_Status"]
    )

    # --------------------------------------------------------
    # X and Y
    # --------------------------------------------------------

    X = df[feature_columns]

    Y = df["Loan_Status"]

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.1,
        stratify=Y,
        random_state=2
    )

    # --------------------------------------------------------
    # Support Vector Machine
    # --------------------------------------------------------

    classifier = SVC(
        kernel="linear"
    )

    # Train model
    classifier.fit(
        X_train,
        Y_train
    )

    # --------------------------------------------------------
    # Training Accuracy
    # --------------------------------------------------------

    X_train_prediction = classifier.predict(
        X_train
    )

    training_accuracy = accuracy_score(
        Y_train,
        X_train_prediction
    )

    # --------------------------------------------------------
    # Test Accuracy
    # --------------------------------------------------------

    X_test_prediction = classifier.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        Y_test,
        X_test_prediction
    )

    return (
        classifier,
        feature_columns,
        training_accuracy,
        test_accuracy,
        len(df)
    )


# ============================================================
# TRAIN MODEL
# ============================================================

try:

    (
        classifier,
        feature_columns,
        training_accuracy,
        test_accuracy,
        records_used
    ) = train_model(loan_dataset)

except Exception as error:

    st.error("❌ Model training failed.")

    st.exception(error)

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Bank Loan Approval Prediction")

st.write(
    "Enter the applicant's information below to predict "
    "whether the loan is likely to be approved."
)

st.info(
    "This is an educational machine-learning application. "
    "The prediction should not be treated as an actual "
    "banking or financial decision."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Information")

st.sidebar.write(
    "**Model:** Support Vector Machine"
)

st.sidebar.write(
    "**Kernel:** Linear"
)

st.sidebar.metric(
    "Training Accuracy",
    f"{training_accuracy * 100:.2f}%"
)

st.sidebar.metric(
    "Test Accuracy",
    f"{test_accuracy * 100:.2f}%"
)

st.sidebar.write(
    f"**Records Used:** {records_used}"
)

st.sidebar.write(
    f"**Dataset:** {DATA_FILE.name}"
)


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.header("Applicant Information")

left_column, right_column = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_column:

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
# RIGHT COLUMN
# ============================================================

with right_column:

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

gender_value = 1 if gender == "Male" else 0

married_value = 1 if married == "Yes" else 0

dependents_value = (
    4 if dependents == "3+"
    else int(dependents)
)

education_value = (
    1 if education == "Graduate"
    else 0
)

self_employed_value = (
    1 if self_employed == "Yes"
    else 0
)

property_area_value = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}[property_area]


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    [[
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
    ]],
    columns=feature_columns
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

st.header("Loan Prediction")

if st.button(
    "🔍 Predict Loan Status",
    type="primary",
    width="stretch"
):

    prediction = classifier.predict(
        input_data
    )

    if prediction[0] == 1:

        st.success(
            "## ✅ Loan Approved"
        )

        st.write(
            "The machine-learning model predicts that "
            "this application is likely to be approved."
        )

    else:

        st.error(
            "## ❌ Loan Not Approved"
        )

        st.write(
            "The machine-learning model predicts that "
            "this application is likely to be rejected."
        )


# ============================================================
# APPLICANT SUMMARY
# ============================================================

with st.expander("View Applicant Information"):

    st.write(f"**Gender:** {gender}")
    st.write(f"**Married:** {married}")
    st.write(f"**Dependents:** {dependents}")
    st.write(f"**Education:** {education}")
    st.write(f"**Self Employed:** {self_employed}")

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

    credit_status = (
        "Good Credit History"
        if credit_history == 1.0
        else "No / Bad Credit History"
    )

    st.write(
        f"**Credit History:** {credit_status}"
    )

    st.write(
        f"**Property Area:** {property_area}"
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.expander("Dataset Information"):

    st.write(
        f"**Dataset File:** {DATA_FILE.name}"
    )

    st.write(
        f"**Original Rows:** {loan_dataset.shape[0]}"
    )

    st.write(
        f"**Original Columns:** {loan_dataset.shape[1]}"
    )

    st.write(
        f"**Rows Used for Training:** {records_used}"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Bank Loan Approval Prediction | "
    "Support Vector Machine | Streamlit"
)
