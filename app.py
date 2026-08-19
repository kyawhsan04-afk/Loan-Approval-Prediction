import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Bank Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Loan Approval Prediction")
st.write(
    "Enter the applicant's information below to predict whether "
    "the loan is likely to be approved."
)


# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("dataset.csv")
    return data


loan_dataset = load_data()


# ---------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------

@st.cache_resource
def train_model(data):

    # Remove missing values
    data = data.dropna().copy()

    # Encode Loan_Status
    data["Loan_Status"] = data["Loan_Status"].replace({
        "N": 0,
        "Y": 1
    })

    # Replace 3+ dependents with 4
    data["Dependents"] = data["Dependents"].replace("3+", 4)

    # Convert categorical columns
    data = data.replace({
        "Married": {
            "No": 0,
            "Yes": 1
        },
        "Gender": {
            "Male": 1,
            "Female": 0
        },
        "Self_Employed": {
            "Yes": 1,
            "No": 0
        },
        "Property_Area": {
            "Rural": 0,
            "Semiurban": 1,
            "Urban": 2
        },
        "Education": {
            "Graduate": 1,
            "Not Graduate": 0
        }
    })

    # Make sure Dependents is numeric
    data["Dependents"] = pd.to_numeric(data["Dependents"])

    # Separate features and target
    X = data.drop(
        columns=["Loan_ID", "Loan_Status"],
        errors="ignore"
    )

    Y = data["Loan_Status"]

    # Train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.1,
        stratify=Y,
        random_state=2
    )

    # Create SVM classifier
    classifier = SVC(kernel="linear")

    # Train model
    classifier.fit(X_train, Y_train)

    # Test accuracy
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)

    return classifier, X.columns.tolist(), accuracy


classifier, feature_columns, test_accuracy = train_model(loan_dataset)


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.header("Model Information")

st.sidebar.write("Model: Support Vector Machine")
st.sidebar.write("Kernel: Linear")

st.sidebar.metric(
    "Test Accuracy",
    f"{test_accuracy * 100:.2f}%"
)


# ---------------------------------------------------
# User Input
# ---------------------------------------------------

st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:

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


with col2:

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

    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=12.0,
        value=360.0,
        step=12.0
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda x: "Good (1)" if x == 1.0 else "Bad (0)"
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )


# ---------------------------------------------------
# Convert Input to Model Format
# ---------------------------------------------------

input_data = {
    "Gender": 1 if gender == "Male" else 0,

    "Married": 1 if married == "Yes" else 0,

    "Dependents": 4 if dependents == "3+" else int(dependents),

    "Education": 1 if education == "Graduate" else 0,

    "Self_Employed": 1 if self_employed == "Yes" else 0,

    "ApplicantIncome": applicant_income,

    "CoapplicantIncome": coapplicant_income,

    "LoanAmount": loan_amount,

    "Loan_Amount_Term": loan_term,

    "Credit_History": credit_history,

    "Property_Area": {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }[property_area]
}


input_dataframe = pd.DataFrame(
    [input_data],
    columns=feature_columns
)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

st.divider()

if st.button("Predict Loan Status", type="primary"):

    prediction = classifier.predict(input_dataframe)

    if prediction[0] == 1:

        st.success(
            "Loan Approved"
        )

        st.write(
            "Based on the trained machine-learning model, "
            "the applicant is predicted to have an approved loan status."
        )

    else:

        st.error(
            "Loan Not Approved"
        )

        st.write(
            "Based on the trained machine-learning model, "
            "the applicant is predicted to have a non-approved loan status."
        )


# ---------------------------------------------------
# Dataset Information
# ---------------------------------------------------

with st.expander("View Dataset Information"):

    st.write(
        f"Dataset contains {loan_dataset.shape[0]} rows "
        f"and {loan_dataset.shape[1]} columns."
    )

    st.dataframe(loan_dataset.head())
