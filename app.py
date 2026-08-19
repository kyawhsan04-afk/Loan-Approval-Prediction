import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ============================================================
# Bank Loan Approval Prediction
# Machine Learning Model: Support Vector Machine (SVM)
# ============================================================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Bank Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# Title
# ============================================================

st.title("🏦 Bank Loan Approval Prediction")

st.write(
    "This application uses a Support Vector Machine (SVM) "
    "machine-learning model to predict loan approval."
)

st.info(
    "This application is for educational purposes only. "
    "The prediction should not be used as a real banking decision."
)


# ============================================================
# Load Dataset
# ============================================================

DATA_FILE = "train_u6lujuX_CVtuZ9i (1).csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


try:
    loan_dataset = load_data()

except FileNotFoundError:
    st.error(
        f"Dataset not found: {DATA_FILE}"
    )
    st.stop()


# ============================================================
# Data Preprocessing
# ============================================================

@st.cache_resource
def train_model(data):

    # Make a copy so the original dataset is not modified
    data = data.copy()

    # Remove missing values
    data = data.dropna()

    # Convert Loan_Status into numerical values
    data["Loan_Status"] = data["Loan_Status"].map({
        "N": 0,
        "Y": 1
    })

    # Convert 3+ dependents to 4
    data["Dependents"] = data["Dependents"].replace(
        "3+",
        "4"
    )

    # Convert categorical columns
    data["Gender"] = data["Gender"].map({
        "Male": 1,
        "Female": 0
    })

    data["Married"] = data["Married"].map({
        "Yes": 1,
        "No": 0
    })

    data["Education"] = data["Education"].map({
        "Graduate": 1,
        "Not Graduate": 0
    })

    data["Self_Employed"] = data["Self_Employed"].map({
        "Yes": 1,
        "No": 0
    })

    data["Property_Area"] = data["Property_Area"].map({
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    })

    # Convert Dependents to integer
    data["Dependents"] = pd.to_numeric(
        data["Dependents"]
    )

    # ========================================================
    # Features and Target
    # ========================================================

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

    X = data[feature_columns]

    Y = data["Loan_Status"]

    # ========================================================
    # Train Test Split
    # ========================================================

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.1,
        stratify=Y,
        random_state=2
    )

    # ========================================================
    # SVM Model
    # ========================================================

    classifier = SVC(
        kernel="linear"
    )

    classifier.fit(
        X_train,
        Y_train
    )

    # ========================================================
    # Model Evaluation
    # ========================================================

    train_prediction = classifier.predict(X_train)

    train_accuracy = accuracy_score(
        Y_train,
        train_prediction
    )

    test_prediction = classifier.predict(X_test)

    test_accuracy = accuracy_score(
        Y_test,
        test_prediction
    )

    return (
        classifier,
        feature_columns,
        train_accuracy,
        test_accuracy,
        len(data)
    )


# ============================================================
# Train Model
# ============================================================

classifier, feature_columns, train_accuracy, test_accuracy, dataset_size = (
    train_model(loan_dataset)
)


# ============================================================
# Sidebar - Model Information
# ============================================================

st.sidebar.title("Model Information")

st.sidebar.write("**Algorithm:** Support Vector Machine")
st.sidebar.write("**Kernel:** Linear")

st.sidebar.metric(
    "Training Accuracy",
    f"{train_accuracy * 100:.2f}%"
)

st.sidebar.metric(
    "Testing Accuracy",
    f"{test_accuracy * 100:.2f}%"
)

st.sidebar.write(
    f"**Training records:** {dataset_size}"
)


# ============================================================
# Applicant Information
# ============================================================

st.header("Applicant Information")

col1, col2 = st.columns(2)


# ============================================================
# Column 1
# ============================================================

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


# ============================================================
# Column 2
# ============================================================

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

    loan_term = st.selectbox(
        "Loan Amount Term",
        [360, 180, 120, 240, 300, 60, 480]
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda x:
            "Good Credit History" if x == 1.0
            else "No/Bad Credit History"
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )


# ============================================================
# Convert User Input
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
# Create Input DataFrame
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
# Prediction Section
# ============================================================

st.divider()

st.header("Loan Prediction")

if st.button(
    "Predict Loan Status",
    type="primary",
    use_container_width=True
):

    prediction = classifier.predict(
        input_data
    )

    if prediction[0] == 1:

        st.success(
            "### Loan Approved"
        )

        st.write(
            "The machine-learning model predicts "
            "that the loan application is likely to be approved."
        )

    else:

        st.error(
            "### Loan Not Approved"
        )

        st.write(
            "The machine-learning model predicts "
            "that the loan application is likely to be rejected."
        )


# ============================================================
# Show Applicant Data
# ============================================================

with st.expander("View Applicant Information"):

    display_data = input_data.copy()

    display_data["Gender"] = (
        "Male" if gender_value == 1
        else "Female"
    )

    display_data["Married"] = (
        "Yes" if married_value == 1
        else "No"
    )

    display_data["Education"] = (
        "Graduate" if education_value == 1
        else "Not Graduate"
    )

    display_data["Self_Employed"] = (
        "Yes" if self_employed_value == 1
        else "No"
    )

    display_data["Property_Area"] = property_area

    display_data["Dependents"] = dependents

    st.dataframe(
        display_data,
        use_container_width=True
    )


# ============================================================
# Dataset Information
# ============================================================

with st.expander("View Dataset"):

    st.write(
        f"Original dataset size: "
        f"{loan_dataset.shape[0]} rows × "
        f"{loan_dataset.shape[1]} columns"
    )

    st.dataframe(
        loan_dataset.head(10),
        use_container_width=True
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Bank Loan Approval Prediction | "
    "Machine Learning Project using SVM and Streamlit"
)
