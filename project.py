from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo 
import streamlit as st
from sklearn.model_selection import train_test_split



st.title("Classification Model Dashboard")

# Fetch and prepare data
credit_data = fetch_ucirepo(id=144)
X = credit_data.data.features
y = credit_data.data.targets.squeeze()

# Rename columns
column_map = {
    'Attribute1': 'status',                # Status of existing checking account
    'Attribute2': 'duration',              # Duration in months
    'Attribute3': 'credit_history',        # Credit history
    'Attribute4': 'purpose',               # Purpose
    'Attribute5': 'credit_amount',         # Credit amount
    'Attribute6': 'savings',               # Savings account/bonds
    'Attribute7': 'employment',            # Present employment since
    'Attribute8': 'installment_rate',      # Installment rate in percentage of disposable income
    'Attribute9': 'personal_status',       # Personal status and sex
    'Attribute10': 'other_debtors',        # Other debtors / guarantors
    'Attribute11': 'residence',            # Present residence since
    'Attribute12': 'property',             # Property
    'Attribute13': 'age',                  # Age in years
    'Attribute14': 'other_installments',   # Other installment plans
    'Attribute15': 'housing',              # Housing
    'Attribute16': 'existing_credits',     # Number of existing credits at this bank
    'Attribute17': 'job',                  # Job
    'Attribute18': 'dependents',           # Number of people being liable to provide maintenance for
    'Attribute19': 'telephone',            # Telephone (yes/no)
    'Attribute20': 'foreign_worker'        # Foreign worker (yes/no)
}

X = X.rename(columns=column_map)

# Identify features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

st.subheader("Feature Categories")
col1, col2 = st.columns(2)
with col1:
    st.write("Categorical Features:")
    for feature in categorical_features:
        st.write(f"- {feature}")
with col2:
    st.write("Numerical Features:")
    for feature in numerical_features:
        st.write(f"- {feature}")

# Sidebar for feature selection
st.sidebar.title("Select Variable")
selected_cats = st.sidebar.multiselect("Select Categorical Features", categorical_features)
model_button = st.sidebar.button("Train Your Model")

# Train the model and store columns
if model_button and selected_cats:
    X_selected = pd.get_dummies(X[selected_cats], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.session_state['model'] = model
    st.session_state['model_columns'] = X_selected.columns.tolist()
    st.session_state['accuracy'] = accuracy
    st.success(f"Model trained! Accuracy: {accuracy:.2%}")

# Only show prediction UI if model is trained
if 'model' in st.session_state:
    # Mapping dictionaries (A# to description)
    status_map = {
        'A11': '< 0 DM',
        'A12': '0 <= ... < 200 DM',
        'A13': '>= 200 DM or salary assignment ≥ 1 year',
        'A14': 'no checking account'
    }
    credit_history_map = {
        'A30': 'no credits taken/ all paid back duly',
        'A31': 'all credits at this bank paid back duly',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account/ other credits existing (not at this bank)'
    }
    purpose_map = {
        'A40': 'car (new)',
        'A41': 'car (used)',
        'A42': 'furniture/equipment',
        'A43': 'radio/television',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': '(vacation - does not exist?)',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others'
    }

    # User input (use codes, not descriptions)
    status = st.selectbox("Status of Checking Account", list(status_map.keys()), format_func=lambda x: status_map[x])
    credit_history = st.selectbox("Credit History", list(credit_history_map.keys()), format_func=lambda x: credit_history_map[x])
    purpose = st.selectbox("Purpose", list(purpose_map.keys()), format_func=lambda x: purpose_map[x])
    duration = st.number_input("Duration in months", min_value=4, max_value=72, value=12)

    # Only use selected features for prediction
    user_input = {}
    if 'status' in selected_cats:
        user_input['status'] = status
    if 'credit_history' in selected_cats:
        user_input['credit_history'] = credit_history
    if 'purpose' in selected_cats:
        user_input['purpose'] = purpose
    if 'duration' in selected_cats:
        user_input['duration'] = duration

    if st.button("Predict Credit Health"):
        input_df = pd.DataFrame([user_input])
        input_encoded = pd.get_dummies(input_df)
        # Align columns with training data
        input_encoded = input_encoded.reindex(columns=st.session_state['model_columns'], fill_value=0)
        model = st.session_state['model']
        prediction = model.predict(input_encoded)
        probability = model.predict_proba(input_encoded)[0][list(model.classes_).index(1)] * 100

        if prediction[0] == 1:
            st.success("✅ Good Credit Risk")
        else:
            st.error("❌ Bad Credit Risk")

        st.subheader("Risk Probability vs. Benchmarks")
        st.write(f"**Predicted Risk Probability:** {probability:.1f}%")
        st.markdown(f"""
        | Benchmark               | Accuracy/Threshold |
        |-------------------------|--------------------|
        | Random Guessing         | 50%                |
        | Industry Standard       | 70%                |
        | Your Model              | {st.session_state['accuracy']:.2%}         |
        """)

else:
    st.info("Train your model using the sidebar to enable predictions.")

