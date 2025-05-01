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
import time

# Initialize session state variables
if 'using_usd' not in st.session_state:
    st.session_state['using_usd'] = True

# Add custom CSS for an aesthetically pleasing dashboard with light blue background
st.markdown("""
<style>
    .stApp {
        background-color: #e8f4f8;
    }
    .main .block-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    /* Simpler sidebar styling that won't cause issues */
    [data-testid="stSidebar"] {
        background-color: #d9edf7;
    }
    /* Timing and feature cards */
    .timing-box {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px 15px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        border-left: 4px solid #3498db;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .timing-icon {
        font-size: 24px;
        margin-right: 15px;
    }
    .timing-content {
        flex-grow: 1;
    }
    .timing-title {
        font-weight: 500;
        color: #2c3e50;
    }
    .timing-value {
        color: #3498db;
        font-weight: 600;
        font-size: 18px;
    }
    .currency-indicator {
        background-color: #e3f2fd;
        color: #1976d2;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 4px;
        display: inline-block;
    }
    .feature-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 3px solid #2ecc71;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .feature-title {
        font-weight: 600;
        color: #27ae60;
        margin-bottom: 5px;
    }
    .feature-description {
        color: #555;
        font-size: 14px;
    }
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border-radius: 4px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    /* Risk score gauge styling */
    .risk-gauge-container {
        margin: 20px 0;
        padding: 15px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .risk-gauge-title {
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 10px;
        font-size: 16px;
        text-align: center;
    }
    .risk-gauge {
        position: relative;
        height: 120px;
        margin: 0 auto;
    }
    .risk-gauge-background {
        height: 60px;
        width: 100%;
        background: linear-gradient(90deg, #2ecc71, #f1c40f, #e74c3c);
        border-radius: 5px;
    }
    .risk-gauge-pointer {
        position: absolute;
        width: 3px;
        height: 60px;
        background-color: #2c3e50;
        bottom: 0;
    }
    .risk-score-text {
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        margin-top: 10px;
    }
    .risk-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 5px;
    }
    .risk-label {
        font-size: 12px;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# Function to display timing in a nice box
def show_timing(title, time_taken_ms):
    time_str = f"{time_taken_ms:.2f} ms"
    if time_taken_ms > 1000:
        time_str = f"{time_taken_ms/1000:.2f} s"
        
    html = f"""
    <div class="timing-box">
        <div class="timing-icon">⏱️</div>
        <div class="timing-content">
            <div class="timing-title">{title}</div>
            <div class="timing-value">{time_str}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Function to convert 1994 Deutsche Mark to 2025 USD
def convert_dm_to_usd(dm_value):
    # Historical exchange rate from 1994
    dm_to_usd_1994 = 0.65  # Approximate exchange rate in 1994
    
    # Inflation factor from 1994 to 2025
    # Using an estimated cumulative inflation of 107% over this period
    usd_inflation_factor = 2.07
    
    # Calculate the converted amount
    return dm_value * dm_to_usd_1994 * usd_inflation_factor

st.title("Financial Risk Prediction Dashboard")
st.image("https://media.giphy.com/media/hZE5xoaM0Oxw4xiqH7/giphy.gif", caption="Financial Analytics")

# Description and context
st.markdown("""
### About this Dashboard
This tool helps predict credit risk using the German Credit Dataset from 1994. Because the original data 
uses Deutsche Mark (DM) currency from 1994, we've added a conversion feature to show values in 2025 USD.

**Historical Context:** The Deutsche Mark (DM) was the official currency of West Germany (1948-1990) and 
unified Germany (1990-2001) before the Euro was adopted. The dataset originates from a time when Germany 
was adapting to reunification, making this data historically significant.
""")

# Currency toggle display
using_usd = st.session_state.get('using_usd', True)
current_currency = "USD (2025)" if using_usd else "DM (1994)"

st.markdown(f"""
<div style="display: flex; align-items: center; margin: 25px 0;">
    <span style="font-weight: 500; margin-right: 10px; color: #2c3e50;">Currency Mode:</span>
    <div class="currency-indicator">
        {current_currency}
    </div>
    <span style="margin-left: 15px; color: #7f8c8d; font-size: 14px;">
        {f"1 DM (1994) ≈ $1.35 USD (2025)" if using_usd else f"$1 USD (2025) ≈ 0.74 DM (1994)"}
    </span>
</div>
""", unsafe_allow_html=True)

# Data loading and preprocessing
start_time = time.time()
credit_data = fetch_ucirepo(id=144)
X = credit_data.data.features
y = credit_data.data.targets

# Fix: Handle squeezed series safely
if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
    y = y.iloc[:, 0]

# Rename columns and convert currency
column_map = {
    'Attribute1': 'status', 'Attribute2': 'duration', 'Attribute3': 'credit_history',
    'Attribute4': 'purpose', 'Attribute5': 'credit_amount', 'Attribute6': 'savings',
    'Attribute7': 'employment', 'Attribute8': 'installment_rate', 'Attribute9': 'personal_status',
    'Attribute10': 'other_debtors', 'Attribute11': 'residence', 'Attribute12': 'property',
    'Attribute13': 'age', 'Attribute14': 'other_installments', 'Attribute15': 'housing',
    'Attribute16': 'existing_credits', 'Attribute17': 'job', 'Attribute18': 'dependents',
    'Attribute19': 'telephone', 'Attribute20': 'foreign_worker'
}

X = X.rename(columns=column_map)
X['credit_amount_usd'] = X['credit_amount'].apply(convert_dm_to_usd)

preprocessing_time = (time.time() - start_time) * 1000

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

show_timing("Data Preprocessing Time", preprocessing_time)

# Display categorical features explanation
st.subheader("Categorical Features")
categorical_explanation = {
    "status": "Status of customer's checking account",
    "credit_history": "Customer's credit history",
    "purpose": "Purpose for which credit is needed",
    "savings": "Customer's savings accounts/bonds",
    "employment": "Present employment duration",
    "personal_status": "Personal status and gender",
    "other_debtors": "Other debtors / guarantors",
    "property": "Property owned by customer",
    "other_installments": "Other installment plans",
    "housing": "Housing situation",
    "job": "Type of job",
    "telephone": "Has telephone",
    "foreign_worker": "Foreign worker status"
}

cols = st.columns(2)
for i, (feature, description) in enumerate(categorical_explanation.items()):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">{feature}</div>
            <div class="feature-description">{description}</div>
        </div>
        """, unsafe_allow_html=True)

# Display sample data with currency conversion
sample_data = X[['credit_amount', 'credit_amount_usd']].sample(5).reset_index(drop=True)
sample_data.columns = ['Credit Amount (DM, 1994)', 'Credit Amount (USD, 2025)']
st.subheader("Sample Data with Currency Conversion")
st.markdown("""
This table shows random samples from the dataset with both original Deutsche Mark (DM) values 
and converted USD values. The conversion uses an approximate 1994 exchange rate adjusted for inflation to 2025.
""")
st.table(sample_data.style.format({
    'Credit Amount (DM, 1994)': '{:.2f} DM',
    'Credit Amount (USD, 2025)': '${:.2f}'
}))

# Sidebar interaction
st.sidebar.title("Model Configuration")
st.sidebar.markdown("""
Choose the features you want to include in your predictive model. Different combinations
of features will result in different model performance.
""")

selected_cats = st.sidebar.multiselect(
    "Select Categorical Features", 
    categorical_features,
    help="Choose categorical variables to include in your model"
)

use_usd = st.sidebar.checkbox(
    "Use USD (2025) instead of DM (1994)", 
    value=True,
    help="Convert all monetary values from original Deutsche Mark to modern USD"
)

# Add model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Logistic Regression", "Decision Tree", "Support Vector Machine"],
    help="Choose the machine learning algorithm to train"
)

model_button = st.sidebar.button("Train Your Model", help="Start training with selected features")

# Model training
if model_button:
    if len(selected_cats) < 1:
        st.warning("Please select at least one categorical feature to train the model.")
        st.stop()

    # Start timing
    start_model_time = time.time()
    selected_features = selected_cats.copy()

    if 'credit_amount' in selected_features and use_usd:
        selected_features.remove('credit_amount')
        selected_features.append('credit_amount_usd')

    X_selected = pd.get_dummies(X[selected_features], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Create the selected model type
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)  # Increased max_iter to ensure convergence
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    else:  # Support Vector Machine
        model = SVC(probability=True, random_state=42)
        
    # Train the model and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate feature importance when possible
    feature_importance = None
    if model_type == "Logistic Regression":
        # For logistic regression, use coefficients as importance
        feature_importance = pd.DataFrame({
            'feature': X_selected.columns,
            'importance': model.coef_[0]
        })
        feature_importance['abs_importance'] = abs(feature_importance['importance'])
        feature_importance = feature_importance.sort_values('abs_importance', ascending=False)
    elif model_type == "Decision Tree":
        # For decision trees, use feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_selected.columns,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
    model_training_time = (time.time() - start_model_time) * 1000
    model_training_time = (time.time() - start_model_time) * 1000

    st.session_state['model'] = model
    st.session_state['model_columns'] = X_selected.columns.tolist()
    st.session_state['accuracy'] = accuracy
    st.session_state['model_training_time'] = model_training_time
    st.session_state['using_usd'] = use_usd
    
    # Store feature importance if available
    if feature_importance is not None:
        st.session_state['feature_importance'] = feature_importance

    show_timing("Model Training Time", model_training_time)
    st.success(f"Model trained! Accuracy: {accuracy:.2%}")
    
    # Display key model metrics
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-title">Model Performance</div>
        <div class="feature-description">
            <strong>Model Type:</strong> {model_type}<br>
            <strong>Accuracy:</strong> {accuracy:.2%}<br>
            <strong>Features Used:</strong> {len(selected_features)}<br>
            <strong>Currency Mode:</strong> {"USD (2025)" if use_usd else "DM (1994)"}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display feature importance if available
    if feature_importance is not None:
        st.subheader("Feature Importance")
        st.write("These features have the most impact on credit risk prediction:")
        
        # Display top features in a table
        top_features = feature_importance.head(min(len(feature_importance), 5))
        
        # Create a DataFrame for display with formatted values
        display_df = top_features.copy()
        
        # For Logistic Regression, show direction of influence
        if model_type == "Logistic Regression":
            # Add a column for direction of influence
            display_df['direction'] = ['Increases risk' if imp < 0 else 'Decreases risk' 
                                      for imp in display_df['importance']]
            
            # Rename columns for display
            display_df = display_df[['feature', 'importance', 'direction']]
            display_df.columns = ['Feature', 'Impact Strength', 'Effect on Risk']
        else:
            # Just show feature and importance for other models
            display_df = display_df[['feature', 'importance']]
            display_df.columns = ['Feature', 'Importance']
        
        # Display the table
        st.table(display_df)

# Prediction block
if 'model' in st.session_state:
    # Fix: Prevent crash if model columns missing
    if 'model_columns' not in st.session_state:
        st.error("⚠️ Model columns not found. Please retrain the model.")
        st.stop()

    using_usd = st.session_state.get('using_usd', True)
    model_columns = st.session_state['model_columns']

    # Translation dictionaries
    status_map = {
        'A11': 'Negative bxalance (< $0)' if using_usd else '< 0 DM',
        'A12': 'Low balance ($0 to < $1,500)' if using_usd else '0 <= ... < 200 DM',
        'A13': 'Adequate balance (≥ $1,500)' if using_usd else '>= 200 DM or salary assignment ≥ 1 year',
        'A14': 'No checking account'
    }
    credit_history_map = {
        'A30': 'no credits taken/ all paid back duly',
        'A31': 'all credits at this bank paid back duly',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account/ other credits existing (not at this bank)'
    }
    purpose_map = {
        'A40': 'car (new)', 'A41': 'car (used)', 'A42': 'furniture/equipment',
        'A43': 'radio/television', 'A44': 'domestic appliances', 'A45': 'repairs',
        'A46': 'education', 'A47': '(vacation - does not exist?)',
        'A48': 'retraining', 'A49': 'business', 'A410': 'others'
    }
    savings_map = {
        'A61': '< $750' if using_usd else '< 100 DM',
        'A62': '$750 - $1,800' if using_usd else '100 <= ... < 500 DM',
        'A63': '$1,800 - $4,500' if using_usd else '500 <= ... < 1000 DM',
        'A64': '>= $4,500' if using_usd else '>= 1000 DM',
        'A65': 'unknown/no savings account'
    }

    # Only ask for inputs the model was trained on
    st.subheader("Enter Credit Information")
    user_input = {}
    
    if 'status' in selected_cats:
        status = st.selectbox("Status of Checking Account", list(status_map.keys()), 
                              format_func=lambda x: f"{x} - {status_map[x]}")
        user_input['status'] = status
    
    if 'credit_history' in selected_cats:
        credit_history = st.selectbox("Credit History", list(credit_history_map.keys()),
                                     format_func=lambda x: f"{x} - {credit_history_map[x]}")
        user_input['credit_history'] = credit_history
    
    if 'purpose' in selected_cats:
        purpose = st.selectbox("Purpose", list(purpose_map.keys()),
                              format_func=lambda x: f"{x} - {purpose_map[x]}")
        user_input['purpose'] = purpose
    
    if 'duration' in selected_cats:
        duration = st.number_input("Duration in months", min_value=4, max_value=72, value=12)
        user_input['duration'] = duration
    
    if 'savings' in selected_cats:
        savings = st.selectbox("Savings", list(savings_map.keys()),
                              format_func=lambda x: f"{x} - {savings_map[x]}")
        user_input['savings'] = savings

    # Handle credit amount with proper currency conversion
    currency_label = "Credit Amount (USD, 2025)" if using_usd else "Credit Amount (DM, 1994)"
    if ('credit_amount' in selected_cats) or ('credit_amount_usd' in model_columns):
        if using_usd:
            credit_amount_usd = st.number_input(currency_label, min_value=100.0, max_value=50000.0, value=5000.0, step=100.0)
            credit_amount_dm = credit_amount_usd / convert_dm_to_usd(1)
            st.write(f"Equivalent to: {credit_amount_dm:.2f} DM (1994)")
        else:
            credit_amount_dm = st.number_input(currency_label, min_value=100.0, max_value=50000.0, value=5000.0, step=100.0)
            credit_amount_usd = convert_dm_to_usd(credit_amount_dm)
            st.write(f"Equivalent to: ${credit_amount_usd:.2f} USD (2025)")

        if using_usd and 'credit_amount_usd' in model_columns:
            user_input['credit_amount_usd'] = credit_amount_usd
        elif not using_usd and 'credit_amount' in selected_cats:
            user_input['credit_amount'] = credit_amount_dm

    if st.button("Predict Credit Health"):
        pred_start_time = time.time()
        input_df = pd.DataFrame([user_input])
        input_encoded = pd.get_dummies(input_df)
        
        # Ensure all columns match the training data
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Only keep the columns used during training
        input_encoded = input_encoded[model_columns]
        
        model = st.session_state['model']
        prediction = model.predict(input_encoded)
        
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_encoded)[0][list(model.classes_).index(1)] * 100
            risk_percentage = f"Confidence: {probability:.1f}%"
        else:
            risk_percentage = ""

        prediction_time = (time.time() - pred_start_time) * 1000
        
        if prediction[0] == 1:
            st.success(f"✅ Good Credit Risk {risk_percentage}")
        else:
            st.error(f"❌ Bad Credit Risk {risk_percentage}")
            
        show_timing("Prediction Processing Time", prediction_time)

else:
    st.info("Train your model using the sidebar to enable predictions.")
