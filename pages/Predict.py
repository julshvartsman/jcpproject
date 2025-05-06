import streamlit as st
import pandas as pd
import time

# Page config
st.set_page_config(
    page_title="Financial Risk Prediction Dashboard", 
    page_icon="📊", 
    layout="wide"
)

# Add custom CSS (same as home page for consistency)
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
    [data-testid="stSidebar"] {
        background-color: #d9edf7;
    }
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
</style>
""", unsafe_allow_html=True)

# Function to convert 1994 Deutsche Mark to 2025 USD (same as home page)
def convert_dm_to_usd(dm_value):
    # Historical exchange rate from 1994
    dm_to_usd_1994 = 0.65  # Approximate exchange rate in 1994
    
    # Inflation factor from 1994 to 2025
    # Using an estimated cumulative inflation of 107% over this period
    usd_inflation_factor = 2.07
    
    # Calculate the converted amount
    return dm_value * dm_to_usd_1994 * usd_inflation_factor

# Function to display timing in a nice box (same as home page)
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

# Title and header (same as home page)
st.title("Financial Risk Prediction Dashboard")
st.image("https://media.giphy.com/media/hZE5xoaM0Oxw4xiqH7/giphy.gif", caption="Financial Analytics")

# Sidebar with model info
st.sidebar.title("Current Model")
if 'model' in st.session_state:
    st.sidebar.markdown(f"""
    **Model Type:** Logistic Regression  
    **Accuracy:** {st.session_state.get('accuracy', 'N/A'):.2%}  
    **Features Used:** {len(st.session_state.get('selected_features', []))}  
    """)
    if st.sidebar.button("Train New Model"):
        st.switch_page("Home.py")
else:
    st.sidebar.warning("No model trained yet!")
    st.error("❌ No model found. Please train a model first.")
    st.stop()

# Currency display (same as home page)
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

# Main prediction interface
model_columns = st.session_state['model_columns']
selected_features = st.session_state.get('selected_features', [])

# Translation dictionaries (same as home page)
status_map = {
    'A11': 'Negative balance (< $0)' if using_usd else 'Negative balance (< 0 DM)',
    'A12': 'Low balance ($0 to < $1,500)' if using_usd else 'Low balance (0 <= ... < 200 DM)',
    'A13': 'Adequate balance (≥ $1,500)' if using_usd else 'Adequate balance (>= 200 DM or salary assignment ≥ 1 year)',
    'A14': 'No checking account'
}
credit_history_map = {
    'A30': 'No credits taken / all paid back duly',
    'A31': 'All credits at this bank paid back duly',
    'A32': 'Existing credits paid back duly till now',
    'A33': 'Delay in paying off in the past',
    'A34': 'Critical account / other credits existing (not at this bank)'
}
purpose_map = {
    'A40': 'Car (new)', 'A41': 'Car (used)', 'A42': 'Furniture/equipment',
    'A43': 'Radio/television', 'A44': 'Domestic appliances', 'A45': 'Repairs',
    'A46': 'Education', 'A47': 'Vacation',
    'A48': 'Retraining', 'A49': 'Business', 'A410': 'Others'
}
savings_map = {
    'A61': 'Less than $750' if using_usd else 'Less than 100 DM',
    'A62': '$750 - $1,800' if using_usd else '100 <= ... < 500 DM',
    'A63': '$1,800 - $4,500' if using_usd else '500 <= ... < 1000 DM',
    'A64': '$4,500 or more' if using_usd else '1000 DM or more',
    'A65': 'Unknown / no savings account'
}
employment_map = {
    'A71': 'Unemployed',
    'A72': 'Less than 1 year',
    'A73': '1 to less than 4 years',
    'A74': '4 to less than 7 years',
    'A75': '7 or more years'
}
personal_status_map = {
    'A91': 'Male: divorced/separated',
    'A92': 'Female: divorced/separated/married',
    'A93': 'Male: single',
    'A94': 'Male: married/widowed',
    'A95': 'Female: single'
}
other_debtors_map = {
    'A101': 'None',
    'A102': 'Co-applicant',
    'A103': 'Guarantor'
}
property_map = {
    'A121': 'Real estate',
    'A122': 'Building society savings/life insurance',
    'A123': 'Car or other',
    'A124': 'Unknown/no property'
}
other_installments_map = {
    'A141': 'Bank',
    'A142': 'Stores',
    'A143': 'None'
}
housing_map = {
    'A151': 'Rent',
    'A152': 'Own',
    'A153': 'For free'
}
job_map = {
    'A171': 'Unemployed/unskilled - non-resident',
    'A172': 'Unskilled - resident',
    'A173': 'Skilled employee/official',
    'A174': 'Management/self-employed/highly qualified employee'
}
telephone_map = {
    'A191': 'None',
    'A192': 'Yes (under customer name)'
}
foreign_worker_map = {
    'A201': 'Yes',
    'A202': 'No'
}

# Mapping between feature names and their respective dictionaries
feature_maps = {
    'status': status_map,
    'credit_history': credit_history_map,
    'purpose': purpose_map,
    'savings': savings_map,
    'employment': employment_map,
    'personal_status': personal_status_map,
    'other_debtors': other_debtors_map,
    'property': property_map,
    'other_installments': other_installments_map,
    'housing': housing_map,
    'job': job_map,
    'telephone': telephone_map,
    'foreign_worker': foreign_worker_map
}

# Credit Information Input Form
st.subheader("Enter Credit Information")
user_input = {}

# Create a form for better organization
with st.form("prediction_form"):
    # Display inputs for all categorical features the user selected for the model
    for feature in selected_features:
        if feature in feature_maps:
            feature_map = feature_maps[feature]
            value = st.selectbox(f"{feature.replace('_', ' ').title()}", 
                                list(feature_map.keys()),
                                format_func=lambda x: feature_map[x])
            user_input[feature] = value
        elif feature == 'duration':
            duration = st.number_input("Duration in months", min_value=4, max_value=72, value=12)
            user_input['duration'] = duration
        elif feature == 'age':
            age = st.number_input("Age in years", min_value=18, max_value=90, value=35)
            user_input['age'] = age
        elif feature == 'installment_rate':
            rate = st.slider("Installment Rate (% of disposable income)", min_value=1, max_value=4, value=2)
            user_input['installment_rate'] = rate
        elif feature == 'residence':
            residence = st.number_input("Present residence since (years)", min_value=1, max_value=4, value=2)
            user_input['residence'] = residence
        elif feature == 'existing_credits':
            credits = st.number_input("Number of existing credits", min_value=1, max_value=4, value=1)
            user_input['existing_credits'] = credits
        elif feature == 'dependents':
            dependents = st.number_input("Number of dependents", min_value=1, max_value=2, value=1)
            user_input['dependents'] = dependents
    
    # Handle credit amount with proper currency conversion
    currency_label = "Credit Amount (USD, 2025)" if using_usd else "Credit Amount (DM, 1994)"
    if ('credit_amount' in selected_features) or ('credit_amount_usd' in model_columns):
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
        elif not using_usd and 'credit_amount' in selected_features:
            user_input['credit_amount'] = credit_amount_dm

    submit_button = st.form_submit_button("Predict Credit Health", use_container_width=True)

# Process prediction when form is submitted
if submit_button:
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
    
    # Display results with visual formatting
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    with col1:
        if prediction[0] == 1:
            st.markdown("""
            <div style="padding: 20px; border-radius: 10px; background-color: #d4edda; border: 1px solid #c3e6cb;">
                <h3 style="color: #155724; margin: 0;">✅ Good Credit Risk</h3>
                <p style="color: #155724; font-size: 18px; margin: 10px 0 0 0;">{}</p>
            </div>
            """.format(risk_percentage), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="padding: 20px; border-radius: 10px; background-color: #f8d7da; border: 1px solid #f5c6cb;">
                <h3 style="color: #721c24; margin: 0;">❌ Bad Credit Risk</h3>
                <p style="color: #721c24; font-size: 18px; margin: 10px 0 0 0;">{}</p>
            </div>
            """.format(risk_percentage), unsafe_allow_html=True)
    
    with col2:
        show_timing("Prediction Processing Time", prediction_time)
    
    # Option to make another prediction
    st.markdown("---")
    if st.button("Make Another Prediction", use_container_width=True):
        st.rerun()