


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
df_num = pd.read_csv('data/german.data-numeric', delim_whitespace=True, header=None)
df_cat = pd.read_csv('data/german.data', delim_whitespace=True, header=None)

st.write(df_num.head())
st.write(df_num.describe())
st.write(df_num.info())
st.write(df_cat.head())
st.write(df_cat.describe())
st.write(df_cat.info())


credit_data = fetch_ucirepo(id=144)
X = credit_data.data.features
y = credit_data.data.targets
    
    
st.subheader("Dataset Summary")
try:
    # Load dataset
    data = fetch_ucirepo(id=144)
    X = data.data.features

    # Column mapping based on UCI documentation
    column_map = {
        'Attribute1': 'status',
        'Attribute2': 'duration',
        'Attribute3': 'credit_history',
        'Attribute4': 'purpose',
        'Attribute5': 'credit_amount',
        'Attribute6': 'savings',
        'Attribute7': 'employment',
        'Attribute8': 'installment_rate',
        'Attribute9': 'personal_status',
        'Attribute10': 'other_debtors',
        'Attribute11': 'residence',
        'Attribute12': 'property',
        'Attribute13': 'age',
        'Attribute14': 'other_installments',
        'Attribute15': 'housing',
        'Attribute16': 'existing_credits',
        'Attribute17': 'job',
        'Attribute18': 'dependents',
        'Attribute19': 'telephone',
        'Attribute20': 'foreign_worker'
    }
    
    # Rename columns for better readability
    X = X.rename(columns=column_map)
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Display feature categories in Streamlit
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
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

