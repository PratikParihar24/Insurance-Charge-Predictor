import streamlit as st
import sys
import os

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'src'))

from predict import preprocess_and_predict

# --- Streamlit Front-End Configuration ---

st.set_page_config(
    page_title="Insurance Charge Predictor",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("🏥 Individual Medical Insurance Charge Predictor")
st.markdown("A Machine Learning model predicting annual medical insurance charges based on personal and health factors.")
st.markdown("---")

# --- 1. Input Section: Symmetrical Layout ---

# Create two columns for a clean, symmetric input layout
col_general, col_health = st.columns(2)

with col_general:
    st.subheader("General & Demographic Factors")
    age = st.slider("Age (Years)", 18, 100, 30, help="Age of the primary beneficiary.")
    children = st.slider("Number of Children", 0, 5, 0, help="Number of dependents covered by the plan.")
    
    region_options = ('southeast', 'southwest', 'northeast', 'northwest')
    region = st.selectbox("Region of Residence", region_options, help="Where the beneficiary lives (4 regions in the US).")

with col_health:
    st.subheader("Health & Risk Factors")
    
    # BMI as a number input for precision
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=28.0, help="BMI > 30 is generally considered obese.")
    
    # Sex and Smoker status as radio buttons for clear choices
    sex_options = ('male', 'female')
    sex = st.radio("Sex", sex_options, horizontal=True) # horizontal=True makes it compact
    
    smoker_options = ('no', 'yes')
    smoker = st.radio("Smoker Status", smoker_options, index=0, horizontal=True, help="**The most important factor.** Smokers pay significantly more.")

st.markdown("---")

# --- 2. Prediction Trigger and Result Display ---

# Use a container for the result to keep it separate and visually focused
result_container = st.container()

# Prediction button centered and prominently placed
if st.button("Calculate Predicted Annual Charge", use_container_width=True):
    
    # Package all user inputs
    user_input = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    
    # Display a loading spinner while processing
    with st.spinner('Calculating...'):
        predicted_charge = preprocess_and_predict(user_input)
    
    # Display results in the container
    with result_container:
        st.subheader("💰 Predicted Annual Insurance Charge")
        
        # Use a large, attention-grabbing metric box
        st.success(f"**${predicted_charge:,.2f}**")
        
        # Add interpretation (Risk Analysis)
        st.markdown("#### Risk Analysis")
        
        if smoker == 'yes':
            st.error("🔴 **HIGH RISK:** Smoking is the primary driver of this high cost.")
        elif bmi > 30:
            st.warning("🟠 **ELEVATED RISK:** BMI over 30 (Obesity) significantly increased the predicted charge.")
        elif age > 50:
            st.warning("🟠 **ELEVATED RISK:** Age is a contributing factor to the predicted charge.")
        else:
            st.info("🟢 **AVERAGE RISK:** Charge is estimated based on general demographic and regional factors.")
            