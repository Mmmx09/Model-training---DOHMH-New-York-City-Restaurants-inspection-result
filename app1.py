import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="NYC Restaurant Score Prediction Demo",
    page_icon="🍽️",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = 'best_model.pkl'
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

st.title("Online Model Prediction")
st.markdown("Adjust the parameters below to predict the restaurant's next health inspection score.")

model = load_model()

expected_features = [
    'CAMIS', 'DBA', 'BORO', 'BUILDING', 'STREET', 'ZIPCODE', 'PHONE',
    'CUISINE DESCRIPTION', 'ACTION', 'VIOLATION CODE',
    'VIOLATION DESCRIPTION', 'CRITICAL FLAG', 'INSPECTION TYPE', 'Latitude',
    'Longitude', 'Community Board', 'Council District', 'Census Tract',
    'BIN', 'BBL', 'NTA', 'Location', 'days_since_last', 'avg_last_3_scores',
    'inspection_year', 'inspection_month', 'inspection_weekday',
    'ZIPCODE_na', 'Latitude_na', 'Longitude_na', 'Community Board_na',
    'Council District_na', 'Census Tract_na', 'BIN_na', 'BBL_na'
]

if model:
    st.subheader("Input Features")
    
    col1, col2 = st.columns(2)

    action_options = {
        1: "Violations were cited in the following area(s).",
        2: "Establishment Closed by DOHMH. Violations were cited in the following area(s) and those requiring immediate action were addressed.",
        3: "Establishment re-opened by DOHMH.",
        4: "No violations were recorded at the time of this inspection.",
        5: "Establishment re-closed by DOHMH." 
    }
    action_mapping_text = ", ".join([f"{k}: {v}" for k, v in action_options.items()])
    
    with col1:
        st.markdown("#### Core Features")
        avg_score = st.slider(
            "Average of Last 3 Inspection Scores", 
            0, 100, 15, help="Based on previous inspection scores."
        )
        days_last = st.number_input(
            "Days Since Last Inspection (days_since_last)", 
            0, 2000, 180, help="Time elapsed since the restaurant's last check."
        )
        
    with col2:
        st.markdown("#### Other Features (Simulated Input)")
        
        action_code = st.selectbox(
            "Action Code (Encoded Index)", 
            [0, 1, 2, 3, 4, 5], 
            index=1,
            help=f"Select the action corresponding to the inspection event. Encoded mapping: {action_mapping_text}."
        )
        violation_code = st.number_input(
            "Violation Code Encoded Index", 
            value=43,
            help="Simulated encoded index for VIOLATION CODE."
        )
        inspection_month = st.number_input("Inspection Month (1-12)", min_value=1, max_value=12, value=12)
        inspection_weekday = st.number_input("Inspection Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)

    predict_btn = st.button("Predict Score")

    if predict_btn:
        try:
            input_data = pd.DataFrame(0, index=[0], columns=expected_features)

            input_data['avg_last_3_scores'] = avg_score
            input_data['days_since_last'] = days_last
            input_data['ACTION'] = action_code
            input_data['VIOLATION CODE'] = violation_code
            input_data['inspection_month'] = inspection_month
            input_data['inspection_weekday'] = inspection_weekday
            
            prediction = model.predict(input_data)[0]
            
            st.markdown("---")
            st.metric(label="Predicted Score", value=f"{prediction:.2f}")
            
            if prediction <= 13:
                st.success("Predicted Grade: **A** (Excellent Hygiene)")
            elif prediction <= 27:
                st.warning("Predicted Grade: **B** (Good Hygiene)")
            else:
                st.error("Predicted Grade: **C** (Needs Improvement)")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Note: The Demo requires the input feature columns to exactly match the order and quantity used during model training, due to the complex preprocessing pipeline.")

else:
    st.error("The 'best_model.pkl' file was not detected. Please run Notebook 3 to generate and save the model, then place it in this directory.")