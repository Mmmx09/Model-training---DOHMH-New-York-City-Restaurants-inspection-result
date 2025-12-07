import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="NYC restaurant inspection result prediction", page_icon="🍽️")
st.markdown("""
    <style>
    .main {
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
        background-color: #FF4B4B;
        color: white;
    }
    .result-box {
        padding: 50px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load('best_model.pkl')
    except:
        return None

model = load_model()

st.title("NYC restaurant inspection result predictio")
st.write("Input the restaurant information and predict its violation score for the next hygiene inspection (the lower the score, the better)")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🏠 Restaurant information")

    rest_name = st.text_input("Restaurant Name", placeholder="Example: Joe's Pizza")
    borough = st.selectbox("📍 Borough", ['Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Staten Island'])
    cuisine = st.selectbox("🍳 Cuisine", ['American', 'Chinese', 'Italian', 'Mexican', 'Japanese', 'Latin', 'Bakery', 'Other'])

with col2:
    st.markdown("### 📊 Historical and regulatory records")
    
    avg_scores = st.number_input("📉 Avg Last 3 Scores", 
                                 min_value=0, max_value=100, value=12, 
                                 help="The average score of the past three checks.")
    days_last = st.number_input("🗓️ Days Since Last", 
                                min_value=0, max_value=3650, value=180,
                                help="Number of days since the last inspection.")

    action_options = [
        'Violations were cited in the following area(s).',
        'Establishment Closed by DOHMH. Violations were cited in the following area(s) and those requiring immediate action were addressed.',
        'Establishment re-opened by DOHMH. ',
        'No violations were recorded at the time of this inspection.', 
        'Establishment re-closed by DOHMH. '
    ]
    action = st.selectbox("⚖️ Action", action_options)

if st.button("Predict Score"):
    
    input_df = pd.DataFrame({
        'BORO': [borough],
        'CUISINE DESCRIPTION': [cuisine],
        'avg_last_3_scores': [avg_scores],   
        'days_since_last': [days_last],     
        'ACTION': [action]                  
    })

    if model:
        try:
            prediction = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"The model prediction is incorrect.: {e}")
            st.info("Hint: Please check whether the column names in input_df are consistent with the features during model training.")
            prediction = 12

    st.markdown("---")
   
    if prediction <= 13:
        grade = "A"
        color = "#28a745" 
        msg = "Excellent"
    elif prediction <= 27:
        grade = "B"
        color = "#ffc107" 
        msg = "Good"
    else:
        grade = "C"
        color = "#dc3545" 
        msg = "Warning"

    st.markdown(f"""
    <div class="result-box" style="border: 2px solid {color}; background-color: {color}10;">
        <h3 style="color: grey;">{rest_name if rest_name else 'This Restaurant'} prediction result</h3>
        <h1 style="font-size: 80px; margin: 0; color: {color};">{grade}</h1>
        <p style="font-size: 24px; font-weight: bold;">prediction score: {prediction:.1f}</p>
        <p style="color: {color};">{msg}</p>
    </div>
    """, unsafe_allow_html=True)