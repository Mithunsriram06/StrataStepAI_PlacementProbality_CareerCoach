import streamlit as st
import pandas as pd
import pickle
from recommendation import get_career_recommendations

st.set_page_config(page_title="StrataStepAI", layout="wide", page_icon="🎯")

#Loading the ML model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model not found! Please run `python ml_model.py` first.")
        return None

model = load_model()

if 'reset_key' not in st.session_state:
    st.session_state.reset_key = 0

def reset_session():
    
    current_key = st.session_state.get('reset_key', 0)
    st.session_state.clear()
    st.session_state.reset_key = current_key + 1

#Sidebar navigatiom
st.sidebar.title("StrataStepAI")
page = st.sidebar.radio("Navigation", ["Placement probability checker", "Career recommendation"])

st.sidebar.divider()
st.sidebar.button("Reset Session", on_click=reset_session, type="primary")

#Phase 1: Placement Probability Checker
if page == "Placement probability checker":
    st.title("📊 Placement Probability Checker")
    st.markdown("Enter your academic and extracurricular details to predict your placement score.")
    r_key = str(st.session_state.reset_key)
    col1, col2 = st.columns(2)
    
    with col1:
        cgpa = st.text_input("CGPA (e.g., 8.5)", "0.0", key=f"cgpa_{r_key}")
        aptitude = st.text_input("Aptitude Test Score (out of 100)", "0", key=f"apt_{r_key}")
        ssc_marks = st.text_input("SSC Marks (%)", "0", key=f"ssc_{r_key}")
        hsc_marks = st.text_input("HSC Marks (%)", "0", key=f"hsc_{r_key}")
        
    with col2:
        internships = st.selectbox("Internships Completed", ["0", "1", "2", "2+"], key=f"int_{r_key}")
        projects = st.selectbox("Projects Completed", ["0", "1", "2", "2+"], key=f"proj_{r_key}")
        workshops = st.selectbox("Workshops/Certifications", ["0", "1", "2", "2+"], key=f"work_{r_key}")
        soft_skills = st.selectbox("Soft Skills Rating (out of 5)", ["below 4", "4-4.5", "4.5 above"], key=f"soft_{r_key}")
        
    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        extracurricular = st.radio("Extracurricular Activities", ["No", "Yes"], key=f"extra_{r_key}")
    with col4:
        placement_training = st.radio("Placement Training Completed", ["No", "Yes"], key=f"train_{r_key}")

    if st.button("Calculate Probability Score", use_container_width=True):
        try:
            map_counts = {"0": 0, "1": 1, "2": 2, "2+": 3}
            map_soft = {"below 4": 3.5, "4-4.5": 4.25, "4.5 above": 4.8}
            
            input_data = pd.DataFrame([{
                'CGPA': float(cgpa),
                'Internships': map_counts[internships],
                'Projects': map_counts[projects],
                'Workshops/Certifications': map_counts[workshops],
                'AptitudeTestScore': int(aptitude),
                'SoftSkillsRating': map_soft[soft_skills],
                'ExtracurricularActivities': 1 if extracurricular == "Yes" else 0,
                'PlacementTraining': 1 if placement_training == "Yes" else 0,
                'SSC_Marks': int(ssc_marks),
                'HSC_Marks': int(hsc_marks)
            }])
            
            if model:
                prob = model.predict_proba(input_data)[0][1]
                score_percentage = round(prob * 100, 2)
                
                st.success(f"### Predicted Placement Probability: {score_percentage}%")
                st.progress(prob)
                
        except ValueError:
            st.error("Please enter valid numerical values in text inputs.")

#Phase 2: Career Recommendation & Roadmap
elif page == "Career recommendation":
    st.title("🧭 Career Recommendation & Roadmap")
    st.markdown("Upload your resume and enter a target job role to get an AI-generated actionable roadmap.")
    r_key = str(st.session_state.reset_key)
    
    job_role = st.text_input("Desired Job Role", placeholder="e.g., Data Scientist, Frontend Developer", key=f"role_{r_key}")
    uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"], key=f"file_{r_key}")
    
    if st.button("Generate Strategy Roadmap", use_container_width=True):
        if job_role and uploaded_resume:
            with st.spinner("StrataStepAI is analyzing your resume and generating a roadmap..."):
                try:
                    result = get_career_recommendations(job_role, uploaded_resume)
                    st.divider()
                    st.markdown(result)
                except Exception as e:
                    st.error(f"API Error: Make sure your .env file has the correct key and you are using the google-genai library. Details: {e}")
        else:
            st.warning("Please provide both a Desired Job Role and a Resume PDF.")