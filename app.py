import streamlit as st

st.set_page_config(page_title="Facial Analysis - Ethnicity, Gender, Age", page_icon="🧊", layout="wide", initial_sidebar_state="expanded",
    menu_items={
        'About': "Upload an image or use your webcam to get a prediction of your race, gender and age according to my CNN Model"
    })

st.title("Main page")

st.sidebar.success("Select an option above")
st.sidebar.info(
    "⚠️ Disclaimer: This model is for demonstration purposes only. "
    "Predictions for ethnicity, gender, and age may not be accurate. "
    "Results should not be used for any critical decisions."
)

