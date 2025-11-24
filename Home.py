import streamlit as st
from PIL import Image

st.set_page_config(page_title="Facial Analysis - Ethnicity, Gender, Age", page_icon="👤", layout="wide", initial_sidebar_state="expanded",
    menu_items={
        'About': "Upload an image or use your webcam to get a prediction of your race, gender and age according to my CNN Model"
    })

st.sidebar.title("Demo")
st.sidebar.success("Select an option above")
st.sidebar.info(
    "⚠️ Disclaimer: This model is for demonstration purposes only. "
    "Predictions for ethnicity, gender, and age may not be accurate. "
    "Results should not be used for any critical decisions."
)

st.title("Facial Analysis using CNNs 👤")
st.markdown(
    """
    This project demonstrates a multi-output Convolutional Neural Network model that predicts:
    - **Ethnicity**: White, Black, Asian, Indian, Others
    - **Gender**: Male or Female
    - **Age**: Approximate age in years

    The CNN model was trained from scratch on the [UTKFace dataset](https://www.kaggle.com/datasets/moritzm00/utkface-cropped) 
    and uses MTCNN for detecting faces in images. 
    """
)

st.markdown(
    """
    ### Model Performance
    - Trained a multi-output CNN from scratch.
    - Achieved best validation accuracy: **75.45%** (Ethnicity), **90.11%** (Gender)
    - Age prediction MSE: **0.0102**
    
    You can see the full **training notebook and analysis [here](https://github.com/jatin-reddy/CNNs-for-Computer-Vision)**.
    """
)

st.subheader("Demo")
demo_img = Image.open("assets/demo.jpeg")
st.image(demo_img, caption="Demo Image", use_container_width=True)

st.markdown(
    """
    ### Try It Out!
    Use the sidebar to:
    1. **Upload an image** for face analysis, or  
    2. **Use your webcam** to get live predictions.

    """
)




