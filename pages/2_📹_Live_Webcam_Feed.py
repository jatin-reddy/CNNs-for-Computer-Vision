import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import cv2

st.set_page_config(page_title="Facial Analysis - Ethnicity, Gender, Age", page_icon="📹", layout="wide", initial_sidebar_state="expanded",
    menu_items={
        'About': "Upload an image to get a prediction of your race, gender and age according to my CNN Model"
    })

st.title("Face analysis on Webcam feed")
st.sidebar.success("Select an option above")
st.sidebar.info(
    "⚠️ Disclaimer: This model is for demonstration purposes only. "
    "Predictions for ethnicity, gender, and age may not be accurate. "
    "Results should not be used for any critical decisions."
)

@st.cache_resource(show_spinner=True)
def lazy_load_model():
    import keras
    model_path = '/Users/jatinreddy/CNNs-for-Computer-Vision/vgg_model_latest.keras'
    return keras.models.load_model(model_path)


def get_results(face_img, model):
    if model is None:
        return np.zeros(5), "Error", "N/A", 0

    # Preprocessing
    face_resized = cv2.resize(face_img, (128, 128))
    face_normalized = face_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(face_normalized, axis=0)
    
    try:
        results = model(img_batch, training=False)
        
        age_output = results[0]
        gender_output = results[1]
        race_output = results[2]

        # age
        age_pred = int(age_output[0][0] * 116.0)
        
        # gender
        gender_list = ["Male", "Female"]
        gender_idx = int(np.round(gender_output[0][0]))
        gender_idx = max(0, min(1, gender_idx))
        gender_pred = gender_list[gender_idx]

        # ethnicity
        race_preds = race_output[0] 
        ethnicity_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']
        dominant_label = ethnicity_labels[np.argmax(race_preds)]
        
        return race_preds, dominant_label, gender_pred, age_pred
        
    except Exception as e:
        print(f"Error: {e}")
        return np.zeros(5), "Error", "Error", 0

class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.detector = None
        self.frame_count = 0
        self.last_faces = []

    def recv(self, frame):
        import av # Lazy loading
        self.frame_count += 1
        # CV2 requires BGR 
        video_frame = frame.to_ndarray(format="bgr24")

        if self.detector is None:
            try: 
                from mtcnn import MTCNN # Lazy loading
                self.detector = MTCNN()
            
            except Exception as e:
                # incase MTCNN doesn't load just return the video frame without any processing
                print("MTCNN error:", e)
                return av.VideoFrame.from_ndarray(video_frame, format="bgr24")

        # MTCNN requires RGB frames
        
        rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)

        # run face detection every 3rd frame
        if self.frame_count % 10 == 0:
            try:
                self.last_faces = self.detector.detect_faces(rgb_frame)
            except Exception:
                self.last_faces = []
        
        for face in self.last_faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            cropped_face = rgb_frame[y:y+h, x:x+w]

            # load our VGG model
            model = lazy_load_model()
            probs, race, gender, age = get_results(cropped_face, model)
            conf = np.max(probs) * 100
            label = f"{race} ({conf:.0f}%) {gender}, {age}"
            

            cv2.rectangle(video_frame, (x, y), (x+w, y+h), (0,255,0), 2) # green
            # Draw results text with background for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(video_frame, (x, y-25), (x+tw+10, y), (0, 255, 0), -1)
            cv2.putText(video_frame, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(video_frame, format='bgr24')
    
st.write('Live Video feed from your webcam')
#RTC_CONFIGURATION = {}

webrtc_streamer(key="webcam", video_processor_factory=VideoProcessor, mode=WebRtcMode.SENDRECV, media_stream_constraints={"video": True, "audio": False}, async_processing=True,)
# rtc_configuration=RTC_CONFIGURATION

        
