# CNNs-for-Computer-Vision
This is my attempt to use a convolutional neural network multi-output model that classifies images of people based on their ethnicity and predicts their age and gender.

The project explores multi-task learning, where a shared CNN backbone learns general facial features and three output heads specialise in classification (ethnicity and gender) and regression (age prediction).

## Key features 
- **Robust Data Pipeline** - Engineered a tf.data pipeline with safe augmentation techniques.
- **Single backbone + three prediction heads**:
  - Age (Regression)
  - Ethnicity (Classification)
  - Gender (Classification)
- Transfer Learning Comparison: Benchmarked the custom model against a fine-tuned MobileNetV2.
- **Streamlit Application** — Real-time face attribute prediction via image upload or live webcam.

***Dataset:*** https://www.kaggle.com/datasets/moritzm00/utkface-cropped

## Model Overview
The final model uses a custom **VGG-style architecture** with **Global Average Pooling** to reduce parameter bloat and improve generalization.

| Task | Output | Metric | Best Validation |
|------|--------|--------|-----------------|
| Age | Regression | MAE | **0.0749** |
| Gender | Classification | Accuracy | **90.11%** |
| Ethnicity | Classification | Accuracy | **75.45%** |

Later, the custom model was used to build a Streamlit application that predicts the three demographics.

## View Full Training Notebook
For full training details — including model code, metrics, plots, and experiments, see: **CNN_tf_keras.ipynb** (located in the repo root)

## Streamlit App Features
- Upload an image and get face attribute predictions
- Live webcam inference using Streamlit_WebRTC
- Lightweight & CPU-friendly deployment (uses lazy loading)
- Handles transparent (RGBA) images

### Installation
Follow the steps below to launch the Face Analysis Streamlit application on your local machine.

**Prerequisites**
Make sure you have:
- **Python 3.10**
- **Git**
- OS: Works on **Windows / macOS / Linux**


#### 1. Clone the Repository
```bash
git clone https://github.com/jatin-reddy/CNNs-for-Computer-Vision.git
cd CNNs-for-Computer-Vision
```
### 2. Create a Virtual Environment

**macOS / Linux**
```bash
python3.10 -m venv venv
source venv/bin/activate
```
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Application
``` bash
streamlit run Home.py
```

## ⚠ Disclaimer
Predictions are **probabilistic** and may **not always be accurate**.  
Results should **not be used for sensitive or critical decision-making**.
