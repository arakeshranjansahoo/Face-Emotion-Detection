import streamlit as st
import cv2
import numpy as np
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

from PIL import Image
import time
import pandas as pd

# Load model
model = load_model("emotion_detection_model.keras")
categories = ['angry', 'fear', 'disgust', 'happy', 'neutral', 'sad', 'surprise']

# Prediction function
def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    processed = np.expand_dims(resized, axis=(0, -1)) / 255.0
    preds = model.predict(processed)
    return categories[np.argmax(preds)], preds[0]

# Streamlit UI setup
st.set_page_config(page_title="Emotion Detection", page_icon="😊", layout="wide")

st.title("😊 Real-Time Emotion Detection Dashboard")
st.markdown("Detect emotions in real-time or from uploaded images using a CNN model.")

# Sidebar
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose Mode", ["📷 Webcam Mode", "🖼️ Image Upload", "📊 History", "ℹ️ About"])

# Initialize history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Webcam Mode
if mode == "📷 Webcam Mode":
    st.subheader("Live Webcam Feed")

    run = st.checkbox("Start Webcam")

    col1, col2 = st.columns([2, 1])  # Dashboard layout

    with col1:
        FRAME_WINDOW = st.image([])

    with col2:
        emotion_placeholder = st.empty()
        chart_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to grab frame.")
            break

        emotion, preds = predict_emotion(frame)

        # Save to history (keep last 10)
        st.session_state.history.append({"time": time.strftime("%H:%M:%S"), "emotion": emotion})
        if len(st.session_state.history) > 10:
            st.session_state.history.pop(0)

        with col1:
            FRAME_WINDOW.image(frame, channels="BGR")

        with col2:
            emotion_placeholder.markdown(
                f"<h2 style='text-align:center;'>Detected Emotion: "
                f"<span style='color:green;'>{emotion.capitalize()}</span></h2>",
                unsafe_allow_html=True
            )

            chart_data = {categories[i]: float(preds[i]) for i in range(len(categories))}
            chart_placeholder.bar_chart(chart_data)

        time.sleep(0.05)

    cap.release()

# Image Upload Mode
elif mode == "🖼️ Image Upload":
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        emotion, preds = predict_emotion(img_cv)

        # Save to history
        st.session_state.history.append({"time": time.strftime("%H:%M:%S"), "emotion": emotion})
        if len(st.session_state.history) > 10:
            st.session_state.history.pop(0)

        with col2:
            st.markdown(
                f"<h2 style='text-align:center;'>Detected Emotion: "
                f"<span style='color:blue;'>{emotion.capitalize()}</span></h2>",
                unsafe_allow_html=True
            )

            chart_data = {categories[i]: float(preds[i]) for i in range(len(categories))}
            st.bar_chart(chart_data)

# History Mode
elif mode == "📊 History":
    st.subheader("Detection History (Last 10 Records)")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.table(df)

        # Show trend of detected emotions
        emotion_counts = df["emotion"].value_counts()
        st.bar_chart(emotion_counts)
    else:
        st.info("No history available yet. Try running webcam or uploading an image.")

# About Section
elif mode == "ℹ️ About":
    st.subheader("About this App")
    st.markdown("""
    This application uses a Convolutional Neural Network (CNN) trained on the **FER-2013 dataset**
    to classify facial emotions into:
    **Angry, Fear, Disgust, Happy, Neutral, Sad, Surprise**.

    ### Features
    ✅ Real-time webcam detection  
    ✅ Single image upload prediction  
    ✅ Dashboard view with probability charts  
    ✅ History tracker (last 10 predictions with timestamps)  
    ✅ Clean & interactive UI  

    ---
    Built by A Rakesh Ranjan Sahoo, using **Streamlit, OpenCV, and TensorFlow**.
    """)
