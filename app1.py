import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your pre-trained classification model
model = load_model('deepfake-detection-model.keras')  # Update with actual path

def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(cv2.CAP_PROP_FPS)  # Get FPS
    results = []

    while cap.isOpened():
        frameId = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get current frame number
        ret, frame = cap.read()
        
        if not ret:
            break  # End if video is over

        # Process every 0.5 seconds (previously every 1 second)
        if frameId % (int(frameRate) // 2) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            # Try different parameters to improve face detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))

            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w]
                crop_img = cv2.resize(crop_img, (128, 128))
                data = img_to_array(crop_img) / 255.0
                data = np.expand_dims(data, axis=0)  # Reshape for model
                
                predictions = model.predict(data)
                predicted_prob = predictions[0][1]  # Probability of Real
                

                if predicted_prob > 0.6:  # Threshold can be tuned (try 0.5 to 0.7)
                    results.append(1)  # Real
                else:
                    results.append(0)  # Fake
    cap.release()
    return results

st.title("Deep Fake Video Detection")
st.write("Upload a video to check if it is real or fake.")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video("temp_video.mp4")
    
    if st.button("Detect Deepfake"):
        results = detect_deepfake("temp_video.mp4")
        real_count = results.count(1)
        fake_count = results.count(0)
        total = len(results)
        st.write(f"Real Faces Detected: {real_count}")
        st.write(f"Fake Faces Detected: {fake_count}")
        if results:
            if fake_count==0:
                st.write("The video is likely **real**.")
            else:
                st.write("The video is likely a **fake**.")
        else:
            st.write("No faces detected in the video.")