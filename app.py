import streamlit as st 
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained model
model = load_model("CNN_Model_Final_Saved.keras")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Spotify Authentication
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="9c4ca04644f14ef682a98d748992b3cc",
                                                           client_secret="ba5487ae938747689fd7d44882a997f9"))

# Emotion-to-Spotify Playlist Mapping
emotion_playlists = {
    "Angry": "0KPEhXA3O9jHFtpd1Ix5OB",
    "Disgust": "37i9dQZF1DX4WYpdgoIcn6",
    "Fear": "37i9dQZF1DX7XfRr4cb6cr",
    "Happy": "37i9dQZF1DXdPec7aLTmlC",
    "Neutral": "37i9dQZF1DWWQRwui0ExPn",
    "Sad": "37i9dQZF1DX7qK8ma5wgG1",
    "Surprise": "37i9dQZF1DX9XIFQuFvzM4"
}

# Preprocess the frame for the model
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (48, 48))  # Resize to match the model input size
    if len(resized_frame.shape) == 2:  # If grayscale, convert to RGB
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
    elif resized_frame.shape[-1] == 1:  # If single channel, expand to 3 channels
        resized_frame = np.concatenate([resized_frame] * 3, axis=-1)
    img_array = np.expand_dims(resized_frame, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Predict emotion from the face
def predict_emotion(face):
    processed_face = preprocess_frame(face)
    predictions = model.predict(processed_face)
    return emotion_labels[np.argmax(predictions)]

# Get Spotify playlist iframe for detected emotion
def get_spotify_playlist_iframe(emotion):
    playlist_id = emotion_playlists.get(emotion)
    if playlist_id:
        iframe_code = f"""
        <iframe style="border-radius:12px" 
                src="https://open.spotify.com/embed/playlist/{playlist_id}?utm_source=generator&theme=0" 
                width="100%" 
                height="500" 
                frameBorder="0" 
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
                loading="lazy">
        </iframe>
        """
        return iframe_code
    else:
        return "No playlist available for this emotion."

# Streamlit UI
st.title("Emotion-Based Song Recommendations System 🎶")

# Webcam Control Buttons
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False
if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = None

# Start and Stop Webcam
if not st.session_state.run_webcam:
    if st.button("Start Webcam"):
        st.session_state.run_webcam = True
else:
    if st.button("Stop Webcam"):
        st.session_state.run_webcam = False
        st.session_state.detected_emotion = None

# Webcam and Emotion Detection
if st.session_state.run_webcam:
    cap = cv2.VideoCapture(0)
    video_feed = st.empty()
    emotion_display = st.empty()
    playlist_display = st.empty()

    # Real-time webcam feed with face detection
    while st.session_state.run_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam.")
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        # Process detected faces
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]  # Crop the face region
            detected_emotion = predict_emotion(face)
            st.session_state.detected_emotion = detected_emotion

            # Draw rectangle around face and display detected emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display video feed and detected emotion
        video_feed.image(frame, channels="BGR", caption="Webcam Feed")
        if st.session_state.detected_emotion:
            emotion_display.write(f"**Detected Emotion:** {st.session_state.detected_emotion}")
            playlist_iframe = get_spotify_playlist_iframe(st.session_state.detected_emotion)
            playlist_display.markdown(playlist_iframe, unsafe_allow_html=True)

    cap.release()
    st.write("Webcam stopped.")
