# Song-Recommendation-System
Emotion-Based Music Recommendation System using Real-Time Facial Expression Analysis

1.	INTRODUCTION
	Project Background
The Emotion-Based Music Recommendation System is an intelligent application designed to enhance user experience by providing personalized music based on real-time emotional state. The system utilizes facial expression analysis to detect emotions and recommends music playlists accordingly. By integrating computer vision, machine learning, and music streaming services, this project aims to create an interactive and emotionally-responsive digital environment, reducing the need for manual playlist selection and offering a novel way to enjoy music.
	Reference Application
This system draws inspiration from existing emotion-aware applications such as Spotify's mood-based playlists and AI-powered emotion recognition tools like Microsoft's Emotion API. These platforms highlight the growing trend of using affective computing to personalize digital experiences, particularly in entertainment and mental well-being applications.
	Project Objectives
	To develop a real-time facial emotion detection system using a Convolutional Neural Network (CNN) model.  
	To classify emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.  
	To integrate with Spotify API to fetch and embed emotion-specific music playlists.  
	To create a user-friendly web interface using Streamlit for seamless interaction.  
	To ensure system responsiveness and compatibility across different devices.

2.	 SYSTEM OVERVIEW
	Key Features:
	Real-time Face Detection: Uses OpenCV and Haarcascade for continuous face detection from webcam feed.
	Emotion Classification: Implements CNN model to classify facial expressions into 7 emotion categories with confidence scoring.
	Web-based Interface: Streamlit-powered UI with start/stop webcam controls and real-time emotion display
	Music Integration: Connects to Spotify API to fetch and embed emotion-specific playlists
	Session Management: Maintains user session state for seamless interaction and emotion tracking
	Responsive Design: Adapts to different screen sizes with proper layout and component sizing.
	Dataset Details:
The core component of this project, the Convolutional Neural Network (CNN) classifier, is trained and validated using the Facial Expression Recognition (FER-2013) dataset. This public dataset is the standard benchmark for seven-class facial emotion classification, aligning precisely with the project's seven target emotion categories (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).


Dataset Source Link: The specific dataset used, which is commonly employed in this project structure, can be found here: 
https://www.kaggle.com/datasets/msambare/fer2013/data
The dataset specifications are as follows:
•	Total Samples: 35,887 images.
•	Image Format: All images are standardized to 48x48 pixel grayscale.
Data Split: The dataset is pre-partitioned into a dedicated Training Set (28,709 images) and a Testing Set (3,589 images).
