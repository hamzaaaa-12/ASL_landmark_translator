# Real-Time ASL Alphabet Translator

A real-time American Sign Language (ASL) alphabet translator that recognizes static hand gestures and converts them into text using **MediaPipe Hand Landmarks**, **Geometric Feature Extraction**, and a **K-Nearest Neighbors (KNN) classifier**.

The project uses computer vision to detect hand landmarks, extract geometric features, classify ASL alphabet gestures, and build text through a real-time webcam interface.

## Project Overview

The main goal of this project is to develop a real-time system that translates ASL alphabet hand gestures into readable text.

Instead of classifying raw hand images using a Convolutional Neural Network (CNN), this implementation uses hand landmark coordinates and mathematical operations to represent hand shapes numerically.

This makes the recognition process based on hand geometry rather than image appearance.

## How It Works

The system follows this pipeline:

1. **Hand Detection:** MediaPipe detects the hand and extracts 21 landmarks.
2. **Feature Extraction:** Geometric features are calculated from landmark coordinates.
3. **Feature Representation:** Each hand gesture is represented by exactly 30 numerical features.
4. **Classification:** A trained KNN model predicts the corresponding ASL alphabet letter.
5. **Prediction Stabilization:** Consecutive predictions are used to reduce unstable recognition.
6. **Text Construction:** Recognized letters and gestures are used to build a sentence.

## Feature Extraction

Each hand is represented using 30 geometric features divided into three categories.

| Feature Category | Number of Features | Description |
|---|---:|---|
| Distance Features | 15 | Distances between fingertips and between the wrist and fingertips |
| Joint Angle Features | 10 | Angles calculated from finger joint landmarks |
| Straightness Features | 5 | Measures the straightness of each finger |
| **Total** | **30** | |

### 1. Distance Features

Euclidean Distance is used to calculate distances between selected hand landmarks.

The distances are normalized using the palm reference distance between:

- Wrist (Landmark 0)
- Middle Finger MCP (Landmark 9)

This helps reduce the influence of hand size and camera distance.

### 2. Joint Angle Features

Joint angles are calculated using the dot product between two vectors.

Two angles are extracted for each finger, resulting in 10 angle features.

### 3. Finger Straightness

Finger straightness is calculated by comparing:

- The direct distance between the first and last landmarks of a finger.
- The total distance along the finger's landmark segments.

This produces five straightness features, one for each finger.

## Classification

The project uses a **K-Nearest Neighbors (KNN)** classifier to recognize ASL alphabet hand gestures with 98% training accuracy and 97% test accuracy.

The classifier receives the extracted 30-dimensional feature representation and predicts the corresponding letter.

The trained model is saved using Joblib and loaded during live recognition.

## Gesture Controls

The system supports additional gestures to construct text:

| Gesture | Action |
|---|---|
| Hold a stable ASL letter | Add the recognized letter |
| Thumb + Index Finger Pinch | Add a space |
| Left Swipe | Delete the last character |
| Delete Last Character button | Delete the last character |
| Clear Text button | Clear the complete sentence |

Letter recognition uses a stability-based mechanism to prevent the same letter from being added repeatedly while the hand remains in the same position.

## Technologies Used

- Python
- OpenCV
- MediaPipe Tasks API
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Streamlit-WebRTC

## Project Structure

```text
ASL_landmark_translator/
│
├── data/
│
├── models/
│   ├── asl_knn_model.pkl
│   └── hand_landmarker.task
│
├── notebooks/
│
├── app.py
├── extract_dataset.py
├── handlandmark.py
├── live_predict_opencv.py
├── live_predict_streamlit.py
├── test_features.py
├── test_modules.py
│
├── requirements.txt
└── README.md
```

### Main Files

- **app.py:** Streamlit application interface and webcam configuration.
- **handlandmark.py:** MediaPipe hand detection, landmark visualization, and extraction of the 30 geometric features.
- **extract_dataset.py:** Dataset feature extraction workflow.
- **live_predict_opencv.py:** OpenCV-based live prediction implementation.
- **live_predict_streamlit.py:** Real-time prediction, gesture recognition, and text construction for the Streamlit application.
- **test_features.py:** Feature extraction testing.
- **test_modules.py:** Module and dependency testing.
- **models/:** Stores the trained KNN model and MediaPipe Hand Landmarker asset.
- **data/:** Stores project datasets.
- **notebooks/:** Development and experimentation notebooks.

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ASL_landmark_translator
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Model Files

Make sure the following files are available inside the `models/` directory:

- `asl_knn_model.pkl`
- `hand_landmarker.task`

The application requires both files for recognition.

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit, allow camera access, and start translating ASL gestures.

## Future Improvements

- Improve recognition accuracy across different users and lighting conditions.
- Expand the supported sign vocabulary.
- Improve recognition of motion-based ASL letters such as J and Z.
- Optimize real-time processing performance.
- Add text-to-speech functionality.
- Improve gesture recognition and sentence construction.

## Author

**Hamza Ibrahim**

Computer Science and Artificial Intelligence Student

Interested in Artificial Intelligence, Computer Vision, and building real-world technology solutions.
