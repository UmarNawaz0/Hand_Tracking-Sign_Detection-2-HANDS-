# Hand Tracking & Sign Language Interpreter (Two-Handed)

This project is a **Sign Language Interpreter** using **Machine Learning**, designed for recognizing alphabets using **both hands**.

## Hand Gesture Recognition using Machine Learning
This system implements **real-time hand gesture recognition** using **OpenCV, MediaPipe, and a Random Forest classifier**.

## Features
- **Hand tracking** (`HAND_TRACKING.py`) - Detects and tracks both hands.
- **Image collection** (`collect_imgs.py`) - Captures training images for two-handed alphabets.
- **Dataset creation** (`create_dataset.py`) - Processes collected images and generates a dataset.
- **Model training** (`train_classifier.py`) - Trains a classifier for gesture recognition.
- **Real-time inference** (`inference_classifier.py`) - Recognizes two-handed gestures in real-time.

## Usage Guide
1. **Collect Training Data**  
   Run `collect_imgs.py` to capture images for **two-handed** alphabets. first for right hand and then for left hand
  
3. **Create Dataset**  
   Run `create_dataset.py` to generate a pickle file for training.
4. **Train the Model**  
   Run `train_classifier.py` to train the classifier on the dataset.
5. **Perform Real-Time Recognition**  
   Run `inference_classifier.py` to recognize **two-handed** hand gestures.

## Installed Libraries
- **OpenCV (`cv2`)** - Image capture, processing, and display.
- **MediaPipe (`mediapipe`)** - Hand tracking and landmark extraction.
- **NumPy (`numpy`)** - Numerical operations and feature extraction.
- **scikit-learn (`sklearn`)** - Machine learning model training and evaluation.
- **pickle (`pickle`)** - Saving and loading models/datasets.
- **os (`os`)** - File system operations.
- **time (`time`)** - FPS calculations for `HAND_TRACKING.py`.

