import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os

# Constants
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = 48 # Input size for the model (typical for FER-2013)
FACE_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
CALIBRATION_DATA_DIR = 'calibration_data'
BASE_MODEL_PATH = 'base_model.pth'
PERSONALIZED_MODEL_PATH = 'personalized_model.pth'

# Check for face cascade file
if not os.path.exists(FACE_CASCADE_PATH):
    raise FileNotFoundError(f"Haar Cascade file not found at {FACE_CASCADE_PATH}. Download it from OpenCV's repository.")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Preprocessing transform for model input
preprocess_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    # Normalize if the model was trained with normalization
    # transforms.Normalize(mean=[0.5], std=[0.5]) # Example normalization
])

def detect_face(frame):
    """Detects the largest face in a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, None # No face detected

    # Find the largest face based on area (w*h)
    largest_face_idx = np.argmax([w*h for (x,y,w,h) in faces])
    (x, y, w, h) = faces[largest_face_idx]

    # Return the bounding box and the cropped face region from the original frame
    face_roi = frame[y:y+h, x:x+w]
    return (x, y, w, h), face_roi

def preprocess_face(face_roi):
    """Applies preprocessing transformations to the cropped face."""
    if face_roi is None or face_roi.size == 0:
        return None
    try:
        tensor = preprocess_transform(face_roi)
        return tensor
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def get_device():
    """Gets the appropriate device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")