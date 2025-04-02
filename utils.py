# utils.py
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import os
from PIL import Image # Import PIL Image

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

# --- Define TWO Transform Pipelines ---

# 1. Transform for real-time detection (input is likely NumPy array from OpenCV)
#    Includes ToPILImage() at the start.
detection_preprocess_transform = transforms.Compose([
    transforms.ToPILImage(), # Expects NumPy array or Tensor
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # Use same normalization as training
])

# 2. Transform for datasets loaded by ImageFolder (input is PIL Image)
#    OMITS ToPILImage() at the start.
dataset_preprocess_transform = transforms.Compose([
    # No ToPILImage() needed here, ImageFolder gives PIL Image
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # Use same normalization as training
])


def detect_face(frame):
    """Detects the largest face in a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, None # No face detected

    largest_face_idx = np.argmax([w*h for (x,y,w,h) in faces])
    (x, y, w, h) = faces[largest_face_idx]

    face_roi = frame[y:y+h, x:x+w]
    return (x, y, w, h), face_roi

def preprocess_face(face_roi):
    """Applies preprocessing transformations TO A NUMPY ARRAY face crop."""
    # This function specifically uses the 'detection_preprocess_transform'
    if face_roi is None or face_roi.size == 0:
        return None
    try:
        # Ensure input is suitable for ToPILImage if it's not already PIL
        if not isinstance(face_roi, (torch.Tensor, np.ndarray, Image.Image)):
             print(f"Warning: preprocess_face received unexpected type {type(face_roi)}")
             return None
        # If it's color OpenCV image (NumPy HWC BGR), convert to RGB for PIL
        if isinstance(face_roi, np.ndarray) and face_roi.ndim == 3:
             face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        tensor = detection_preprocess_transform(face_roi) # Use the transform expecting NumPy/Tensor
        return tensor
    except Exception as e:
        print(f"Error during preprocessing in preprocess_face: {e}")
        return None

def get_device():
    """Gets the appropriate device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")