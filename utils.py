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
IMAGE_DIR = 'images'
SPRITE_SIZE = (100, 100)

# Check for face cascade file
if not os.path.exists(FACE_CASCADE_PATH):
    raise FileNotFoundError(f"Haar Cascade file not found at {FACE_CASCADE_PATH}. Download it from OpenCV's repository.")

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# --- Define TWO Transform Pipelines ---
detection_preprocess_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset_preprocess_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- Sprite Handling ---
sprite_cache = {}

def load_sprite(emotion):
    if emotion not in EMOTIONS:
        return None
    if emotion in sprite_cache:
        return sprite_cache[emotion]

    sprite_path = os.path.join(IMAGE_DIR, f"{emotion}.png")
    if not os.path.exists(sprite_path):
        print(f"[Warning] Sprite not found: {sprite_path}")
        sprite_cache[emotion] = None
        return None

    try:
        sprite = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
        if sprite is None:
            raise IOError("Failed to load sprite image.")
        resized_sprite = cv2.resize(sprite, SPRITE_SIZE, interpolation=cv2.INTER_AREA)
        sprite_cache[emotion] = resized_sprite
        return resized_sprite
    except Exception as e:
        print(f"[Error] Failed to load or resize sprite {sprite_path}: {e}")
        sprite_cache[emotion] = None
        return None

def overlay_sprite(background, sprite, position):
    if sprite is None:
        return background
    try:
        bg_h, bg_w = background.shape[:2]
        sp_h, sp_w = sprite.shape[:2]
        x, y = position
        if x + sp_w > bg_w: x = bg_w - sp_w
        if y + sp_h > bg_h: y = bg_h - sp_h
        if x < 0: x = 0
        if y < 0: y = 0
        roi = background[y:y + sp_h, x:x + sp_w]

        if sprite.shape[2] == 3:
            roi[:] = sprite
        elif sprite.shape[2] == 4:
            sprite_bgr = sprite[:, :, :3]
            alpha = sprite[:, :, 3] / 255.0
            alpha_3d = np.stack([alpha] * 3, axis=2)
            blended_roi = (sprite_bgr * alpha_3d + roi * (1.0 - alpha_3d)).astype(np.uint8)
            background[y:y + sp_h, x:x + sp_w] = blended_roi
        else:
            print(f"[Warning] Sprite has unexpected shape: {sprite.shape}")
    except Exception as e:
        print(f"[Error] Failed to overlay sprite: {e}")
    return background
# --- End Sprite Handling ---

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0: return None, None
    largest_face_idx = np.argmax([w*h for (x,y,w,h) in faces])
    (x, y, w, h) = faces[largest_face_idx]
    face_roi = frame[y:y+h, x:x+w]
    return (x, y, w, h), face_roi

def preprocess_face(face_roi):
    if face_roi is None or face_roi.size == 0: return None
    try:
        if not isinstance(face_roi, (torch.Tensor, np.ndarray, Image.Image)):
             print(f"Warning: preprocess_face received unexpected type {type(face_roi)}")
             return None
        if isinstance(face_roi, np.ndarray) and face_roi.ndim == 3:
             face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        tensor = detection_preprocess_transform(face_roi)
        return tensor
    except Exception as e:
        print(f"Error during preprocessing in preprocess_face: {e}")
        return None

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")