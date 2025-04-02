import cv2
import torch
import numpy as np
from model import load_personalized_model, load_base_model
from utils import (
    EMOTIONS, detect_face, preprocess_face, get_device,
    PERSONALIZED_MODEL_PATH, BASE_MODEL_PATH
)

def run_detection(use_personalized=True):
    """Runs real-time emotion detection using the webcam."""
    print("\n--- Starting Real-time Emotion Detection ---")
    device = get_device()
    print(f"Using device: {device}")

    # Load the appropriate model
    if use_personalized:
        print("Attempting to load PERSONALIZED model...")
        model = load_personalized_model(path=PERSONALIZED_MODEL_PATH, device=device)
        if model is None:
            print("Falling back to BASE model.")
            model = load_base_model(path=BASE_MODEL_PATH, device=device)
            if model is None:
                print("Error: Could not load base model either. Exiting.")
                return
    else:
        print("Loading BASE model...")
        model = load_base_model(path=BASE_MODEL_PATH, device=device)
        if model is None:
             print("Error: Could not load base model. Exiting.")
             return

    model.eval() # Set model to evaluation mode

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        display_frame = frame.copy()
        bbox, face_roi = detect_face(frame)

        if bbox and face_roi is not None:
            (x, y, w, h) = bbox

            # Preprocess the detected face
            input_tensor = preprocess_face(face_roi)

            if input_tensor is not None:
                input_tensor = input_tensor.unsqueeze(0).to(device) # Add batch dimension and send to device

                # Perform inference
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)

                emotion_label = EMOTIONS[predicted_class.item()]
                confidence_score = confidence.item()

                # Draw bounding box and prediction
                label_text = f"{emotion_label} ({confidence_score:.2f})"
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display_frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Draw bounding box even if preprocessing failed
                 cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2) # Yellow box
                 cv2.putText(display_frame, "Processing Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            cv2.putText(display_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Real-time Emotion Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- Detection Stopped ---")