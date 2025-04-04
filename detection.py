# detection.py
import cv2
import torch
import numpy as np
from model import load_personalized_model, load_base_model
from utils import (
    EMOTIONS, detect_face, preprocess_face, get_device,
    PERSONALIZED_MODEL_PATH, BASE_MODEL_PATH,
    load_sprite, overlay_sprite, SPRITE_SIZE
)

def run_detection(use_personalized=True):
    mode_string = "Personalized" if use_personalized else "Base Model"
    print(f"\n--- Starting Real-time Emotion Detection ({mode_string}) ---")
    print("Press 'b' to go back to the main menu.")
    print("Press 'q' to quit detection (same as 'b').")
    device = get_device()
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    model = None

    try:
        if use_personalized:
            print("Attempting to load PERSONALIZED model...")
            model = load_personalized_model(path=PERSONALIZED_MODEL_PATH, device=device)
            if model is None:
                print("Falling back to BASE model.")
                model = load_base_model(path=BASE_MODEL_PATH, device=device)
                if model is None:
                    print("Error: Could not load base model either. Exiting detection.")
                    return
                else:
                    mode_string = "Base Model (Fallback)"
        else:
            print("Loading BASE model...")
            model = load_base_model(path=BASE_MODEL_PATH, device=device)
            if model is None:
                print("Error: Could not load base model. Exiting detection.")
                return

        model.eval()

        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from webcam.")
            return
        frame_h, frame_w = frame.shape[:2]
        sprite_x = frame_w - SPRITE_SIZE[0] - 10
        sprite_y = 10
        sprite_position = (sprite_x, sprite_y)
        current_emotion_sprite = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            display_frame = frame.copy()
            bbox, face_roi = detect_face(frame)
            detected_emotion_label = None

            if bbox and face_roi is not None:
                (x, y, w, h) = bbox
                input_tensor = preprocess_face(face_roi)

                if input_tensor is not None and model is not None:
                    input_tensor = input_tensor.unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted_class = torch.max(probabilities, 1)
                    detected_emotion_label = EMOTIONS[predicted_class.item()]
                    confidence_score = confidence.item()

                    label_text = f"{detected_emotion_label} ({confidence_score:.2f})"
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(display_frame, (x, y - text_h - 15), (x + text_w + 5 , y - 10), (0, 255, 0), -1)
                    cv2.putText(display_frame, label_text, (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                elif model is None:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(display_frame, "Model Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(display_frame, "Processing Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if detected_emotion_label:
                current_emotion_sprite = load_sprite(detected_emotion_label)

            display_frame = overlay_sprite(display_frame, current_emotion_sprite, sprite_position)
            cv2.imshow(f"Real-time Emotion Detection ({mode_string})", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print(f"Detection ({mode_string}) quit by user ('q').")
                break
            elif key == ord('b'):
                print(f"Detection ({mode_string}) stopped by user ('b'). Returning to menu.")
                break

        print(f"--- Detection ({mode_string}) Loop Ended ---")

    finally:
        print("Releasing webcam and closing detection window...")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)