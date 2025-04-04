# calibration.py
import cv2
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import load_base_model
from utils import (
    EMOTIONS, detect_face, preprocess_face,
    dataset_preprocess_transform, # Use the correct transform for ImageFolder
    CALIBRATION_DATA_DIR, BASE_MODEL_PATH, PERSONALIZED_MODEL_PATH, get_device,
    load_sprite, overlay_sprite, SPRITE_SIZE
)

NUM_IMAGES_PER_EMOTION = 15 # Number of images to capture per emotion
FINETUNE_EPOCHS = 5        # Number of epochs for fine-tuning
FINETUNE_LR = 0.0001       # Learning rate for fine-tuning

def collect_calibration_data():
    print("--- Starting Calibration ---")
    print(f"We need to capture {NUM_IMAGES_PER_EMOTION} images for each emotion.")
    print("\nA window will appear. Make the requested expression shown by the sprite.")
    print("When ready, press 'c' to capture the image.")
    print("Press 'b' to go back to the main menu.")
    print("Press 'q' to quit calibration (same as 'b').")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    calibration_successful = True

    try:
        if not os.path.exists(CALIBRATION_DATA_DIR):
            os.makedirs(CALIBRATION_DATA_DIR)

        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from webcam.")
            return False
        frame_h, frame_w = frame.shape[:2]
        sprite_x = frame_w - SPRITE_SIZE[0] - 10
        sprite_y = 10
        sprite_position = (sprite_x, sprite_y)

        for emotion in EMOTIONS:
            emotion_dir = os.path.join(CALIBRATION_DATA_DIR, emotion)
            if not os.path.exists(emotion_dir):
                os.makedirs(emotion_dir)
            target_sprite = load_sprite(emotion)
            print(f"\nPrepare to show: {emotion.upper()}")

            count = 0
            while count < NUM_IMAGES_PER_EMOTION:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    calibration_successful = False
                    break

                display_frame = frame.copy()
                bbox, face_roi = detect_face(frame)
                display_frame = overlay_sprite(display_frame, target_sprite, sprite_position)

                if bbox:
                    (x, y, w, h) = bbox
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = f"Show: {emotion.upper()} ({count}/{NUM_IMAGES_PER_EMOTION})"
                    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, "Press 'c' capture, 'b' back, 'q' quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(display_frame, "Press 'c' capture, 'b' back, 'q' quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(display_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_frame, "Press 'b' back, 'q' quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Calibration", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("Calibration quit by user ('q').")
                    calibration_successful = False
                    break
                elif key == ord('b'):
                    print("Calibration stopped by user ('b'). Returning to menu.")
                    calibration_successful = False
                    break

                elif key == ord('c') and face_roi is not None:
                    img_name = os.path.join(emotion_dir, f"{emotion}_{int(time.time())}_{count}.png")
                    cv2.imwrite(img_name, face_roi)
                    print(f"Captured image {count+1}/{NUM_IMAGES_PER_EMOTION} for {emotion}")
                    count += 1
                    display_frame_flash = overlay_sprite(frame.copy(), target_sprite, sprite_position)
                    cv2.rectangle(display_frame_flash, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.imshow("Calibration", display_frame_flash)
                    cv2.waitKey(100)

            if not calibration_successful:
                break

        if calibration_successful:
            print("\nCalibration data collection complete.")
        else:
            print("\nCalibration data collection aborted or failed.")

        return calibration_successful

    finally:
        print("Releasing webcam and closing calibration window...")
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)


def fine_tune_model():
    """Fine-tunes the base model using the collected calibration data."""
    print("\n--- Starting Fine-tuning ---")
    device = get_device()
    print(f"Using device: {device}")
    model = load_base_model(path=BASE_MODEL_PATH, device=device)
    if model is None: return
    if not os.path.exists(CALIBRATION_DATA_DIR):
        print(f"Error: Calibration directory '{CALIBRATION_DATA_DIR}' not found.")
        return

    print(f"[INFO] Loading calibration data from: {CALIBRATION_DATA_DIR}")
    try:
        calibration_dataset = ImageFolder(CALIBRATION_DATA_DIR, transform=dataset_preprocess_transform)
    except Exception as e:
        print(f"[ERROR] Failed to load calibration dataset: {e}")
        return

    if len(calibration_dataset) == 0:
        print("Error: No calibration images found. Please run calibration first.")
        return
    if len(calibration_dataset.classes) != len(EMOTIONS):
         print(f"Warning: Found {len(calibration_dataset.classes)} classes, expected {len(EMOTIONS)}.")
         print(f"Classes found: {calibration_dataset.classes}")

    print(f"[INFO] Calibration dataset loaded: {len(calibration_dataset)} samples found.")
    calibration_loader = DataLoader(calibration_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(f"Fine-tuning for {FINETUNE_EPOCHS} epochs...")
    for epoch in range(FINETUNE_EPOCHS):
        running_loss = 0.0
        images_processed = 0
        epoch_start_time = time.time()
        for i, data in enumerate(calibration_loader, 0):
            try:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                images_processed += inputs.size(0)
            except Exception as batch_error:
                print(f"[ERROR] Error during batch {i} processing in epoch {epoch+1}: {batch_error}")
                return

        if images_processed > 0:
             epoch_loss = running_loss / images_processed
             epoch_time = time.time() - epoch_start_time
             print(f"Epoch [{epoch+1}/{FINETUNE_EPOCHS}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
        else:
             print(f"Epoch [{epoch+1}/{FINETUNE_EPOCHS}] - No images processed.")

    try:
        torch.save(model.state_dict(), PERSONALIZED_MODEL_PATH)
        print(f"Fine-tuned model saved successfully to {PERSONALIZED_MODEL_PATH}")
    except Exception as e:
        print(f"Error saving fine-tuned model: {e}")

    print("--- Fine-tuning Complete ---")