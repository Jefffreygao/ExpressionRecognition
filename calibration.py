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
    EMOTIONS, detect_face, preprocess_face, preprocess_transform,
    CALIBRATION_DATA_DIR, BASE_MODEL_PATH, PERSONALIZED_MODEL_PATH, get_device
)

NUM_IMAGES_PER_EMOTION = 5 # Number of images to capture per emotion
FINETUNE_EPOCHS = 5        # Number of epochs for fine-tuning
FINETUNE_LR = 0.0001       # Learning rate for fine-tuning

def collect_calibration_data():
    """Guides the user to capture images for each emotion."""
    print("--- Starting Calibration ---")
    print(f"We need to capture {NUM_IMAGES_PER_EMOTION} images for each of the following emotions:")
    print(", ".join(EMOTIONS))
    print("\nA window will appear. Make the requested expression.")
    print("When ready, press 'c' to capture the image.")
    print("Press 'q' to quit calibration at any time.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    if not os.path.exists(CALIBRATION_DATA_DIR):
        os.makedirs(CALIBRATION_DATA_DIR)

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(CALIBRATION_DATA_DIR, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)

        print(f"\nPrepare to show: {emotion.upper()}")
        # Give user time to prepare
        for i in range(5, 0, -1):
            print(f"{i}...")
            time.sleep(1)

        count = 0
        while count < NUM_IMAGES_PER_EMOTION:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            display_frame = frame.copy()
            bbox, face_roi = detect_face(frame)

            if bbox:
                (x, y, w, h) = bbox
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Display instructions on the frame
                text = f"Show: {emotion.upper()} ({count}/{NUM_IMAGES_PER_EMOTION})"
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


            cv2.imshow("Calibration", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Calibration aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return False # Indicate abortion

            if key == ord('c') and face_roi is not None:
                # Save the *original grayscale* face crop for potential review,
                # but fine-tuning uses the transformed tensor later.
                # For simplicity here, let's save the processed image directly if needed,
                # or just rely on the DataLoader to handle it during fine-tuning.
                # Let's save the color crop for now.
                img_name = os.path.join(emotion_dir, f"{emotion}_{int(time.time())}_{count}.png")
                cv2.imwrite(img_name, face_roi)
                print(f"Captured image {count+1}/{NUM_IMAGES_PER_EMOTION} for {emotion}")
                count += 1
                # Optional: Short feedback flash
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
                cv2.imshow("Calibration", display_frame)
                cv2.waitKey(100) # Display red flash briefly


    print("\nCalibration data collection complete.")
    cap.release()
    cv2.destroyAllWindows()
    return True # Indicate success

def fine_tune_model():
    """Fine-tunes the base model using the collected calibration data."""
    print("\n--- Starting Fine-tuning ---")
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load Base Model
    model = load_base_model(path=BASE_MODEL_PATH, device=device)
    if model is None: return # Error handled in load_base_model

    # 2. Prepare Dataset and DataLoader
    if not os.path.exists(CALIBRATION_DATA_DIR):
        print(f"Error: Calibration directory '{CALIBRATION_DATA_DIR}' not found.")
        return

    # Use ImageFolder - requires images to be in CALIBRATION_DATA_DIR/emotion_name/image.png
    # Apply the same preprocessing used for inference
    calibration_dataset = ImageFolder(CALIBRATION_DATA_DIR, transform=preprocess_transform)

    if len(calibration_dataset) == 0:
        print("Error: No calibration images found. Please run calibration first.")
        return

    # Check if all classes are present (optional but good practice)
    if len(calibration_dataset.classes) != len(EMOTIONS):
         print(f"Warning: Found {len(calibration_dataset.classes)} classes, expected {len(EMOTIONS)}. Ensure all emotion folders have images.")
         # Ensure class_to_idx matches EMOTIONS order if necessary - ImageFolder sorts alphabetically
         print(f"Classes found by ImageFolder: {calibration_dataset.classes}")
         # Potentially remap indices if order doesn't match EMOTIONS, but ImageFolder is usually consistent if folders are named correctly.


    calibration_loader = DataLoader(calibration_dataset, batch_size=8, shuffle=True) # Small batch size for fine-tuning

    # 3. Define Optimizer and Loss Function
    # Fine-tune only the last few layers or the whole model with a small LR
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)
    criterion = nn.CrossEntropyLoss()

    # 4. Fine-tuning Loop
    model.train() # Set model to training mode
    print(f"Fine-tuning for {FINETUNE_EPOCHS} epochs...")
    for epoch in range(FINETUNE_EPOCHS):
        running_loss = 0.0
        images_processed = 0
        for i, data in enumerate(calibration_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            images_processed += inputs.size(0)

        epoch_loss = running_loss / images_processed
        print(f"Epoch [{epoch+1}/{FINETUNE_EPOCHS}], Loss: {epoch_loss:.4f}")

    # 5. Save the Fine-tuned Model
    try:
        torch.save(model.state_dict(), PERSONALIZED_MODEL_PATH)
        print(f"Fine-tuned model saved successfully to {PERSONALIZED_MODEL_PATH}")
    except Exception as e:
        print(f"Error saving fine-tuned model: {e}")

    print("--- Fine-tuning Complete ---")