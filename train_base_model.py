# train_base_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import time
import argparse

# Import model definition and utilities from existing files
from model import SimpleCNN
from utils import EMOTIONS, IMG_SIZE, BASE_MODEL_PATH, get_device

# --- Configuration ---
DATASET_DIR = 'dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

# Hyperparameters (adjust as needed)
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50 # FER-2013 might need more epochs (50-100+) for good performance

# How often to print training progress within an epoch (e.g., every 50 batches)
LOG_INTERVAL = 50

def main(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    """Trains the base emotion recognition model."""

    start_time = time.time()
    device = get_device()
    print(f"--- Training Configuration ---")
    print(f"Device:         {device}")
    print(f"Epochs:         {epochs}")
    print(f"Batch Size:     {batch_size}")
    print(f"Learning Rate:  {lr}")
    print(f"Dataset Path:   {DATASET_DIR}")
    print(f"Log Interval:   {LOG_INTERVAL} batches")
    print(f"----------------------------")

    # --- Data Loading and Preprocessing ---
    print("\n[INFO] Setting up data loaders...")

    # Define transformations
    # Training transforms: includes augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), # Slight rotation augmentation
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize grayscale images
    ])

    # Testing transforms: no augmentation, just preprocessing
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]) # Use the same normalization
    ])

    # Check if dataset directories exist
    if not os.path.isdir(TRAIN_DIR):
        print(f"[ERROR] Training directory not found at {TRAIN_DIR}")
        return
    if not os.path.isdir(TEST_DIR):
        print(f"[ERROR] Testing directory not found at {TEST_DIR}")
        return

    # Create datasets
    try:
        print(f"[INFO] Loading training data from: {TRAIN_DIR}")
        train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
        print(f"[INFO] Loading testing data from: {TEST_DIR}")
        test_dataset = ImageFolder(TEST_DIR, transform=test_transform)
        print("[INFO] Datasets loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error creating ImageFolder datasets: {e}")
        print("Please ensure 'dataset/train' and 'dataset/test' exist and contain subfolders named after emotions.")
        return

    # Verify classes match expected emotions (optional but good check)
    if sorted(train_dataset.classes) != sorted(EMOTIONS):
         print("[WARNING] Dataset classes do not perfectly match expected EMOTIONS list.")
         print(f"  Expected: {sorted(EMOTIONS)}")
         print(f"  Found:    {sorted(train_dataset.classes)}")
         num_classes = len(train_dataset.classes)
         print(f"[INFO] Proceeding with {num_classes} classes found in dataset.")
    else:
        num_classes = len(EMOTIONS)
        print(f"[INFO] Found {num_classes} classes matching expected emotions: {train_dataset.classes}")


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[INFO] Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"[INFO] Number of training batches: {len(train_loader)}, Number of testing batches: {len(test_loader)}")
    print("[INFO] Data loaders ready.")

    # --- Model, Loss, Optimizer ---
    print("\n[INFO] Initializing model, loss function, and optimizer...")
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Optional: Learning rate scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print("[INFO] Model, Loss, Optimizer initialized.")

    # --- Training Loop ---
    best_test_accuracy = 0.0
    print(f"\n[INFO] Starting training for {epochs} epochs...")
    total_batches = len(train_loader)

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        batch_start_time = time.time() # For timing batches

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            batch_loss = loss.item()
            running_loss += batch_loss * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Print progress every LOG_INTERVAL batches
            if (i + 1) % LOG_INTERVAL == 0 or (i + 1) == total_batches:
                batches_processed = i + 1
                percentage_complete = 100.0 * batches_processed / total_batches
                current_batch_time = time.time() - batch_start_time
                avg_batch_time = current_batch_time / batches_processed if i > 0 else current_batch_time # Avoid div by zero
                print(f"  Epoch {epoch+1} | Batch {batches_processed}/{total_batches} ({percentage_complete:.1f}%) | "
                      f"Batch Loss: {batch_loss:.4f} | Avg Batch Time: {avg_batch_time*1000:.2f}ms") # Print batch time in ms

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc_train = 100.0 * correct_train / total_train

        # --- Validation Step ---
        print(f"\n  [INFO] Starting validation for Epoch {epoch + 1}...")
        validation_start_time = time.time()
        model.eval()  # Set model to evaluation mode
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        with torch.no_grad(): # No gradients needed for validation
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_acc_test = 100.0 * correct_test / total_test
        epoch_test_loss = test_loss / len(test_dataset)
        validation_end_time = time.time()
        epoch_end_time = time.time() # Total epoch time includes validation

        print(f"  [INFO] Validation finished in {validation_end_time - validation_start_time:.2f}s")
        print(f"--- Epoch {epoch + 1} Summary ---")
        print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc_train:.2f}%")
        print(f"  Test Loss:  {epoch_test_loss:.4f} | Test Acc:  {epoch_acc_test:.2f}%")
        print(f"  Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")
        print(f"-------------------------")


        # Optional: Adjust learning rate with scheduler
        # scheduler.step()

        # --- Save the best model ---
        if epoch_acc_test > best_test_accuracy:
            best_test_accuracy = epoch_acc_test
            try:
                torch.save(model.state_dict(), BASE_MODEL_PATH)
                print(f"[SAVE] *** New best model saved to {BASE_MODEL_PATH} (Test Acc: {best_test_accuracy:.2f}%) ***")
            except Exception as e:
                print(f"[ERROR] Error saving model: {e}")

    total_time = time.time() - start_time
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Test Accuracy Achieved: {best_test_accuracy:.2f}%")
    print(f"Base model weights saved to: {BASE_MODEL_PATH}")


if __name__ == "__main__":
    # --- Argument Parser (Optional) ---
    parser = argparse.ArgumentParser(description="Train Base Emotion Recognition Model")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Training batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--log_interval', type=int, default=LOG_INTERVAL, help='How often to log training progress (batches)')
    args = parser.parse_args()

    # Update global LOG_INTERVAL if provided via command line
    LOG_INTERVAL = args.log_interval

    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)