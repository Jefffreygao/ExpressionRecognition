import argparse
import os
from calibration import collect_calibration_data, fine_tune_model
from detection import run_detection
from utils import CALIBRATION_DATA_DIR, PERSONALIZED_MODEL_PATH, BASE_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Real-time Personalized Emotion Recognizer")
    parser.add_argument('--mode', type=str, required=True, choices=['calibrate', 'detect', 'detect-base'],
                        help="'calibrate': Collect data and fine-tune."
                             "'detect': Run detection with the personalized model."
                             "'detect-base': Run detection with only the base model.")

    args = parser.parse_args()

    if args.mode == 'calibrate':
        # Step 1: Collect Data
        if not os.path.exists(BASE_MODEL_PATH):
             print(f"Error: Base model '{BASE_MODEL_PATH}' not found. Cannot proceed with calibration/fine-tuning.")
             return

        success = collect_calibration_data()

        # Step 2: Fine-tune only if data collection was successful and not aborted
        if success:
            # Check if enough data was collected (basic check)
            if os.path.exists(CALIBRATION_DATA_DIR) and len(os.listdir(CALIBRATION_DATA_DIR)) > 0:
                 fine_tune_model()
            else:
                 print("Calibration data directory is empty or missing. Fine-tuning skipped.")
        else:
            print("Calibration aborted or failed. Fine-tuning skipped.")

    elif args.mode == 'detect':
        if not os.path.exists(PERSONALIZED_MODEL_PATH):
             print(f"Error: Personalized model '{PERSONALIZED_MODEL_PATH}' not found.")
             print("Please run in 'calibrate' mode first.")
             # Option: Offer to run with base model instead?
             # print("Attempting to run with base model...")
             # run_detection(use_personalized=False)
             return
        else:
             run_detection(use_personalized=True)

    elif args.mode == 'detect-base':
         if not os.path.exists(BASE_MODEL_PATH):
             print(f"Error: Base model '{BASE_MODEL_PATH}' not found. Cannot run base detection.")
             return
         run_detection(use_personalized=False)

if __name__ == "__main__":
    # Crucial Check: Ensure haar cascade exists before starting
    if not os.path.exists('haarcascade_frontalface_default.xml'):
         print("FATAL ERROR: 'haarcascade_frontalface_default.xml' not found in the current directory.")
         print("Download it from OpenCV's repository and place it here.")
         exit() # Stop execution if the cascade is missing

    main()