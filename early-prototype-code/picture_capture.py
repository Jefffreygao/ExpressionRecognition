# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
import cv2 as cv2
import os
import torch
from torchvision import transforms
from Model import FacialRecognitionModel

expressions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
folder = "personalized_dataset"
total_people = 0
id = 0
width = 1920
height = 1080

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FacialRecognitionModel(num_classes=7)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def set_folder():
    if not os.path.exists(folder):
        os.makedirs(folder)

    for expression in expressions:
        expression_path = os.path.join(folder, expression)
        if not os.path.exists(expression_path):
            os.makedirs(expression_path)

def capture_expressions(index):
    cap = cv2.VideoCapture(0)
    cap.set(4, height)
    cap.set(3, width)

    for expression in expressions:
        print(f"Display '{expression}' expression. Press 'C' to start capturing images.")
        on_loop = True
        counter = 0
        while on_loop:
            ret, frame = cap.read()
            cv2.putText(frame, f"Display the {expression.capitalize()} Facial Expression | Press 'C' Key to capture |",
                        (150, 50), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
            cv2.imshow('Capture Window', frame)
            if not ret or (cv2.waitKey(1) & 0xFF == 27):
                expression_folder = f"{folder}/{expression}"
                cv2.imwrite(os.path.join(expression_folder,
                                         alternate_filename(expression_folder, f"{expression}N{counter}I{id}.jpg")),
                            cv2.resize(frame, (48, 48)))
                counter += 1
                on_loop = False
    print(f"Image Capturing is finished for person {index}")
    cap.release()
    cv2.destroyAllWindows()

def alternate_filename(folder_path, filename):
    counter = 1
    new_filename = filename
    b, e = os.path.splitext(filename)
    path = os.path.join(folder_path, new_filename)
    while os.path.exists(path):
        new_filename = f"{b}({counter}){e}"
        path = os.path.join(folder_path, new_filename)
        counter += 1
    return new_filename

def classify_emotions(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(0)
    cap.set(4, height)
    cap.set(3, width)
    on_loop = True
    while on_loop:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tensor = transforms.ToTensor()(cv2.resize(gray, (48, 48))).unsqueeze(0).to(device)
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])(tensor).to(device)
        cv2.putText(frame, f"Real Time Predicted Emotion: {expressions[torch.argmax(model(normalize), 1).item()].capitalize()}",
                    (350, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0) , 2, cv2.LINE_AA)
        cv2.putText(frame, f"Move the camera till it views your whole face for precise accuracy",
                    (350, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Press ESC to quit",
                    (1300, 850), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        if not ret or (cv2.waitKey(1) & 0xFF == 27):
            cap.release()
            cv2.destroyAllWindows()
            on_loop = False
        cv2.imshow('Gesture Classification Window', frame)





if __name__ == "__main__":

    model_path = "fer_resnet50train.pth"
    model = load_model(model_path)
    set_folder()
    print("Select mode:")
    print("1: Capture Photos")
    print("2: Classify Real Time Face Gestures")
    mode = input("Enter 1 or 2: ")
    if mode == "1":
        total_people = int(input("Enter number of people to capture: "))
        for i in range(total_people):
            capture_expressions(i)
    elif mode == "2":
        classify_emotions(model)
    else:
        print("Invalid choice. Exiting.")
