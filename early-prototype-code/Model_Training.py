#import Starting_Model as train_model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from torchvision.models import ResNet50_Weights
from Model import FacialRecognitionModel
data_dir = "dataset/archive/train"
personalized_dir = "personalized_dataset"



def load_fer_loaders(data_dir):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isModel = True
    model = FacialRecognitionModel(num_classes=7).to(device)
    if isModel:
        model = FacialRecognitionModel(num_classes=7).to(device)
    else:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 7)
    return train_loader, val_loader, model

def get_train_loading(data_dir):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    return train_loader



def train_model(model, train_loader, num_epochs=5):
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history[f'train_loss'].append(epoch_loss)
        history[f'train_acc'].append(epoch_acc)

        print(f'Epoch: {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return history
def validate_model(model, val_loader, num_epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []

    accurate = 0
    total = 0
    total_labels = []
    total_predictions = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total_labels.extend(labels.cpu().numpy())
            total_predictions.extend(predicted.cpu().numpy())

            accurate += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * accurate / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

def run_training(data_path):
    train_loader, val_loader, model = load_fer_loaders(data_path)
    epochs = 10
    history = train_model(model, train_loader, num_epochs=epochs)
    saved_name = f"fer_resent{epochs}.pth"
    torch.save(model.state_dict(), saved_name)
    validate_model(model, val_loader, num_epochs=epochs)
    print("saved model")

def personalized_training(data_path):
    train_loader = get_train_loading(data_path)
    model = FacialRecognitionModel(num_classes=7)
    epochs = 10
    model.load_state_dict(torch.load(f"fer_resent{epochs}.pth"))
    history = train_model(model, train_loader, num_epochs=10)
    saved_name = f"personalized_fer_resent{epochs}.pth"
    torch.save(model.state_dict(), saved_name)
#add compare to old val data
    print("saved model")




def choice_training():
    choice = input("Select dataset for training: \n1: FER+ ResNet50 Dataset\n2: Personalized Dataset\nEnter choice (1 or 2): ")

    if choice == "1":
        data_path = "dataset/archive/train"
        run_training(data_path)
    elif choice == "2":
        data_path = "personalized_dataset"
        personalized_training(data_path)
    else:
        print("Invalid")
        exit()


if __name__ == "__main__":
    choice_training()