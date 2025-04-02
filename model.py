import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple CNN architecture (adapt as needed, e.g., use torchvision.models.resnet18)
# This is a basic example; a ResNet or VGG variant would likely perform better.
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        # Input: 1x48x48 grayscale image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2) # -> 64x24x24
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2) # -> 128x12x12
        self.dropout2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2) # -> 256x6x6
        self.dropout3 = nn.Dropout(0.25)

        # Flatten -> 256 * 6 * 6 = 9216
        self.fc1 = nn.Linear(9216, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1) # Flatten

        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = self.dropout4(x)
        x = F.relu(self.fc_bn2(self.fc2(x)))
        x = self.dropout5(x)
        x = self.fc3(x) # Output layer (logits)
        return x

def load_base_model(path="base_model.pth", num_classes=7, device='cpu'):
    """Loads the base model structure and weights."""
    model = SimpleCNN(num_classes=num_classes)
    try:
        # Load weights if path exists and is valid
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded base model weights from {path}")
    except FileNotFoundError:
        print(f"Warning: Base model weights not found at {path}. Using initialized model.")
    except Exception as e:
        print(f"Warning: Error loading base model weights from {path}: {e}. Using initialized model.")
    model.to(device)
    return model

def load_personalized_model(path="personalized_model.pth", num_classes=7, device='cpu'):
    """Loads the fine-tuned personalized model structure and weights."""
    model = SimpleCNN(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded personalized model weights from {path}")
    except FileNotFoundError:
        print(f"Error: Personalized model weights not found at {path}. Run calibration first.")
        return None
    except Exception as e:
        print(f"Error loading personalized model weights from {path}: {e}")
        return None
    model.to(device)
    return model