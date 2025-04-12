import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
#model using resnet and lstm
class FacialRecognitionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialRecognitionModel, self).__init__()

        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Output shape: (batch_size, 2048, 1, 1)


        self.lstm = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2, batch_first=True)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.resnet(x)
        x = x.view(batch_size, 1, -1)


        x, _ = self.lstm(x)
        x = x[:, -1, :]


        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
