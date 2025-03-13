from torch import nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        #AlexNet expects 3x227x227 input
        #Convnet order is depth in, depth out, kernel size, stride, padding
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #Conv1: 55x55x96 Pool: 27x27x96
        x = self.pool(F.relu(self.conv2(x))) #13x13x256
        x = F.relu(self.conv3(x)) #13x13x384
        x = F.relu(self.conv4(x)) #13x13x384
        x = self.pool(F.relu(self.conv5(x))) #13x13x256 -> 6x6x256
        x = x.view(-1, 6*6*256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x