from torch import nn
import torch.nn.functional as F

class NewAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        #ConvNets don't care about input width/height, only depth
        #AlexNet expects 3x227x227 input
        #Convnet order is depth in, depth out, kernel size, stride, padding
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.fc6 = nn.Linear(256*6*6, 4096)
        self.bn6 = nn.BatchNorm1d(4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        self.fc8 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) #Conv1: 55x55x96 Pool: 27x27x96
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) #13x13x256
        x = F.relu(self.bn3(self.conv3(x))) #13x13x384
        x = F.relu(self.bn4(self.conv4(x))) #13x13x384
        x = self.pool(F.relu(self.bn5(self.conv5(x)))) #13x13x256 -> 6x6x256
        x = x.view(-1, 6*6*256)
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.fc8(x)
        return x