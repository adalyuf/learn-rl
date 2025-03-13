from torch import nn
import torch.nn.functional as F

class ExampleNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(64*4*4, 500) #depth of prior layer, times width/height of input ... 32/2 = 16/2 = 8/2 = 4
    self.fc2 = nn.Linear(500, 10)
    self.dropout = nn.Dropout(0.25)

  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)
    x = self.pool(x)
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = x.view(-1, 64*4*4)
    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x