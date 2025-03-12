import torch
import numpy as np
import matplotlib.pyplot as plt
import gc

from torchvision import datasets, transforms

from torch.utils.data import DataLoader, SubsetRandomSampler

from torch import nn, optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

train_transforms = transforms.Compose([
    transforms.Resize((227,227)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
   transforms.ToTensor(),
   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
test_transforms = transforms.Compose([
    transforms.Resize((227,227)),
   transforms.ToTensor(),
   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=train_transforms)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=test_transforms)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

validation_ratio = 0.2
train_size = len(train_data)
print(f"Training data size: {train_size}")

#We will generate indices and shuffle them to split train data
indices = list(range(train_size))
np.random.shuffle(indices) #This is an in-place operation?!
split = int(np.floor(validation_ratio*train_size))
validation_indices = indices[:split]
train_indices = indices[split:]
print(f"Splitting on {split}, validation size: {len(validation_indices)} training size: {len(train_indices)}")

#These indices will be used in a sampler to select these from the training data
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(list(range(len(test_data))))

train_loader = DataLoader(train_data, batch_size=64, sampler=train_sampler)
validation_loader = DataLoader(train_data, batch_size=64, sampler=validation_sampler)
test_loader = DataLoader(test_data, batch_size=64, sampler=test_sampler)

images, labels = next(iter(train_loader))
writer.add_images("sample_images", images[1:17], global_step=0)

class Net(nn.Module):
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

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        #ConvNets don't care about input width/height, only depth
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

class CleanAlexNet(nn.Module):
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

model = CleanAlexNet()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

def model_train(model, n_epochs):
    validation_loss_min = np.inf

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        validation_loss = 0.0

        #Begin training
        model.train()
        for data, target in train_loader:
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                model = model.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Memory/allocated", torch.cuda.memory_allocated(), epoch)
            writer.add_scalar("Memory/reserved", torch.cuda.memory_reserved(), epoch)
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        #Begin validation
        model.eval()
        for data, target in validation_loader:
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
                model = model.cuda()
            output = model(data)
            loss = criterion(output, target)
            writer.add_scalar("Loss/validation", loss, epoch)
            validation_loss += loss.item() * data.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        validation_loss = validation_loss / len(validation_loader.dataset)
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.2f} \tValidation Loss: {validation_loss:.2f}')
        writer.add_scalars("Combined loss", {
            "train": train_loss
            , "validation": validation_loss
            }, epoch)

        if validation_loss < validation_loss_min:
            print(f"Validation loss reduced from {validation_loss_min:.2f} to {validation_loss:.2f}. Saving model")
            torch.save(model.state_dict(), 'models/model_cifar.pt')
            validation_loss_min = validation_loss

def model_eval():
    model = CleanAlexNet()
    model.load_state_dict(torch.load('models/model_cifar.pt'))

    test_loss = 0.0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    n_runs = 0 

    model.eval()
    for data, target in test_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
            model = model.cuda()
        output = model(data)
        loss = criterion(output, target)
        test_loss = test_loss + (loss.item() * data.size(0))
        _, pred = torch.max(output, dim=1)
        correct = pred.eq(target.data)
        writer.add_scalar("Eval/memory_allocated"   , torch.cuda.memory_allocated() , n_runs*data.size(0))
        writer.add_scalar("Eval/memory_reserved"    , torch.cuda.memory_reserved()  , n_runs*data.size(0))
        writer.add_scalar("Eval/loss"               , loss, n_runs*data.size(0))
        writer.add_scalar("Eval/accuracy"           , torch.sum(correct).item()/data.size(0), n_runs*data.size(0))
        writer.add_scalar("Eval/n_runs"             , n_runs                        , n_runs*data.size(0))
        writer.add_scalar("Eval/batch_size"         , data.size(0)                  , n_runs*data.size(0))
        # For some reason, this is only 1/4 of the actual number of runs
        n_runs += 1
  
        for i in range(data.size(0)): #Data size 0 indicates batch size
            label = target.data[i]
            class_correct[label] += correct[i].item() #Apparently you can add True/False as if they are 1 or 0
            class_total[label] += 1
           
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.6f}')

    for i in range(10):
        if class_total[i] > 0:
            print(f'Test Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]}% ({np.sum(class_correct[i])}/{np.sum(class_total[i])})')
        else:
            print(f'Test Accuracy of {classes[i]}: N/A (no training examples)')

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    
    writer.add_text("accuracy", (100. * np.sum(class_correct) / np.sum(class_total)).__str__())