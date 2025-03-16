import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
import os 
import wandb 

# PyTorch imports
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
# Importing custom models
from class_example import ExampleNet
from original_alexnet import AlexNet
from new_alexnet import NewAlexNet

from types import SimpleNamespace


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
batch_size = 64
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
train_sampler       = SubsetRandomSampler(train_indices)
validation_sampler  = SubsetRandomSampler(validation_indices)
test_sampler        = SubsetRandomSampler(list(range(len(test_data))))

train_loader        = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
validation_loader   = DataLoader(train_data, batch_size=batch_size, sampler=validation_sampler)
test_loader         = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)

config_defaults = SimpleNamespace(
    epochs=5,
    learning_rate=2e-3,
    model_choice="AlexNet",
    wandb_project="alexnet-opts",
)


def model_train(config=config_defaults):
    wandb.tensorboard.patch(root_logdir="runs")
    wandb.init(project=config.wandb_project, config=config, sync_tensorboard=True)
    config = wandb.config #Fetch the config from wandb
    writer = SummaryWriter(log_dir=f'runs/{config.model_choice}')
    
    model_choice = config.model_choice
    epochs = config.epochs
    lr = config.learning_rate
    
    if model_choice == "ExampleNet":
        model = ExampleNet()
    elif model_choice == "AlexNet":
        model = AlexNet()
    elif model_choice == "NewAlexNet":
        model = NewAlexNet()
    else:
        raise ValueError(f"Model choice {model_choice} not recognized")
    print(f"Using config: {config}")

    if not os.path.exists(f'models/{model_choice}'):
        os.mkdir(f'models/{model_choice}')
    if not os.path.exists(f'runs/{model_choice}'):
        os.mkdir(f'runs/{model_choice}')

    images, labels = next(iter(train_loader))
    writer.add_images("sample_images", images[1:17], global_step=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    validation_loss_min = np.inf

    for epoch in range(1, epochs+1):
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
            torch.save(model.state_dict(), f'models/{model_choice}/model_cifar.pt')
            validation_loss_min = validation_loss
            
    print("Training complete. Evaluating model...")
    model_eval(model_choice, epochs, lr)

def model_eval(model_choice, epochs=None, lr=None):
    if model_choice == "ExampleNet":
        model = ExampleNet()
    elif model_choice == "AlexNet":
        model = AlexNet()
    elif model_choice == "NewAlexNet":
        model = NewAlexNet()
    else:
        raise ValueError(f"Model choice {model_choice} not recognized")
    print(f"Using model: {model_choice}")
    writer = SummaryWriter(log_dir=f'runs/{model_choice}')
    wandb.init(project="alexnet-opts", sync_tensorboard=True)

    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(f'models/{model_choice}/model_cifar.pt'))

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
    writer.add_hparams({
        "model": model_choice,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs
    }, {
        "hparam/accuracy": (100. * np.sum(class_correct) / np.sum(class_total))
    })
    writer.close()
    wandb.finish()