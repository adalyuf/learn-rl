import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
import os 
import wandb 
import argparse

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

validation_ratio = 0.1 #10% of the training data will be used for validation
batch_size = 128
num_workers = 8
dataset_name = 'imagenet'
data_dir = '/home/adaly/pytorch_testing/data/imagenet/'

train_transforms = transforms.Compose([
  transforms.Resize((227,227)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(30),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
  transforms.Resize((227,227)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data  = datasets.ImageFolder(root=f'{data_dir}/train',  transform=train_transforms)
test_data   = datasets.ImageFolder(root=f'{data_dir}/val',  transform=test_transforms)

train_size = len(train_data)
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

train_loader        = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
validation_loader   = DataLoader(train_data, batch_size=batch_size, sampler=validation_sampler, num_workers=num_workers, pin_memory=True)
test_loader         = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)

if not os.path.exists(f'models/{dataset_name}'):
    os.mkdir(f'models/{dataset_name}')
if not os.path.exists(f'runs/{dataset_name}'):
    os.mkdir(f'runs/{dataset_name}')
if not os.path.exists(f'wandb/{dataset_name}'):
    os.mkdir(f'wandb/{dataset_name}')

config_defaults = SimpleNamespace(
    epochs=5,
    learning_rate=0.01,
    model_choice="AlexNet",
    wandb_project="alexnet-opts",
    bn_before_relu=True, # This is a hyperparameter for the NewAlexNet model
    dropout_rate=0.1,
    initialization='xavier',
    momentum=0.9,
    learning_rate_decay=0.9,
)

sweep_config = {
    'method': 'grid', # 'random', 'grid', 'bayes'
    'name': 'Imagenet Test Sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'Eval/accuracy'
    },
    'parameters': {
        'epochs': {
            'values': [10]
        },
        'learning_rate': {
            'values': [0.01]
        },
        'model_choice': {
            'values': ['AlexNet']
        },
        'bn_before_relu': {
            'values': [True]
        },
        'dropout_rate': {
            'values': [0.0]
        },
        'initialization': {
            'values': ['xavier'] # 'xavier', 'kaiming', 'orthogonal'
        },
        'momentum': {
            'values': [0.9]
        },
        'learning_rate_decay': {
            'values': [0.9]
        }
    }
}
def sweep():
    sweep_id = wandb.sweep(sweep_config, project=config_defaults.wandb_project)
    wandb.agent(sweep_id, function=model_train)

for model in sweep_config['parameters']['model_choice']['values']:
    if not os.path.exists(f'wandb/{dataset_name}/{model}'):
        os.mkdir(f'wandb/{dataset_name}/{model}')
    if not os.path.exists(f'models/{dataset_name}/{model}'):
        os.mkdir(f'models/{dataset_name}/{model}')
    if not os.path.exists(f'runs/{dataset_name}/{model}'):
        os.mkdir(f'runs/{dataset_name}/{model}')

for data, target in train_loader:
    assert target.min() >= 0, "Labels contain negative values!"
    assert target.max() < 1000, "Labels exceed the number of classes!"
    assert target.dtype == torch.long, "Labels are not of type long!"
    if torch.cuda.is_available():
        target = target.cuda()
        data = data.cuda()
    break

def model_train(config=config_defaults, use_last=False):
    with wandb.init(project=config.wandb_project, config=config, dir=f'wandb/{dataset_name}/{config.model_choice}'):
        config = wandb.config #Fetch the config from wandb
        
        model_choice = config.model_choice
        epochs = config.epochs
        lr = config.learning_rate
        
        if model_choice == "ExampleNet":
            model = ExampleNet()
        elif model_choice == "AlexNet":
            model = AlexNet(num_classes=1000)
        elif model_choice == "NewAlexNet":
            model = NewAlexNet(bn_before_relu=config.bn_before_relu, num_classes=1000, dropout_rate=config.dropout_rate)
            print(f"Using bn_before_relu: {config.bn_before_relu}")
        else:
            raise ValueError(f"Model choice {model_choice} not recognized")
        if use_last:  
            model.load_state_dict(torch.load(f'models/{dataset_name}/{model_choice}/model.pt'))
            print(f"Loading last model from models/{dataset_name}/{model_choice}/model.pt")
        print(f"Using config: {config}")

        def init_weights(module):
            if config.initialization == 'xavier':
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
            elif config.initialization == 'kaiming':
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif config.initialization == 'orthogonal':
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight)
            # Initialize biases to zero for all layers
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        model.apply(init_weights)

        images, labels = next(iter(train_loader))
        sample_images = []
        for i in range(1,17):
            image = wandb.Image(images[i], caption=labels[i])
            sample_images.append(image)
        wandb.log({"sample_images": sample_images})

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config.momentum)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.learning_rate_decay)

        validation_loss_min = np.inf

        print(f"Training {model_choice} for {epochs} epochs with learning rate {lr}")
        for epoch in range(1, epochs+1):
            train_loss = 0.0
            validation_loss = 0.0
            images_processed = 0
            batches_processed = 0
            print(f"CUDA Available: {torch.cuda.is_available()}")
            
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
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                images_processed += data.size(0)
                batches_processed += 1
                if batches_processed % 10 == 0: #Log every 10 batches
                    wandb.log({
                        "train/batch_loss"         : loss.item(),
                        "train/batch_accuracy"     : (torch.sum(torch.argmax(output, dim=1).eq(target.data)).item()/data.size(0)),
                        "progress/batches_processed"  : batches_processed,
                        "progress/batch_size"         : data.size(0),
                        "progress/images_processed"   : images_processed
                    })

            #Begin validation
            model.eval()
            for data, target in validation_loader:
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                    model = model.cuda()
                output = model(data)
                loss = criterion(output, target)
                validation_loss += loss.item() * data.size(0)
                images_processed += data.size(0)
                batches_processed += 1
                if batches_processed % 10 == 0: #Log every 10 batches
                    wandb.log({
                        "train/batch_val_loss"     : loss.item(),
                        "train/batch_val_accuracy" : (torch.sum(torch.argmax(output, dim=1).eq(target.data)).item()/data.size(0)),
                        "progress/batches_processed"  : batches_processed,
                        "progress/batch_size"         : data.size(0),
                        "progress/images_processed"   : images_processed
                    })
            
            scheduler.step() #Decay the learning rate after each epoch

            train_loss = train_loss / len(train_loader.dataset)
            validation_loss = validation_loss / len(validation_loader.dataset)
            print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.2f} \tValidation Loss: {validation_loss:.2f}')
            wandb.log({
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "epoch": epoch})
    
            if validation_loss < validation_loss_min:
                print(f"Validation loss reduced from {validation_loss_min:.2f} to {validation_loss:.2f}. Saving model")
                torch.save(model.state_dict(), f'models/{dataset_name}/{model_choice}/model.pt')
                validation_loss_min = validation_loss
                
        print("Training complete. Evaluating model...")
        model_eval(model_choice, epochs, lr, config.bn_before_relu, config.dropout_rate)

def model_eval(model_choice, epochs=None, lr=None, bn_before_relu=True, dropout_rate=0.25):
    if model_choice == "ExampleNet":
        model = ExampleNet()
    elif model_choice == "AlexNet":
        model = AlexNet(num_classes=1000)
    elif model_choice == "NewAlexNet":
        model = NewAlexNet(bn_before_relu=bn_before_relu, num_classes=1000, dropout_rate=dropout_rate)
        print(f"Using bn_before_relu: {bn_before_relu}")
    else:
        raise ValueError(f"Model choice {model_choice} not recognized")
    print(f"Using model: {model_choice}")
    wandb.init(project="alexnet-opts", dir=f'wandb/{dataset_name}/{model_choice}')

    model.load_state_dict(torch.load(f'models/{dataset_name}/{model_choice}/model.pt'))
    criterion = nn.CrossEntropyLoss()

    test_loss = 0.0
    class_correct = list(0.0 for i in range(1000))
    class_total = list(0.0 for i in range(1000))
    images_processed = 0
    batches_processed = 0

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
        images_processed += data.size(0)
        batches_processed += 1
        if batches_processed == 1:
            sample_images = data[:16].cpu().numpy() #Get first 16 images for logging
            sample_images = sample_images.transpose((0, 2, 3, 1))  # Convert from [N, C, H, W] to [N, H, W, C]
            sample_images = (sample_images * 0.5 + 0.5) * 255  # Denormalize and scale to [0, 255]
            sample_images = sample_images.astype(np.uint8)  # Convert to uint8 for visualization
            sample_images = [wandb.Image(img, caption=f'Pred: {pred[i]}\nCorrect: {target[i]}') for i, img in enumerate(sample_images)]
            wandb.log({"Test Images": sample_images})
        if batches_processed % 10 == 0: #Log every 10 batches
            wandb.log({
                "Eval/batch_loss"         : loss.item(),
                "Eval/batch_accuracy"     : torch.sum(correct).item()/data.size(0),
                "progress/images_processed"   : images_processed,
                "progress/batches_processed"  : batches_processed,
                "progress/batch_size"         : data.size(0)
            })
  
        for i in range(data.size(0)): #Data size 0 indicates batch size
            label = target.data[i]
            class_correct[label] += correct[i].item() #Apparently you can add True/False as if they are 1 or 0
            class_total[label] += 1
           
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.6f}')

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    
    wandb.log({
        "Eval/test_loss": test_loss,
        "Eval/accuracy": (100. * np.sum(class_correct) / np.sum(class_total))
    })
    
    wandb.finish()