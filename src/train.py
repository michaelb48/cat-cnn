import torch
import numpy as np
from torchvision.io import read_image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from model import CatCnn
from torchinfo import summary

from src.utils import EarlyStopping

# labels map
labels_map = {
    'eqyptian_cat': 0,
    'jaguar': 1,
    'lynx': 2,
    'persian_cat': 3,
    'siamese_cat': 4,
    'tabby_cat': 5,
    'tiger': 6,
}

# configuration dependent hyperparameters
config = {
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 3e-4,
    'patience': 10,
    'delta': 0.1,
}

# transforms to encode the images and their labels
train_data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

eval_data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

label_transform = transforms.Lambda(
    lambda y: torch.zeros(7, dtype=torch.float).scatter_(0, torch.tensor(labels_map[y]), value=1))

# these are the paths to the directories holding our training and evaluation data
TRAIN_DIR = os.path.join(os.getcwd(), '..', 'resources', 'train')
EVAL_DIR = os.path.join(os.getcwd(), '..', 'resources', 'val')

# use our custom dataset with the dataloader to create an iterable
training_data = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_data_transform,
    target_transform=label_transform
)
test_data = datasets.ImageFolder(
    root=EVAL_DIR,
    transform=eval_data_transform,
    target_transform=label_transform
)
train_dataloader = DataLoader(training_data, batch_size=config.get('batch_size'), shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=config.get('batch_size'), shuffle=False)

# check for available devices
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# move the model to our device
model = CatCnn().to(device)

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate'))

# loss criterion
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

# configure early stopping
early_stopping = EarlyStopping(patience=config.get('patience'), delta=config.get('delta'))