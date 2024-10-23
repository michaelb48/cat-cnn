import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
from model import CatCnn
from torchinfo import summary
from src.utils import save_model

# configuration dependent hyperparameters
config = {
    'batch_size': 32,
    'epochs': 20,
    'sample':7*1300,
    'learning_rate': 3e-4,
    'random_seed': 42
}

# transforms to encode the images and their labels
train_data_transform = transforms.Compose([
    transforms.RandomResizedCrop((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

eval_data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                         std=[0.2675, 0.2565, 0.2761])
])

label_transform = transforms.Lambda(
    lambda y: torch.zeros(7, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

# these are the paths to the directories holding our training and evaluation data
TRAIN_DIR = os.path.join(os.getcwd(), '..', 'resources', 'train')
EVAL_DIR = os.path.join(os.getcwd(), '..', 'resources', 'val')

# use our custom dataset with the dataloader to create an iterable
training_data = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_data_transform,
    target_transform=label_transform
)
pre_val_data = datasets.ImageFolder(
    root=EVAL_DIR,
    transform=eval_data_transform
)
generator = torch.Generator().manual_seed(config.get('random_seed'))
validation_data,test_data = random_split(pre_val_data, [0.5, 0.5], generator=generator)

train_dataloader = DataLoader(training_data, batch_size=config.get('batch_size'), shuffle=True)

validation_dataloader = DataLoader(validation_data, batch_size=config.get('batch_size'), shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=config.get('batch_size'), shuffle=True)

# check for available devices
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# move the model to our device
model = CatCnn()

# optimizer
optim = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate'))

# loss criterion
criterion = torch.nn.CrossEntropyLoss(reduction='mean')

# Training & Validation
def training(model, train_loader, validation_loader, optimizer, criterion, device):
    model.to(device)
    for epoch in range(config["epochs"]):
        model.train()
        metrics = {}
        for batch, (x,y) in enumerate(train_loader):
            # zero the gradients before each loop
            optimizer.zero_grad()
            # move data to gpu
            x = x.to(device)
            y = y.to(device)

            # Feed input data into the model
            pred = model(x)

            # feed the values predicted with the model into cross entropy loss
            # the crossentropy takes care of normalization for us
            loss = criterion(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # check if 100 iteration are hit to log to wandb
            # log only the loss at this point
            if batch % 100 == 0:
                loss, current = loss.item(), batch * config.get("batch_size") + len(x)
                print(f"current loss: {loss:>7f}  [{current:>5d}/{config.get("sample"):>5d}]")

            # check if maximum sample size per epoch is reached
            if (1+batch) * config.get('batch_size') > config.get('sample'):
                break

        # log metrics somehow according to wandb and validate the model
        metrics = validate_model(model, validation_loader, epoch)

    #save the model after the last epoch
    save_model(config, model, optimizer,criterion)

# calculates whether your prediction was among the top k classes for a specific example
def calculate_accuracy(pred, target, k=1):
    top_k_preds = torch.topk(pred, k=k, dim=1).indices  # Shape: (batch_size, k)
    correct = top_k_preds.eq(target.view(-1,1).expand_as(top_k_preds))
    correct_count = correct.sum().item()
    return correct_count/config.get("batch_size")


# Validation loop
def validate_model(model, val_loader,epoch):
    # change model mode because of batch normalization
    model.eval()

    total_num = len(val_loader)
    top_1 = 0
    top_3 = 0
    top_5 = 0

    with torch.no_grad():
        for batch, (x,y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            top_1 += calculate_accuracy(pred, y,1)
            top_3 += calculate_accuracy(pred, y,3)
            top_5 += calculate_accuracy(pred, y,5)
    metrics = {
        'top-1': top_1 / total_num,
        'top-3': top_3 / total_num,
        'top-5': top_5 / total_num
    }

    # Print metrics

    print(f"Evaluation Metrics in epoch {epoch}:\n")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print('\n')
    return metrics



if __name__ == '__main__':
    #summary(model, input_size=(config.get('batch_size'), 3, 224, 224))
    training(model, train_dataloader,validation_dataloader,optim,criterion,device)

