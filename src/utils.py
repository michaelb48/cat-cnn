import math
import os
import torch
import uuid

def save_model(config, model, optimizer, criterion):
    torch.save({
                'config': config,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join('weights','catCnn'+str(uuid.uuid4())))


# calcualtes padding given input dim, output dim, kernel size and stride
# when stride is 1 input and output dimension are identical
# when stride is 2 output dimension is half of input
# based on the formula from pytorch docs https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
def calculate_padding(kernel, stride):
    if stride == 1:
        padding = (kernel - 1) // 2
    elif stride == 2:
        padding = math.ceil((kernel - 2) / 2)
    else:
        raise ValueError("This function only supports stride of 1 or 2.")
    return padding