from torch import nn
from utils import calculate_padding
from residual import ResidualLayer

class CatCnn(nn.Module):
    def __init__(self):
        super(CatCnn,self).__init__()
        self.flatten = nn.Flatten()

        self.hidden_unit_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, bias=False, padding=calculate_padding(7,2)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(init=0.2),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=calculate_padding(3,2)),
        )
        self.hidden_unit_2 = ResidualLayer()
        self.hidden_unit_3 = ResidualLayer()
        self.hidden_unit_4 = ResidualLayer()
        self.pooling = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=32,out_features=7)


    def forward(self, x):
        print(x.shape)
        logits = self.hidden_unit_1(x)
        logits = self.hidden_unit_2(logits)
        logits = self.hidden_unit_3(logits)
        logits = self.hidden_unit_4(logits)
        pooled_val = self.pooling(logits)
        flattened_val = self.flatten(pooled_val)
        logits = self.fc(flattened_val)
        return logits