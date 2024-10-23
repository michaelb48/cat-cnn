from torch import nn
from utils import calculate_padding

class ResidualLayer(nn.Module):
    def __init__(self):
        super(ResidualLayer,self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, bias=False, padding=calculate_padding(5,1)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(init=0.2),
        )
        self.hidden_unit_F = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, bias=True, padding=calculate_padding(3,2)),
            nn.PReLU(init=0.2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=False, padding=calculate_padding(3,1)),
            nn.BatchNorm2d(num_features=32),
            nn.PReLU(init=0.2),
        )
        self.residual_connection_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1,bias =True, stride=2,padding=calculate_padding(1,2))


    def forward(self, x):
        y = self.shared_layers(x)
        main = self.hidden_unit_F(y)
        residual = self.residual_connection_conv(y)
        return main + residual