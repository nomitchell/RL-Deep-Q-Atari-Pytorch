import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_actions=4):
        super(Model, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        # Auto determine input size
        self.dummy_input = torch.zeros(1, 4, 84, 84)
        with torch.no_grad():
            conv_out_size = self.conv_layers(self.dummy_input).view(1, -1).size(1) 

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
