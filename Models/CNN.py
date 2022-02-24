import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 1 1 384 127
        self.conv1 = nn.Conv2d(1, 5, 5)  # 384-18 127-18
        self.pool = nn.MaxPool2d(8, 4)  # 366-18 109-18
        self.conv2 = nn.Conv2d(5, 10, 5)  # 330 91
        self.fc1 = nn.Linear(1050, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1050)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
