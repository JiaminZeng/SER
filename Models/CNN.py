import torch.functional
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 16, (10, 2))
        self.pd1_1 = nn.ZeroPad2d([0, 1, 9, 0])
        self.conv1_2 = nn.Conv2d(1, 16, (2, 8))
        self.pd1_2 = nn.ZeroPad2d([0, 7, 1, 0])
        # concatenate
        # rule

        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.pd2 = nn.ZeroPad2d([0, 2, 2, 0])
        # rule
        self.maxPool22 = nn.MaxPool2d((2, 2), ceil_mode=True)

        self.conv3 = nn.Conv2d(32, 48, (3, 3))
        self.pd3 = nn.ZeroPad2d([0, 2, 2, 0])
        # rule
        # maxPool

        self.conv4 = nn.Conv2d(48, 64, (3, 3))
        self.pd4 = nn.ZeroPad2d([0, 2, 2, 0])
        # rule
        # maxPool

        self.conv5 = nn.Conv2d(64, 80, (3, 3))
        self.pd5 = nn.ZeroPad2d([0, 2, 2, 0])
        # rule

        self.nm = nn.BatchNorm2d

        self.classifier = nn.Sequential(
            nn.Linear(80 * 16 * 8, 256),
            nn.Dropout(p=0.5),
            nn.PReLU(256),
            nn.Linear(256, 4))

    def forward(self, x):
        b = x.shape[0]
        x = x.view(-1, 1, 128, 64)
        xx = self.conv1_1(x)
        xx = self.pd1_1(xx)
        yy = self.conv1_2(x)
        yy = self.pd1_2(yy)
        x = torch.cat([xx, yy], dim=1)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pd2(x)
        x = self.maxPool22(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.pd3(x)
        x = self.maxPool22(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.pd4(x)
        x = self.maxPool22(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.pd5(x)
        x = F.relu(x)

        x = x.view(b, -1)
        x = self.classifier(x)
        return x

# model = ConvNet()
# tr = torch.randn([4, 1, 128, 64])
# print(model(tr).shape)
