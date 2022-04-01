import torch.functional
import torch.nn as nn
import torch.nn.functional as F


class CNN_BASE(nn.Module):

    def __init__(self, activation='ReLu', pool='Max'):
        super(CNN_BASE, self).__init__()

        self.pool22 = nn.MaxPool2d((2, 2), ceil_mode=True)
        self.activation = F.relu

        if activation == 'PReLu':
            self.activation = F.prelu
        elif activation == 'Tanh':
            self.activation = F.tanh

        if pool == 'Avg':
            self.pool22 = nn.AvgPool2d((2, 2), ceil_mode=True)
        elif pool == 'MaxUn':
            self.pool22 = nn.MaxUnpool2d((2, 2))

        self.conv1_1 = nn.Conv2d(1, 16, (10, 2))
        self.pd1_1 = nn.ZeroPad2d([0, 1, 9, 0])
        self.nm1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(1, 16, (2, 8))
        self.pd1_2 = nn.ZeroPad2d([0, 7, 1, 0])
        self.nm1_2 = nn.BatchNorm2d(16)

        # concatenate
        # rule

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.pd2 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm2 = nn.BatchNorm2d(32)
        # rule

        self.conv3 = nn.Conv2d(32, 48, (3, 3))
        self.pd3 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm3 = nn.BatchNorm2d(48)
        # rule
        # Pool

        self.conv4 = nn.Conv2d(48, 64, (3, 3))
        self.pd4 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm4 = nn.BatchNorm2d(64)
        # rule
        # Pool

        self.conv5 = nn.Conv2d(64, 128, (3, 3))
        self.pd5 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm5 = nn.BatchNorm2d(128)
        # rule

        self.classifier = nn.Linear(65536, 4)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(-1, 1, 128, 32)
        xx = self.conv1_1(x)
        xx = self.pd1_1(xx)
        xx = self.nm1_1(xx)
        xx = self.activation(xx)

        yy = self.conv1_2(x)
        yy = self.pd1_2(yy)
        yy = self.nm1_2(yy)
        yy = self.activation(yy)

        x = torch.cat([xx, yy], dim=2)

        x = self.conv2(x)
        x = self.pd2(x)
        x = self.nm2(x)
        x = self.activation(x)
        x = self.pool22(x)

        x = self.conv3(x)
        x = self.pd3(x)
        x = self.nm3(x)
        x = self.activation(x)
        x = self.pool22(x)

        x = self.conv4(x)
        x = self.pd4(x)
        x = self.nm4(x)
        x = self.activation(x)

        x = self.conv5(x)
        x = self.pd5(x)
        x = self.nm5(x)

        x = self.activation(x)
        x = x.view(b, -1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    import numpy as np

    model = CNN_BASE()
    np.random.seed(1)
    test = np.random.random((16, 1, 128, 32)).astype(np.float32)
    x = torch.from_numpy(test)
    print(model(x))
