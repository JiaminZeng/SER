import area_attention
import torch.functional
import torch.nn as nn
import torch.nn.functional as F


class ACCN_Area_Multi(nn.Module):

    def __init__(self):
        super(ACCN_Area_Multi, self).__init__()
        self.name = "ACCN_Area_Multi"
        self.conv1_1 = nn.Conv2d(1, 8, (10, 2))
        self.pd1_1 = nn.ZeroPad2d([0, 1, 9, 0])
        self.nm1_1 = nn.BatchNorm2d(8)
        self.conv1_2 = nn.Conv2d(1, 8, (2, 8))
        self.pd1_2 = nn.ZeroPad2d([0, 7, 1, 0])
        self.nm1_2 = nn.BatchNorm2d(8)

        # concatenate
        # rule

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.pd2 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm2 = nn.BatchNorm2d(32)
        # rule
        self.maxPool22 = nn.MaxPool2d((2, 2), ceil_mode=True)

        self.conv3 = nn.Conv2d(32, 48, (3, 3))
        self.pd3 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm3 = nn.BatchNorm2d(48)
        # rule
        # maxPool

        self.conv4 = nn.Conv2d(48, 64, (3, 3))
        self.pd4 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm4 = nn.BatchNorm2d(64)
        # rule
        # maxPool

        self.conv5 = nn.Conv2d(64, 80, (3, 3))
        self.pd5 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm5 = nn.BatchNorm2d(80)
        # rule

        self.single = area_attention.area_attention.AreaAttention(key_query_size=4, max_area_height=3, max_area_width=3,
                                                                  memory_width=4, memory_height=4)
        self.attn = area_attention.multi_head_area_attention.MultiHeadAreaAttention(self.single, 4, 4, 4, 4, 4)

        self.classifier = nn.Sequential(
            nn.Linear(5120, 512),
            nn.Dropout(p=0.5),
            nn.PReLU(512),
            nn.Linear(512, 4))

        self.mul = torch.mul
        self.softmax = torch.softmax
        self.cat = torch.cat

    def forward(self, x):
        b = x.shape[0]
        x = x.view(-1, 1, 128, 32)
        xx = self.conv1_1(x)
        xx = self.pd1_1(xx)
        xx = self.nm1_1(xx)

        yy = self.conv1_2(x)
        yy = self.pd1_2(yy)
        yy = self.nm1_2(yy)

        x = torch.cat([xx, yy], dim=1)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pd2(x)
        x = self.nm2(x)
        x = self.maxPool22(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.pd3(x)
        x = self.nm3(x)
        x = self.maxPool22(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.pd4(x)
        x = self.nm4(x)
        x = self.maxPool22(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.pd5(x)
        x = self.nm5(x)
        x = F.relu(x)

        x = x.view(-1, 16, 4)
        Q = x
        K = x
        V = x
        # print(x.shape)
        attn = self.attn(Q, K, V)
        x = attn.view(b, -1, 16, 4)
        x = F.relu(x)
        x = x.view(b, -1)
        x = self.classifier(x)
        return x


class ACCN_Area(nn.Module):


    def __init__(self):
        super(ACCN_Area, self).__init__()
        self.name = "ACCN_Area"
        self.conv1_1 = nn.Conv2d(1, 16, (10, 2))
        self.pd1_1 = nn.ZeroPad2d([0, 1, 9, 0])
        self.nm1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(1, 16, (2, 8))
        self.pd1_2 = nn.ZeroPad2d([0, 7, 1, 0])
        self.nm1_2 = nn.BatchNorm2d(16)

        # concatenate
        # rule

        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.pd2 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm2 = nn.BatchNorm2d(32)
        # rule
        self.maxPool22 = nn.MaxPool2d((2, 2), ceil_mode=True)

        self.conv3 = nn.Conv2d(32, 48, (3, 3))
        self.pd3 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm3 = nn.BatchNorm2d(48)
        # rule
        # maxPool

        self.conv4 = nn.Conv2d(48, 64, (3, 3))
        self.pd4 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm4 = nn.BatchNorm2d(64)
        # rule
        # maxPool

        self.conv5 = nn.Conv2d(64, 80, (3, 3))
        self.pd5 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm5 = nn.BatchNorm2d(80)
        # rule

        self.attn = area_attention.area_attention.AreaAttention(key_query_size=4, max_area_height=3, max_area_width=3,
                                                                memory_width=4, memory_height=4)

        self.classifier = nn.Sequential(
            nn.Linear(5120, 512),
            nn.Dropout(p=0.5),
            nn.PReLU(512),
            nn.Linear(512, 4))

        self.mul = torch.mul
        self.softmax = torch.softmax
        self.cat = torch.cat

    def forward(self, x):
        b = x.shape[0]
        x = x.view(-1, 1, 128, 32)
        xx = self.conv1_1(x)
        xx = self.pd1_1(xx)
        xx = self.nm1_1(xx)

        yy = self.conv1_2(x)
        yy = self.pd1_2(yy)
        yy = self.nm1_2(yy)

        x = torch.cat([xx, yy], dim=1)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pd2(x)
        x = self.nm2(x)
        x = self.maxPool22(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.pd3(x)
        x = self.nm3(x)
        x = self.maxPool22(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.pd4(x)
        x = self.nm4(x)
        x = self.maxPool22(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.pd5(x)
        x = self.nm5(x)
        x = F.relu(x)

        x = x.view(-1, 16, 4)
        Q = x
        K = x
        V = x
        # print(x.shape)
        attn = self.attn(Q, K, V)
        # print(attn.shape)
        x = attn.view(b, -1, 16, 4)
        x = F.relu(x)
        x = x.view(b, -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = ACCN_Area_Multi()
    x = torch.randn(16, 1, 128, 32)
    print(model(x))
