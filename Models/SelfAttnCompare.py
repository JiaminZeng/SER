import torch.functional
import torch.nn as nn
import torch.nn.functional as F

"""
    Test MultiHead between Self-Build and Torch-Build
"""


class ACCN_Torch_0_0(nn.Module):
    """
        63.22
        +aug 64.23
    """

    def __init__(self):
        super(ACCN_Torch_0_0, self).__init__()
        self.name = "0_0"
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

        self.attn = torch.nn.MultiheadAttention(4, 4)

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

        # print(x.shape)
        x = x.view(-1, 16, 4)
        x = x.transpose(0, 1)
        Q = x
        K = x
        V = x
        attn, _ = self.attn(Q, K, V)
        x = attn.transpose(0, 1)
        x = x.view(b, -1, 16, 4)
        # print(x.shape)
        x = F.relu(x)
        x = x.reshape(b, -1)
        x = self.classifier(x)
        return x


class ACCN_Time(nn.Module):
    """

    """

    def __init__(self):
        super(ACCN_Time, self).__init__()
        self.name = "0_0"
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

        self.attn = torch.nn.MultiheadAttention(4, 4)

        self.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.Dropout(p=0.5),
            nn.PReLU(512),
            nn.Linear(512, 4))

        self.mul = torch.mul
        self.softmax = torch.softmax
        self.cat = torch.cat

    def forward(self, x):
        b = x.shape[0]
        x = x.view(-1, 1, 63, 32)
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

        x = x.view(-1, 8, 4)
        x = x.transpose(0, 1)
        Q = x
        K = x
        V = x
        attn, _ = self.attn(Q, K, V)
        x = attn.transpose(0, 1)
        x = x.view(b, -1, 8, 4)
        # print(x.shape)
        x = F.relu(x)
        x = x.reshape(b, -1)
        x = self.classifier(x)
        return x


class ACCN_0_0(nn.Module):
    """
        batch = 16,lr=0.001 acc=60.79,61
    """

    def __init__(self):
        super(ACCN_0_0, self).__init__()
        self.name = "0_0"
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

        self.attention_query = torch.nn.ModuleList()
        self.attention_key = torch.nn.ModuleList()
        self.attention_value = torch.nn.ModuleList()
        self.attention_heads = 4

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(80, 80, (1, 1)))
            self.attention_key.append(nn.Conv2d(80, 80, (1, 1)))
            self.attention_value.append(nn.Conv2d(80, 80, (1, 1)))

        self.classifier = nn.Sequential(
            nn.Linear(20480, 512),
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

        # print(x.shape)
        attn = None
        for i in range(self.attention_heads):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = self.softmax(self.mul(Q, K), -1)
            attention = self.mul(attention, V)
            if attn is None:
                attn = attention
            else:
                attn = self.cat([attn, attention], 1)

        x = attn
        # print(x.shape)

        x = F.relu(x)
        x = x.view(b, -1)
        x = self.classifier(x)
        return x


class ACCN_Torch_GAP(nn.Module):
    """
        63.22
        +aug 64.23
    """

    def __init__(self):
        super(ACCN_Torch_GAP, self).__init__()
        self.name = "0_0"
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

        self.attention_query = torch.nn.ModuleList()
        self.attention_key = torch.nn.ModuleList()
        self.attention_value = torch.nn.ModuleList()
        self.attention_heads = 4

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(80, 80, (1, 1)))
            self.attention_key.append(nn.Conv2d(80, 80, (1, 1)))
            self.attention_value.append(nn.Conv2d(80, 80, (1, 1)))

        self.classifier = nn.Sequential(
            # nn.Linear(32, 512),
            # nn.Dropout(p=0.5),
            # nn.PReLU(512),
            nn.Linear(80, 4))

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

        # print(x.shape)
        attn = None
        for i in range(self.attention_heads):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = self.softmax(self.mul(Q, K), -1)
            attention = self.mul(attention, V)
            if attn is None:
                attn = attention
            else:
                attn = self.cat([attn, attention], 2)

        x = attn
        # print(x.shape)
        x = nn.AdaptiveMaxPool2d([1, 1])(x)

        # x = F.relu(x)
        x = x.view(b, -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = ACCN_Torch_GAP()
    x = torch.randn(16, 1, 128, 32)
    model(x)
    # print(model(x))

"""
    torch.Size([16, 80, 16, 4])
    torch.Size([16, 320, 16, 4])
"""
