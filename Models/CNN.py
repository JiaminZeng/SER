import torch.functional
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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
        self.attention_heads = 8

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(80, 80, (1, 1)))
            self.attention_key.append(nn.Conv2d(80, 80, (1, 1)))
            self.attention_value.append(nn.Conv2d(80, 80, (1, 1)))

        self.classifier = nn.Sequential(
            nn.Linear(40960, 512),
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
        x = F.relu(x)
        x = x.view(b, -1)
        x = self.classifier(x)
        return x


class ConvNet_0(nn.Module):
    def __init__(self):
        super(ConvNet_0, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (8, 8))
        self.pd1 = nn.ZeroPad2d([0, 7, 7, 0])
        self.nm1 = nn.BatchNorm2d(32)

        self.maxPool44 = nn.MaxPool2d((4, 4), ceil_mode=True)

        self.conv2 = nn.Conv2d(32, 48, (8, 8))
        self.pd2 = nn.ZeroPad2d([0, 7, 7, 0])
        self.nm2 = nn.BatchNorm2d(48)
        # rule
        # maxPool

        self.attention_query = torch.nn.ModuleList()
        self.attention_key = torch.nn.ModuleList()
        self.attention_value = torch.nn.ModuleList()
        self.attention_heads = 4

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(48, 48, (1, 1)))
            self.attention_key.append(nn.Conv2d(48, 48, (1, 1)))
            self.attention_value.append(nn.Conv2d(48, 48, (1, 1)))

        self.classifier = nn.Sequential(
            nn.Linear(98304, 512),
            nn.Dropout(p=0.5),
            nn.PReLU(512),
            nn.Linear(512, 4))

        self.mul = torch.mul
        self.softmax = torch.softmax
        self.cat = torch.cat
        self.relu = F.relu

    def forward(self, x):
        b = x.shape[0]
        x = x.view(-1, 1, 128, 64)
        x = self.conv1(x)
        x = self.pd1(x)
        x = self.nm1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.pd2(x)
        x = self.nm2(x)
        x = self.maxPool44(x)
        x = self.relu(x)

        print(x.shape)
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
        x = x.view(b, -1)
        x = self.classifier(x)
        return x


class ConvNet_1(nn.Module):
    def __init__(self):
        super(ConvNet_1, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 16, (10, 5))
        self.pd1_1 = nn.ZeroPad2d([0, 4, 9, 0])
        self.nm1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(1, 8, (5, 10))
        self.pd1_2 = nn.ZeroPad2d([0, 9, 4, 0])
        self.nm1_2 = nn.BatchNorm2d(16)

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

        self.attention_query_0 = torch.nn.ModuleList()
        self.attention_key_0 = torch.nn.ModuleList()
        self.attention_value_0 = torch.nn.ModuleList()
        self.attention_heads_0 = 4

        for i in range(self.attention_heads_0):
            self.attention_query_0.append(nn.Conv2d(80, 80, (1, 1)))
            self.attention_key_0.append(nn.Conv2d(80, 80, (1, 1)))
            self.attention_value_0.append(nn.Conv2d(80, 80, (1, 1)))

        self.attention_query_1 = torch.nn.ModuleList()
        self.attention_key_1 = torch.nn.ModuleList()
        self.attention_value_1 = torch.nn.ModuleList()
        self.attention_heads_1 = 4

        for i in range(self.attention_heads_1):
            self.attention_query_1.append(nn.Conv2d(80, 80, (1, 1)))
            self.attention_key_1.append(nn.Conv2d(80, 80, (1, 1)))
            self.attention_value_1.append(nn.Conv2d(80, 80, (1, 1)))

        self.classifier = nn.Sequential(
            nn.Linear(80 * 16 * 8 * self.attention_heads_1, 512),
            nn.Dropout(p=0.5),
            nn.PReLU(512),
            nn.Linear(512, 4))

        self.mul = torch.mul
        self.softmax = torch.softmax
        self.cat = torch.cat

    def forward(self, x):
        b = x.shape[0]
        x = x.view(-1, 1, 128, 64)
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

        attn = None
        for i in range(self.attention_heads_0):
            Q = self.attention_query_0[i](x)
            K = self.attention_key_0[i](x)
            V = self.attention_value_0[i](x)
            attention = self.softmax(self.mul(Q, K), -1)
            attention = self.mul(attention, V)
            if attn is None:
                attn = attention
            else:
                attn = self.cat([attn, attention], 2)

        attn = None
        for i in range(self.attention_heads_1):
            Q = self.attention_query_1[i](x)
            K = self.attention_key_1[i](x)
            V = self.attention_value_1[i](x)
            attention = self.softmax(self.mul(Q, K), -1)
            attention = self.mul(attention, V)
            if attn is None:
                attn = attention
            else:
                attn = self.cat([attn, attention], 2)

        x = attn
        x = F.relu(x)
        x = x.view(b, -1)
        x = self.classifier(x)
        return x


class ConvNet_2(nn.Module):
    def __init__(self):
        super(ConvNet_2, self).__init__()
        self.attention_query = torch.nn.ModuleList()
        self.attention_key = torch.nn.ModuleList()
        self.attention_value = torch.nn.ModuleList()
        self.attention_heads = 4

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(1, 4, (1, 1)))
            self.attention_key.append(nn.Conv2d(1, 4, (1, 1)))
            self.attention_value.append(nn.Conv2d(1, 4, (1, 1)))

        self.classifier = nn.Sequential(
            nn.Linear(131072, 512),
            nn.Dropout(p=0.5),
            nn.PReLU(512),
            nn.Linear(512, 4))

        self.mul = torch.mul
        self.softmax = torch.softmax
        self.cat = torch.cat

    def forward(self, x):
        b = x.shape[0]
        x = x.view(-1, 1, 128, 64)
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
        x = x.view(b, -1)
        x = self.classifier(x)
        return x


class ConvNet_3(nn.Module):
    def __init__(self):
        super(ConvNet_3, self).__init__()
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
        self.attention_heads = 3

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(16, 16, (1, 1)))
            self.attention_key.append(nn.Conv2d(16, 16, (1, 1)))
            self.attention_value.append(nn.Conv2d(16, 16, (1, 1)))

        self.classifier = nn.Sequential(
            nn.Linear(163840, 512),
            nn.Dropout(p=0.5),
            nn.PReLU(512),
            nn.Linear(512, 4))

        self.mul = torch.mul
        self.softmax = torch.softmax
        self.cat = torch.cat

    def forward(self, x):
        b = x.shape[0]
        x = x.view(-1, 1, 128, 64)
        xx = self.conv1_1(x)
        xx = self.pd1_1(xx)
        xx = self.nm1_1(xx)

        yy = self.conv1_2(x)
        yy = self.pd1_2(yy)
        yy = self.nm1_2(yy)

        x = torch.cat([xx, yy], dim=1)
        x = F.relu(x)

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

        # x = self.conv2(x)
        # x = self.pd2(x)
        # x = self.nm2(x)
        # x = self.maxPool22(x)
        # x = F.relu(x)
        #
        # x = self.conv3(x)
        # x = self.pd3(x)
        # x = self.nm3(x)
        # x = self.maxPool22(x)
        # x = F.relu(x)

        x = self.conv4(x)
        x = self.pd4(x)
        x = self.nm4(x)
        x = self.maxPool22(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.pd5(x)
        x = self.nm5(x)
        x = F.relu(x)

        x = x.view(b, -1)
        x = self.classifier(x)
        return x

# model = ConvNet()
# tr = torch.randn([4, 1, 128, 64])
# print(model(tr).shape)
