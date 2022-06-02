import area_attention
import torch.functional
import torch.nn as nn
import torch.nn.functional as F


class ACNN_PART1_2(nn.Module):

    def __init__(self, activation='RReLu', pool='LP', attn='Base'):
        super(ACNN_PART1_2, self).__init__()

        self.pool22 = nn.MaxPool2d((3, 3), ceil_mode=True)
        self.activation = F.relu
        self.attn_type = attn

        if activation == 'RReLu':
            self.activation = F.rrelu
        elif activation == 'Tanh':
            self.activation = F.tanh

        if pool == 'Avg':
            self.pool22 = nn.AvgPool2d((2, 2), ceil_mode=True)
        elif pool == 'LP':
            self.pool22 = nn.LPPool2d(2, 2, ceil_mode=True)

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

        self.attn = torch.nn.MultiheadAttention(4, 4)
        if attn == 'Area':
            self.single = area_attention.area_attention.AreaAttention(key_query_size=4, max_area_height=3,
                                                                      max_area_width=3,
                                                                      memory_width=4, memory_height=4)
            self.attn = area_attention.multi_head_area_attention.MultiHeadAreaAttention(self.single, 4, 4, 4, 4, 4)

        self.classifier = nn.Sequential(
            nn.Linear(65536, 512),
            nn.Dropout(p=0.5),
            nn.PReLU(512),
            nn.Linear(512, 4))

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

        x = x.view(-1, 16, 4)
        if self.attn_type == 'Base':
            x = x.view(-1, 16, 4)
            x = x.transpose(0, 1)
            Q = x
            K = x
            V = x
            attn, _ = self.attn(Q, K, V)
            attn = attn.transpose(0, 1)
        else:
            Q = x
            K = x
            V = x
            attn = self.attn(Q, K, V)

        x = attn.view(b, -1, 16, 4)
        x = self.activation(x)
        x = x.reshape(b, -1)
        print(x.shape)
        x = self.classifier(x)
        return x


class ACCN_BASE(nn.Module):

    def __init__(self):
        super(ACCN_BASE, self).__init__()
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
        self.maxPool22 = nn.AvgPool2d((2, 2), ceil_mode=True)

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
            self.attention_query.append(nn.Conv2d(80, 32, (1, 1)))
            self.attention_key.append(nn.Conv2d(80, 32, (1, 1)))
            self.attention_value.append(nn.Conv2d(80, 32, (1, 1)))

        self.classifier = nn.Linear(80, 4)

        self.mul = torch.mul
        self.softmax = torch.softmax
        self.cat = torch.cat
        self.avg = nn.AdaptiveAvgPool2d([1, 1])

    def forward(self, x):
        b = x.shape[0]
        x = x.view(-1, 1, 128, 32)
        xx = self.conv1_1(x)
        xx = self.pd1_1(xx)
        xx = self.nm1_1(xx)
        xx = F.relu(xx)

        yy = self.conv1_2(x)
        yy = self.pd1_2(yy)
        yy = self.nm1_2(yy)
        yy = F.relu(yy)

        x = torch.cat([xx, yy], dim=2)

        x = self.conv2(x)
        x = self.pd2(x)
        x = self.nm2(x)
        x = F.relu(x)
        x = self.maxPool22(x)

        x = self.conv3(x)
        x = self.pd3(x)
        x = self.nm3(x)
        x = F.relu(x)
        x = self.maxPool22(x)

        x = self.conv4(x)
        x = self.pd4(x)
        x = self.nm4(x)
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
                attn = self.cat([attn, attention], 3)

        x = F.relu(x)

        x = self.avg(x)

        x = x.view(b, -1)

        x = self.classifier(x)

        return x


class ACCN_0_0(nn.Module):
    """
        batch = 16,lr=0.001 acc=60.79

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


class ACCN_0_1(nn.Module):
    """
        batch = 16,lr=0.001 acc=60.28，60.58

    """

    def __init__(self):
        super(ACCN_0_1, self).__init__()
        self.name = "0_1"
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


class CCN_1_0(nn.Module):
    """
        batch = 16,lr=0.001 acc=58.45,58.46

    """

    def __init__(self):
        super(CCN_1_0, self).__init__()
        self.name = "1_0"
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
        self.maxPool22 = nn.MaxPool2d((2, 2), stride=1)

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

        self.classifier = nn.Sequential(
            nn.Linear(290000, 512),
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

        x = F.relu(x)
        x = x.view(b, -1)
        x = self.classifier(x)
        return x


class CCN_1_1(nn.Module):
    """
        batch = 16,lr=0.001 acc=59.47,59.777

    """

    def __init__(self):
        super(CCN_1_1, self).__init__()
        self.name = "1_1"
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
        self.maxPool22 = nn.MaxPool2d((2, 2), stride=1)

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

        self.classifier = nn.Sequential(
            nn.Linear(290000, 512),
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

        x = x.view(b, -1)
        x = self.classifier(x)
        return x


class CCN_2_0(nn.Module):
    def __init__(self):
        super(CCN_2_0, self).__init__()
        self.name = "2_0"
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

        self.attention_query = torch.nn.ModuleList()
        self.attention_key = torch.nn.ModuleList()
        self.attention_value = torch.nn.ModuleList()
        self.attention_heads = 4

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(80, 32, (1, 1)))
            self.attention_key.append(nn.Conv2d(80, 32, (1, 1)))
            self.attention_value.append(nn.Conv2d(80, 32, (1, 1)))

        self.classifier = nn.Sequential(
            nn.Linear(2048, 4))

        self.mul = torch.mul
        self.softmax = torch.softmax
        self.cat = torch.cat
        # self.avg = nn.AdaptiveAvgPool2d([1, 1])

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
                attn = attn + attention
                # attn = self.cat([attn, attention], 1)

        x = F.relu(attn / 4)
        # x = self.avg(x)
        print(x.shape)

        x = x.view(b, -1)
        x = self.classifier(x)
        return x


class ACCN_2_1(nn.Module):
    """
        batch:16,lr:0.001:  58.76393110435664
    """

    def __init__(self):
        super(ACCN_2_1, self).__init__()
        self.name = "2_1"
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
            self.attention_query.append(nn.Conv2d(80, 32, (1, 1)))
            self.attention_key.append(nn.Conv2d(80, 32, (1, 1)))
            self.attention_value.append(nn.Conv2d(80, 32, (1, 1)))

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
        xx = F.relu(xx)
        # print(1, xx.shape)

        yy = self.conv1_2(x)
        yy = self.pd1_2(yy)
        yy = self.nm1_2(yy)
        yy = F.relu(yy)
        # print(2, yy.shape)

        x = torch.cat([xx, yy], dim=2)
        # print(3, x.shape)

        x = self.conv2(x)
        x = self.pd2(x)
        x = self.nm2(x)
        x = F.relu(x)
        x = self.maxPool22(x)
        # print(4, x.shape)

        x = self.conv3(x)
        x = self.pd3(x)
        x = self.nm3(x)
        x = F.relu(x)
        x = self.maxPool22(x)
        # print(5, x.shape)

        x = self.conv4(x)
        x = self.pd4(x)
        x = self.nm4(x)
        x = F.relu(x)
        # print(6, x.shape)

        x = self.conv5(x)
        x = self.pd5(x)
        x = self.nm5(x)
        x = F.relu(x)
        # print(7, x.shape)

        # attn = None
        # for i in range(self.attention_heads):
        #     Q = self.attention_query[i](x)
        #     K = self.attention_key[i](x)
        #     V = self.attention_value[i](x)
        #     attention = self.softmax(self.mul(Q, K), -1)
        #     attention = self.mul(attention, V)
        #     if attn is None:
        #         attn = attention
        #     else:
        #         attn = self.cat([attn, attention], 3)

        # print(8, x.shape)

        # x = attn.transpose(1, 2)
        # x = x.transpose(2, 3)
        # print(9, x.shape)

        # x = F.relu(x)
        # print(10, x.shape)

        # x = self.avg(x)
        # print(11, x.shape)

        x = x.view(b, -1)
        # print(12, x.shape)

        x = self.classifier(x)
        # print(13, x.shape)

        return x


class ACCN_2_2(nn.Module):
    def __init__(self):
        """
            batch:16,lr:0.001:  59.6757852077001
        """
        super(ACCN_2_2, self).__init__()
        self.name = "2_2"
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

        self.conv6 = nn.Conv2d(80, 128, (3, 3))
        self.pd6 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm6 = nn.BatchNorm2d(128)
        # rule

        self.attention_query = torch.nn.ModuleList()
        self.attention_key = torch.nn.ModuleList()
        self.attention_value = torch.nn.ModuleList()
        self.attention_heads = 4

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(128, 64, (1, 1)))
            self.attention_key.append(nn.Conv2d(128, 64, (1, 1)))
            self.attention_value.append(nn.Conv2d(128, 64, (1, 1)))

        self.classifier = nn.Sequential(
            nn.Linear(65536, 512),
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
        xx = F.relu(xx)
        # print(1, xx.shape)

        yy = self.conv1_2(x)
        yy = self.pd1_2(yy)
        yy = self.nm1_2(yy)
        yy = F.relu(yy)
        # print(2, yy.shape)

        x = torch.cat([xx, yy], dim=2)
        # print(3, x.shape)

        x = self.conv2(x)
        x = self.pd2(x)
        x = self.nm2(x)
        x = F.relu(x)
        x = self.maxPool22(x)
        # print(4, x.shape)

        x = self.conv3(x)
        x = self.pd3(x)
        x = self.nm3(x)
        x = F.relu(x)
        x = self.maxPool22(x)
        # print(5, x.shape)

        x = self.conv4(x)
        x = self.pd4(x)
        x = self.nm4(x)
        x = F.relu(x)
        # print(6, x.shape)

        x = self.conv5(x)
        x = self.pd5(x)
        x = self.nm5(x)
        x = F.relu(x)
        # print(7, x.shape)

        x = self.conv6(x)
        x = self.pd6(x)
        x = self.nm6(x)
        x = F.relu(x)

        # attn = None
        # for i in range(self.attention_heads):
        #     Q = self.attention_query[i](x)
        #     K = self.attention_key[i](x)
        #     V = self.attention_value[i](x)
        #     attention = self.softmax(self.mul(Q, K), -1)
        #     attention = self.mul(attention, V)
        #     if attn is None:
        #         attn = attention
        #     else:
        #         attn = self.cat([attn, attention], 3)

        # print(8, x.shape)

        # x = attn.transpose(1, 2)
        # x = x.transpose(2, 3)
        # print(9, x.shape)

        # x = F.relu(x)
        # print(10, x.shape)

        # x = self.avg(x)
        # print(11, x.shape)

        x = x.view(b, -1)
        # print(12, x.shape)

        x = self.classifier(x)
        # print(13, x.shape)

        return x


class ACCN_2_3(nn.Module):
    def __init__(self):
        """
        batch:16,lr:0.001:  60.790273556231
        """
        super(ACCN_2_3, self).__init__()
        self.name = "2_3"
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

        self.conv6 = nn.Conv2d(80, 128, (3, 3))
        self.pd6 = nn.ZeroPad2d([0, 2, 2, 0])
        self.nm6 = nn.BatchNorm2d(128)
        # rule

        self.attention_query = torch.nn.ModuleList()
        self.attention_key = torch.nn.ModuleList()
        self.attention_value = torch.nn.ModuleList()
        self.attention_heads = 4

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(128, 64, (1, 1)))
            self.attention_key.append(nn.Conv2d(128, 64, (1, 1)))
            self.attention_value.append(nn.Conv2d(128, 64, (1, 1)))

        self.classifier = nn.Sequential(
            nn.Linear(65536, 512),
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
        xx = F.relu(xx)
        # print(1, xx.shape)

        yy = self.conv1_2(x)
        yy = self.pd1_2(yy)
        yy = self.nm1_2(yy)
        yy = F.relu(yy)
        # print(2, yy.shape)

        x = torch.cat([xx, yy], dim=2)
        # print(3, x.shape)

        x = self.conv2(x)
        x = self.pd2(x)
        x = self.nm2(x)
        x = F.relu(x)
        x = self.maxPool22(x)
        # print(4, x.shape)

        x = self.conv3(x)
        x = self.pd3(x)
        x = self.nm3(x)
        x = F.relu(x)
        x = self.maxPool22(x)
        # print(5, x.shape)

        x = self.conv4(x)
        x = self.pd4(x)
        x = self.nm4(x)
        x = F.relu(x)
        # print(6, x.shape)

        x = self.conv5(x)
        x = self.pd5(x)
        x = self.nm5(x)
        x = F.relu(x)
        # print(7, x.shape)

        x = self.conv6(x)
        x = self.pd6(x)
        x = self.nm6(x)
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
                attn = self.cat([attn, attention], 3)

        # print(8, x.shape)

        # x = attn.transpose(1, 2)
        # x = x.transpose(2, 3)
        # print(9, x.shape)

        x = F.relu(x)
        # print(10, x.shape)

        # x = self.avg(x)
        # print(11, x.shape)

        x = x.view(b, -1)
        # print(12, x.shape)

        x = self.classifier(x)
        # print(13, x.shape)

        return x


if __name__ == '__main__':
    import numpy as np

    model = ACNN_PART1_2(attn='Area')
    np.random.seed(1)
    test = np.random.random((16, 1, 128, 32)).astype(np.float32)
    x = torch.from_numpy(test)
    print(model(x))
