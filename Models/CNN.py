import torch.functional
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, (12, 16))
        # self.nm1 = nn.BatchNorm2d(16)
        # relu
        # self.maxPool

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
