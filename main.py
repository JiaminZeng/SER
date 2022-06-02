from warnings import simplefilter

import torch

simplefilter(action='ignore', category=FutureWarning)

from Models.ACNN import *
from Models.CNN import *
from Models.SelfAttnCompare import *
from Models.AreaAttention import *
from dataset import IEMOCAPDataset

num_epochs = 100
batch_size = 16
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ACCN_Torch().to(device)  # 使用模型
feature_type = "MFCC"  # 特征
label_folder_path = './Data/IEMOCAP/Evaluation'  # 音频路径
file_root = './Data/IEMOCAP/Wav'  # 描述文件夹路径
# RAVDESS_path = './Data/RAVDESS'

# Dataset
train_dataset = IEMOCAPDataset(label_folder_path, file_root, feature_type=feature_type, usage="train", seg=True,
                               audio_type='impro')
# train_dataset = RAVDESSDataset(RAVDESS_path, feature_type=feature_type,usage="train")

test_dataset = IEMOCAPDataset(label_folder_path, file_root, feature_type=feature_type, usage="test", seg=True,
                              audio_type='impro')
# test_dataset = RAVDESSDataset(RAVDESS_path, feature_type=feature_type,usage="test")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

classes = ('ang', 'exc', 'neu', 'sad')

# print(train_dataset.n_samples)
# tot 为总音频数，而n_samples是段的数量
tot = test_dataset.tot
print(test_dataset.n_samples)

criterion = nn.CrossEntropyLoss()

# 优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.9, last_epoch=-1)


def train(features, labels):
    features = features.to(device)
    labels = labels.to(device)
    # Forward pass
    outputs = model(features)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # if (i + 1) % 100 == 0:
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')


# 有三种UA与WA计算方法
# 类型1：每段分开后分数相加判断类型
# 类型2：每段分开后先用分数取最大值再判断最终类型
# 类型3：对每段独立判断
# 在类型3下WA和UA能够取得最高

def test():
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(4)]
        n_class_samples = [0 for i in range(4)]

        # 类型1和2
        # lb = [5 for i in range(3000)]
        # sum = torch.zeros([3000, 4]).to(device)

        for features, labels, ids in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)

            # 类型1
            # for i in range(ids.shape[0]):
            #     sum[ids[i]] += outputs[i]
            #     lb[ids[i]] = labels[i].item()

            # 类型2
            # for i in range(ids.shape[0]):
            #     _, predicted = torch.max(outputs[i], 0)
            #     sum[ids[i]][predicted] += 1
            #     lb[ids[i]] = labels[i].item()

            #类型3
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            #类型3
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        #类型1和2
        # for i in range(3000):
        #     if lb[i] != 5:
        #         _, pd = torch.max(sum[i], 0)
        #         n_samples += 1
        #         n_class_samples[lb[i]] += 1
        #         if pd == lb[i]:
        #             n_correct +=1
        #             n_class_correct[lb[i]] +=1

        wa = 100.0 * n_correct / n_samples
        ua = 0
        for i in range(4):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Type:{classes[i]} : {acc}   {n_class_correct[i]}/{n_class_samples[i]}')
            ua = ua + acc

        return ua / 4, wa


# TYPE为1时同时训练与测试，2时仅使用保持的模型测试
TYPE = 1

if __name__ == "__main__":
    max_ua = 0
    max_wa = 0
    if TYPE == 1:
        n_total_steps = len(train_loader)
        for epoch in range(num_epochs):
            for i, (features, labels, ids) in enumerate(train_loader):
                train(features, labels)

            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            ua, wa = test()
            if ua > max_ua:
                max_ua = ua
            if wa > max_wa:
                max_wa = wa
            print(f'CUR  - UA: {ua} %, WA:{wa}')
            print(f'BEST - UA: {max_ua} %, WA:{max_wa}')
            print(f'------------------------')
        torch.save(model.state_dict(), './model.pth')
    else:
        model.load_state_dict(torch.load('./model.pth'))
        ua, wa = test()
        print(f'CUR  - UA: {ua} %, WA:{wa}')
