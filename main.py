from warnings import simplefilter

import torch

simplefilter(action='ignore', category=FutureWarning)

from Models.ACNN import *
# from Models.SelfAttnCompare import ACCN_Time, ACCN_Torch_GAP
# from Models.AreaAttention import ACCN_Area_Multi
from dataset import IEMOCAPDataset

num_epochs = 100
batch_size = 16
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ACCN_BASE().to(device)
feature_type = "MFCC"

label_folder_path = './Data/IEMOCAP/Evaluation'
file_root = './Data/IEMOCAP/Wav'
RAVDESS_path = './Data/RAVDESS'

# Dataset
train_dataset = IEMOCAPDataset(label_folder_path, file_root, feature_type=feature_type, usage="train")
# train_dataset = RAVDESSDataset(RAVDESS_path, feature_type=feature_type,usage="train")

test_dataset = IEMOCAPDataset(label_folder_path, file_root, feature_type=feature_type, usage="test")
# test_dataset = RAVDESSDataset(RAVDESS_path, feature_type=feature_type,usage="test")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

classes = ('ang', 'exc', 'neu', 'sad')

print(train_dataset.n_samples)
print(test_dataset.n_samples)

criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5, last_epoch=-1)


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

    if (i + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')


def test(max_ua, max_wa):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(4)]
        n_class_samples = [0 for i in range(4)]
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)

            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        ua = 100.0 * n_correct / n_samples
        if ua > max_ua:
            max_ua = ua
        print(f'UA: {ua} %')

        wa = 0
        for i in range(4):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Type:{i} : {acc}')
            wa = wa + acc

        if max_wa < wa:
            max_wa = wa
        print(f'WA: {max_wa / 4} %')

        print(f'BEST - UA: {max_ua} %, WA:{max_wa/4}')
        print(f'------------------------')


max_ua = 0
max_wa = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        train(features, labels)
    test(max_ua, max_wa)

print(model.name)
print(f'batch:{batch_size},lr:{learning_rate}')
