from warnings import simplefilter

import torch
import torch.nn as nn

from Models.CNN import ConvNet
from dataset import IEMOCAPDataset

simplefilter(action='ignore', category=FutureWarning)

label_folder_path = './Data/IEMOCAP/Evaluation'
file_root = './Data/IEMOCAP/Wav'
RAVDESS_path = './Data/RAVDESS'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 200
batch_size = 128
learning_rate = 0.001

feature_type = "MFCC"
num_layer = 2
frame_size = 128
input_dim = 64
pool_type = "sum"
hidden_dim = 128
final_dropout = 0.5

msg = f'feature_type{feature_type}\nnum_layer{num_layer}\nframe_size{frame_size}\nimput_dim{input_dim}\n' \
      f'pool_type{pool_type}\nhidden_dim{hidden_dim}\nfinal_dropout{final_dropout}'

# Adj
# A = np.zeros((frame_size, frame_size))
# for i in range(frame_size - 1):
#     A[i][i + 1] = 1
#     A[i + 1][i] = 1
# A[0][frame_size - 1] = 1
# A[frame_size - 1][0] = 1
# A = torch.Tensor(A).to(device)

# Dataset
train_dataset = IEMOCAPDataset(label_folder_path, file_root, feature_type=feature_type, usage="train")
# train_dataset = RAVDESSDataset(RAVDESS_path, feature_type=feature_type,usage="train")

test_dataset = IEMOCAPDataset(label_folder_path, file_root, feature_type=feature_type, usage="test")
# test_dataset = RAVDESSDataset(RAVDESS_path, feature_type=feature_type,usage="test")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

classes = ('ang', 'exc', 'neu', 'sad')

# model = Graph_CNN_ortega(num_layers=num_layer, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=4,
#                          final_dropout=final_dropout, graph_pooling_type=pool_type, device=device, adj=A).to(device)
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
max_acc = 0

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 30 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

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

        acc = 100.0 * n_correct / n_samples
        if acc > max_acc:
            max_acc = acc
        print(f'Accuracy of the network: {acc} %')

        for i in range(4):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %')
        print(f'Best result: {max_acc} %')
        print(f'------------------------')

print(msg)
