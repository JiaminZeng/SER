import numpy as np
import torch
import torch.nn as nn

from Models.GCN import Graph_CNN_ortega
from dataset import IEMOCAPDataset

label_folder_path = './Data/EmoEvaluation'
file_root = './Data/Sentence'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 10
batch_size = 32
learning_rate = 0.01

# Adj
A = np.zeros((128, 128))
for i in range(127):
    A[i][i + 1] = 1
    A[i + 1][i] = 1
A[0][127] = 1
A[127][0] = 1
A = torch.Tensor(A).to(device)

# Dataset
train_dataset = IEMOCAPDataset(label_folder_path, file_root, train=True)
test_dataset = IEMOCAPDataset(label_folder_path, file_root, train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

classes = ('ang', 'hap and ext', 'neu', 'sad')

model = Graph_CNN_ortega(num_layers=2, input_dim=32, hidden_dim=128, output_dim=4, final_dropout=0.5,
                         graph_pooling_type="sum", device=device, adj=A).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        # origin shape: [batch_size, 128, 127]
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
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
    print(f'Accuracy of the network: {acc} %')

    for i in range(4):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
