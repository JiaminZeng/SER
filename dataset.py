import os
import random

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from Utils.FeaturesUtils import MFCC


def walk(label_folder_path, file_root):
    emotions_used = {'ang': 0, 'hap': 1, 'exc': 1, 'neu': 2, 'sad': 3, }
    paths = []
    labels = []
    for root, dirs, files in os.walk(label_folder_path):
        for item in files:
            full_path = os.path.join(root, item)
            if 'Ses' in full_path:
                audio_folder_path = item.split('.')[0]
                with open(full_path, 'r') as f:
                    for line in f:
                        if 'Ses' in line:
                            blocks = line.split('\t')
                            label = blocks[2]
                            if label in emotions_used.keys():
                                label = emotions_used[label]
                                audio_file_path = os.path.join(file_root, audio_folder_path, blocks[1] + '.wav')
                                paths.append(audio_file_path)
                                labels.append(label)
    labels_np = np.array(labels).reshape(-1)
    return paths, labels_np


class IEMOCAPDataset(Dataset):

    def __init__(self, label_folder_path, file_root, train=True):
        self.paths, self.labels = walk(label_folder_path, file_root)
        self.n_samples = self.labels.shape[0]
        random.seed(0)
        self.series = [inx for inx in range(self.n_samples)]
        random.shuffle(self.series)
        if train:
            self.series = self.series[0:5000]
        else:
            self.series = self.series[5000:]
        self.n_samples = len(self.series)
        self.labels = torch.from_numpy(self.labels)

    def __getitem__(self, index):
        feature = torch.Tensor(MFCC(self.paths[self.series[index]]))
        feature.transpose_(0, 1)
        return feature, self.labels[self.series[index]]

    def __len__(self):
        return self.n_samples
