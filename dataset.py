import os
import random

import librosa
import nlpaug.augmenter.audio as naa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data.dataset import Dataset

from Utils.FeaturesUtils import Seg_MFCC, MFCC
from Utils.GetFunction import LFCC


def generator(path):
    aug = naa.VtlpAug(16000, zone=(0.0, 1.0), coverage=1, fhi=4800, factor=(0.8, 1.2))
    for i in range(7):  # 0-7
        wav, _ = librosa.load(path, 16000)
        wavAug = aug.augment(wav)
        sf.write(path + '_' + str(i) + '.wav', wavAug, 16000)


def walk_iemocap(label_folder_path, file_root, use=False):
    emotions_used = {'ang': 0, 'exc': 1, 'neu': 2, 'sad': 3, }
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
                                if use:
                                    for i in range(7):
                                        paths.append(audio_file_path + '_' + str(i) + '.wav')
                                        labels.append(label)
    labels_np = np.array(labels).reshape(-1)
    return paths, labels_np


def walk_ravdess(path):
    # 'ang': 0, 'hap': 1, 'neu': 2, 'sad 3
    emotions_used = {'05': 0, '03': 1, '01': 2, '04': 3, }
    paths = []
    labels = []
    for root, dirs, files in os.walk(path):
        for item in files:
            label = item.split('-')
            if len(label) == 7:
                label = label[2]
            else:
                continue
            if label in emotions_used.keys():
                full_path = os.path.join(root, item)
                paths.append(full_path)
                labels.append(emotions_used[label])

    labels_np = np.array(labels).reshape(-1)
    return paths, labels_np


class IEMOCAPDataset(Dataset):

    def __init__(self, label_folder_path, file_root, feature_type="MFCC", usage="all", aug=False, seg=False):
        self.paths, self.labels = walk_iemocap(label_folder_path, file_root, use=aug)
        self.n_samples = self.labels.shape[0]
        self.seg = seg
        if aug:
            self.n_samples //= 8
        self.feature = feature_type
        random.seed(0)
        self.series = [inx for inx in range(self.n_samples)]
        random.shuffle(self.series)
        num = int(self.n_samples / 5)
        if usage == "all":
            self.series = self.series
        elif usage == "train":
            self.series = self.series[num:]
        else:
            self.series = self.series[:num]
        if aug:
            temp = []
            for item in self.series:
                for inx in range(8):
                    temp.append(item * 8 + inx)
            self.series = temp
            random.shuffle(self.series)

        self.n_samples = len(self.series)
        if seg:
            self.features = []
            self.temp_labels = []
            for id in self.series:
                ret = Seg_MFCC(self.paths[id])
                for item in ret:
                    self.features.append(torch.Tensor(item))
                    self.temp_labels.append(self.labels[id])
            self.labels = np.array(self.temp_labels).reshape(-1)
            self.n_samples = len(self.labels)
            self.series = [inx for inx in range(self.n_samples)]
            random.shuffle(self.series)

        self.labels = torch.from_numpy(self.labels).type(torch.long)

    def __getitem__(self, index):
        id = self.series[index]
        if self.seg:
            return self.features[id], self.labels[id]
        feature = None
        if self.feature == "MFCC":
            feature = torch.Tensor(MFCC(self.paths[id]))
        elif self.feature == "LFCC":
            feature = torch.Tensor(LFCC(self.paths[id]))
        return feature, self.labels[id]

    def __len__(self):
        return self.n_samples


class RAVDESSDataset(Dataset):
    def __init__(self, path, feature_type="MFCC", usage="all"):
        self.paths, self.labels = walk_ravdess(path)
        self.n_samples = self.labels.shape[0]
        self.feature = feature_type
        random.seed(0)
        self.series = [inx for inx in range(self.n_samples)]
        random.shuffle(self.series)
        num = int(self.n_samples / 5)
        if usage == "all":
            self.series = self.series
        elif usage == "train":
            self.series = self.series[num:]
        else:
            self.series = self.series[:num]
        self.n_samples = len(self.series)
        self.labels = torch.from_numpy(self.labels).type(torch.long)

    def __getitem__(self, index):
        feature = ''
        if self.feature == "MFCC":
            feature = torch.Tensor(MFCC(self.paths[self.series[index]]))
        elif self.feature == "LFCC":
            feature = torch.Tensor(LFCC(self.paths[self.series[index]]))
        return feature, self.labels[self.series[index]]

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    from warnings import simplefilter

    simplefilter(action='ignore', category=FutureWarning)
    label_folder_path = './Data/IEMOCAP/Evaluation'
    file_root = './Data/IEMOCAP/Wav'
    train_dataset = IEMOCAPDataset(label_folder_path, file_root, feature_type="MFCC", usage="train", seg=True)
    print(train_dataset.n_samples)
