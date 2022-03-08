RAVDESS_path = './Data/RAVDESS'
label_folder_path = './Data/IEMOCAP/Evaluation'
file_root = './Data/IEMOCAP/Wav'

from dataset import IEMOCAPDataset

train_dataset = IEMOCAPDataset(label_folder_path, file_root, feature_type="MFCC", usage="train", aug="True")
test_dataset = IEMOCAPDataset(label_folder_path, file_root, feature_type="MFCC", usage="test", aug="True")
print(train_dataset.n_samples)
print(test_dataset.n_samples)
