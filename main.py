RAVDESS_path = './Data/RAVDESS'
label_folder_path = './Data/IEMOCAP/Evaluation'
file_root = './Data/IEMOCAP/Wav'

from dataset import RAVDESSDataset,IEMOCAPDataset
train_dataset = IEMOCAPDataset(label_folder_path, file_root, feature_type="MFCC", usage="train")
