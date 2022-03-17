import librosa
import numpy as np
import soundfile as sf


# 拼接函数
def pad(x, max_len=65200):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


# 拼接和提取mfcc函数

def Seg_MFCC(x):
    x, sp = sf.read(x)
    rate = sp
    features = []
    length = len(x)
    inx = 0
    while inx + rate * 2 <= length:
        ed = inx + rate * 2
        t = x[inx:ed]
        mfcc = librosa.feature.mfcc(t, sr=sp, n_mfcc=32)
        mfcc = np.transpose(mfcc)
        features.append(mfcc)
        inx += int(rate * 0.4)
    if length - rate * 2 >= 0:
        t = x[max(0, length - rate * 2):]
        mfcc = librosa.feature.mfcc(t, sr=sp, n_mfcc=32)
        mfcc = np.transpose(mfcc)
        features.append(mfcc)
    return features


def MFCC(x):
    x, sp = sf.read(x)
    if len(x.shape) == 2:
        x = x[:, 0] + x[:, 1]
        x = x / 2
    x = pad(x)

    x = librosa.util.normalize(x)
    mfcc = librosa.feature.mfcc(x, sr=sp, n_mfcc=32)
    # delta = librosa.feature.delta(mfcc)
    # feats = np.concatenate((mfcc, delta), axis=0)
    feats = np.transpose(mfcc)
    return feats


if __name__ == "__main__":
    f = r"../Data/IEMOCAP/Wav/Ses01F_impro01/Ses01F_impro01_F004.wav"
    ret = Seg_MFCC(f)
    for i in ret:
        print(i.shape)
    # print(Seg_MFCC(f)[1].shape)
