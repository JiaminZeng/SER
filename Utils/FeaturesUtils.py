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
def MFCC(x):
    x, sp = sf.read(x)
    print(x.shape)
    x = pad(x)
    x = librosa.util.normalize(x)
    mfcc = librosa.feature.mfcc(x, sr=sp, n_mfcc=32)
    delta = librosa.feature.delta(mfcc)
    # delta2 = librosa.feature.delta(delta)
    feats = np.concatenate((mfcc, delta), axis=0)
    feats = np.transpose(feats)
    return feats
