import os
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.io.wavfile import read
import torchaudio
from torchaudio import transforms, functional


class IntegerEncode:

    def __init__(self, alphabet, max_label_seq=6):
        # reserve 0 for blank label
        self.char2index = {
            "-": 0,
            "pad":1
        }
        self.index2char = {
            0: "-",
            1: "pad"
        }
        self.grapheme_count = 2
        self.process(alphabet)
        self.max_label_seq = max_label_seq

    def process(self, alphabet):
        for a in alphabet:
            self.char2index[a] = self.grapheme_count
            self.index2char[self.grapheme_count] = a
            self.grapheme_count += 1

    def convert_to_ints(self, label):
        y = []
        for char in label:
            y.append(self.char2index[char])
        if len(y) < self.max_label_seq:
            diff = self.max_label_seq - len(y)
            pads = [self.char2index["pad"]] * diff
            y += pads
        return y
    
    def convert_to_chars(self, indices):
        res = ''
        for index in indices:
            if index not in [0, 1]:
                if isinstance(index, torch.Tensor):
                    index = int(index.numpy())
                res += self.index2char[index]
        return res


def normalize(values):
    return (values - np.mean(values)) / np.std(values)



class GoogleSpeechCommand(Dataset):
    """Data set can be found here 
        https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
    """

    def __init__(self, labels, alphabet, data_path, max_frame_len=225, sr=16000, preprocess='raw'):
        
        super(GoogleSpeechCommand, self).__init__()
        self.intencode = IntegerEncode(alphabet)
        self.sr = sr
        self.max_frame_len = max_frame_len
        self.preprocess = preprocess
        self.mfcc = transforms.MFCC(sample_rate=self.sr, n_mfcc=40)
        
        
        self.meta_data = []
        for label in labels:
            path = os.listdir(os.path.join(data_path, label))
            for audio in path:
                audio_path = os.path.join(data_path, label, audio)
                self.meta_data.append((audio_path, label))
        
        
    def __len__(self):
        return len(self.meta_data)
    
    def shuffle(self):
        random.shuffle(self.meta_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio_path, label = self.meta_data[idx]

        audio, _ = torchaudio.load(audio_path)
        x = audio
        if self.preprocess == 'raw':
            pass
#         elif self.preprocess == 'stft':
#             x = torch.stft(x, n_fft=512, hop_length=int(self.sr*0.01), win_length=100, normalized=True)
#             print(x.shape)
#             x = x.squeeze(2)
        elif self.preprocess == 'mfcc':
            x = self.mfcc.forward(x[0,:])
        diff = self.max_frame_len - x.shape[1]
        x = np.pad(x, ((0, 0), (0, diff)), "constant")

        # Each mfcc feature is a channel
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
        # transpose (sample_size, in_frame_len, mfcc_features)
        # to      (sample_size, mfcc_features, in_frame_len)
        #x = np.transpose(x, (1,0)).astype(dtype='float32')
        y = np.array(self.intencode.convert_to_ints(label))
        return (x,y)


def get_loaders(dataset, batch_size, splits, save_indices=False):
    """ splits [train fraction, val fraction, test fraction] """
    
    assert len(splits) == 3
    assert sum(splits) == 1.
    n = len(dataset)
    splits = [int(n*split) for split in splits]
    splits[-1] = n - sum(splits[:2])
    train_dataset, val_dataset, test_dataset = random_split(dataset, splits)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    if save_indices:
        pickle.dump(train_dataset.indices, open('train_indices.pkl', 'wb'))
        pickle.dump(test_dataset.indices, open('test_indices.pkl', 'wb'))
        pickle.dump(val_dataset.indices, open('val_indices.pkl', 'wb'))
    
    return train_loader, val_loader, test_loader, splits
    