import os
import re
import numpy as np
import pandas as pd
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
from torchaudio import transforms, functional

transform = None
do_normalize = False


class IntegerEncode:

    def __init__(self, alphabet):
        # reserve 0 for blank label
        self.char2index = { "-": 0 }
        self.index2char = { 0: "-" }
        self.grapheme_count = 1
        self.process(alphabet)

    def process(self, alphabet):
        for a in alphabet:
            self.char2index[a] = self.grapheme_count
            self.index2char[self.grapheme_count] = a
            self.grapheme_count += 1

    def convert_to_ints(self, label):
        y = []
        for char in label:
            y.append(self.char2index[char])
        return y
    
    def convert_to_chars(self, indices):
        res = ''
        for index in indices:
            if isinstance(index, torch.Tensor):
                index = int(index.numpy())
            if index != 0:
                res += self.index2char[index]
        return res


def normalize(a):
    return (a - a.mean()) / a.std()


class SpeechDataset(Dataset):

    def __init__(self, table_path, alphabet, dataset_path='', max_len=0, preprocess='raw',
                 normalize=False, pkwargs=None):
        global transform, do_normalize
        
        super(SpeechDataset, self).__init__()
        self.table = pd.read_csv(table_path)
        self.dataset_path = dataset_path
        self.intencode = IntegerEncode(alphabet)
        self.max_len = max_len
        if preprocess == "mfcc":
            transform = transforms.MFCC(sample_rate=pkwargs['sr'], n_mfcc=pkwargs['num_features'])
        elif preprocess == "spectrogram":
            transform = transforms.Spectrogram(n_fft=pkwargs['n_fft'], normalized=pkwargs['normalized'])
        
        
    def __len__(self):
        return self.table.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio_path, label = self.table.loc[idx]
        if self.dataset_path:
            audio_path = self.dataset_path + audio_path
        audio, _ = torchaudio.load(audio_path)
        x = audio
        if self.max_len:
            x = np.pad(x, ((0, 0), (0, self.max_len - x.shape[1])), "constant")
        regex = re.compile('[^a-zA-Z ]')
        label = regex.sub('', label).lower()
        y = np.array(self.intencode.convert_to_ints(label))
        return (x, y)
    
def pad(batch):
    global transform, do_normalize
    max_len_x = 0
    max_len_y = 0
    data = []
    target = []
    for sample in batch:
        data += [sample[0]]
        target += [sample[1]]
        n = sample[0].shape[1]
        m = sample[1].shape[0]
        if max_len_x < n:
            max_len_x = n
        if max_len_y < m:
            max_len_y = m
    data = torch.tensor([F.pad(input=x, pad=(0, max_len_x - x.shape[1])).numpy() for x in data])
    if transform:
        data = transform(data[:,0,:])
    if do_normalize:
        data = normalize(data)

    target_lengths = [y.shape[0] for y in target]
    target = [np.pad(y, (0, max_len_y - y.shape[0]), 'constant') for y in target]
    return [data, torch.tensor(target), torch.tensor(target_lengths)]


def get_loaders(dataset, batch_size, splits=None, workers=1):
    """ splits [train fraction, val fraction, test fraction] """

    n = len(dataset)
    assert len(splits) == 3
    assert sum(splits) == 1.
    splits = [int(n*split) for split in splits]
    splits[-1] = n - sum(splits[:2])

    fn = pad
    train_dataset, val_dataset, test_dataset = random_split(dataset, splits)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                              collate_fn=fn, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                            collate_fn=fn, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                             collate_fn=fn, num_workers=workers)
    
    return train_loader, val_loader, test_loader, splits