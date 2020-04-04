import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from Module.dataloading import SpeechDataset, get_loaders
from Module.model import MultiStreamSelfAttentionModel
from Module.decoder import GreedyDecoder
from Module.utils import *


## Loading dataset
path= ""
alphabet = pickle.load(open('alphabet.pkl','rb'))
pkwargs={'sr':48000,'num_features':64}

dataset = SpeechDataset(table_path=path+'dataset.csv',alphabet=alphabet, dataset_path=path,
                        preprocess='mfcc', pkwargs=pkwargs, normalize=False)
print('total samples for dataset:',len(dataset))
train_loader, val_loader, test_loader, splits = get_loaders(dataset, batch_size=64, splits=[.7, 0.1, 0.2], workers=4)
print('splits:', splits)
print('num classes:',dataset.intencode.grapheme_count)


## Defining the deep model
num_channels = 64
num_classes = main_dataset.intencode.grapheme_count
ctc_loss = nn.CTCLoss(reduction='sum', zero_infinity=True)
device = torch.device('cuda:0')

model = MultiStreamSelfAttentionModel(n_streams=3, in_channels=num_channels, num_classes=num_classes,
                                      out_channels=[64,128,256], kernels=[9,7,5], strides=[5,2,1],
                                      dilations=[1,3,5], dropout=0.3, n_attention_heads=32, n_hidden=128)

optimizer = optim.Adam(model.parameters(), lr=6e-4, weight_decay=1e-3)
model.to(device)
print('number of parameters:',get_n_params(model))
print(model)


## training
fit(model, train_loader, val_loader=val_loader, epochs=1000, criterion=ctc_loss,
    device=device, optimizer=optimizer, verbose=10, checkpoint=200, save_path='weights/', 
    save_name='model')

## ploting the losses
plot_hist(model)

## Evaluation
stats = evaluate(model, test_loader, device, ctc_loss, steps=1, decoder=GreedyDecoder,
                   convertor=dataset.intencode.convert_to_chars, metrics=True, display_out=True)
print(stats)