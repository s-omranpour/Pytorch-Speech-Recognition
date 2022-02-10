import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .modules import StreamBlock
from .metrics import wer, cer

class ASRModel(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

        self.streams = nn.ModuleList([
            StreamBlock(config['in_channel'], config['d_model'], 1, config['dropout']),
            StreamBlock(config['in_channel'], config['d_model'], 2, config['dropout']),
            StreamBlock(config['in_channel'], config['d_model'], 3, config['dropout']),
        ])
            
        self.dense = nn.Linear(config['d_model']*len(self.streams), config['n_vocab'])
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm1d(num_classes)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return opt

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x):
        encs = []
        orth_losses = []
        for stream in self.streams:
            enc, l = stream(x)
            encs += [enc]
            orth_losses += [l]
        enc = torch.cat(encs, dim=2)
        h = self.dense(enc)
        # h = h.permute(0,2,1)
        # h = self.do(self.bn(self.relu(h)))
        return F.log_softmax(h, dim=1), sum(orth_losses)

    def step(self, batch, mode='train'):
        # log_probs shape (batch_size, output_len, num_classes)
        log_probs, orth_loss = self.forward(batch['x'])
        assert log_probs.shape[1] >= batch['y'].shape[1]

        # CTC_Loss expects input shape
        # (input_length, batch_size, num_classes)
        log_probs = log_probs.transpose(0, 1)

        # CTC arguments
        # https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788/3
        ctc_loss = self.criterion(log_probs, batch['y'], batch['x_len'] // 3, batch['y_len'])

        alpha = 0.01
        loss = ctc_loss# + alpha*orth_loss
        self.log(f'{mode}_ctc_loss', ctc_loss.item())
        self.log(f'{mode}_orth_loss', orth_loss.item())
        self.log(f'{mode}_loss', loss.item())

        str_y = [self.tokenizer.decode(y) for y in batch['y']]
        str_pred = [self.tokenizer.decode(y) for y in log_probs.transpose(0, 1).argmax(dim=-1)]
        wers = [wer(gt, pred) for gt, pred in zip(str_y, str_pred)]
        cers = [cer(gt, pred) for gt, pred in zip(str_y, str_pred)]
        self.log(f'{mode}_wer', sum(wers)/len(wers))
        self.log(f'{mode}_cer', sum(wers)/len(cers))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        self.step(batch)

    def transcribe(self, x):
        log_probs, _ = self.forward(x.unsqueeze(0))
        preds = log_probs.transpose(0, 1)[0].argmax(dim=-1)
        return self.tokenizer.decode(preds)
