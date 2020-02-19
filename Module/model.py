from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Module.decoder import GreedyDecoder
import matplotlib.pyplot as plt


class Wav2Letter(nn.Module):
    """Wav2Letter Speech Recognition model
        Args:
            num_features (int): number of mfcc features
            num_classes (int): number of unique grapheme class labels
    """

    def __init__(self, num_features, num_classes, criterion, device, notebook=True):
        
        super(Wav2Letter, self).__init__()
        if notebook:
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm
        self.device = device
        self.criterion = criterion
        self.history = {'train':[], 'val':[]}
        
        # Conv1d(in_channels, out_channels, kernel_size, stride)
        self.layers = nn.Sequential(
            nn.Conv1d(num_features, 250, 31, 2),
            nn.BatchNorm1d(250, affine=False),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 19),
            nn.BatchNorm1d(250, affine=False),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 17),
            nn.BatchNorm1d(250, affine=False),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 11),
            nn.BatchNorm1d(250, affine=False),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.BatchNorm1d(250, affine=False),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            nn.BatchNorm1d(250, affine=False),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 5),
            nn.BatchNorm1d(250, affine=False),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 5),
            nn.BatchNorm1d(250, affine=False),
            torch.nn.ReLU(),
            nn.Conv1d(250, 500, 5),
            nn.BatchNorm1d(500, affine=False),
            torch.nn.ReLU(),
            nn.Conv1d(500, 500, 1),
            nn.BatchNorm1d(500, affine=False),
            torch.nn.ReLU(),
            nn.Conv1d(500, num_classes, 1),
        )
        self.to(device)

    def forward(self, batch):
        """Forward pass through Wav2Letter network than 
            takes log probability of output

        Args:
            batch (int): mini batch of data
             shape (batch, num_features, frame_len)

        Returns:
            log_probs (torch.Tensor):
                shape  (batch_size, num_classes, output_len)
        """
        # y_pred shape (batch_size, num_classes, output_len)
        y_pred = self.layers(batch)

        # compute log softmax probability on graphemes
        log_probs = F.log_softmax(y_pred, dim=1)

        return log_probs

    def fit(self, train_loader, epochs, optimizer, val_loader=None, verbose=10, checkpoint=50):
        """Trains Wav2Letter model.

        Args:
            inputs (torch.Tensor): shape (sample_size, num_features, frame_len)
            output (torch.Tensor): shape (sample_size, seq_len)
            epochs (int): number of epochs
        """
        
        self.train()
        for t in self.tqdm(range(epochs)):
            total_loss = 0.
            num_samples = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # reading data
                x, y = batch
                batch_size = x.shape[0]
                output_shape = y.shape[1]

                # log_probs shape (batch_size, num_classes, output_len)
                log_probs = self.forward(x.to(self.device))

                # CTC_Loss expects input shape
                # (input_length, batch_size, num_classes)
                log_probs = log_probs.transpose(1, 2).transpose(0, 1)

                # CTC arguments
                # https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788/3
                input_lengths = torch.full((batch_size,), log_probs.shape[0], dtype=torch.long)
                target_lengths = torch.full((batch_size,), output_shape, dtype=torch.long)

                loss = self.criterion(log_probs, y.to(self.device), input_lengths, target_lengths)

                total_loss += loss.item()
                num_samples += batch_size
                
                loss.backward()
                optimizer.step()
            
            total_loss /= num_samples
            self.history['train'] += [total_loss]
            log = " loss : " + str(total_loss)
            if val_loader:
                val_loss = self.eval(val_loader, out=False)
                self.history['val'] += [val_loss]
                log += " val loss : " + str(val_loss)

            if (t+1) % verbose == 0:
                print("epoch", t + 1,log)
            if (t+1) % checkpoint == 0:
                self.save(t+1)
                

    def eval(self, loader, decoder=None, metric=None, out=True):
        self.train(False)
        total_loss = 0.
        num_samples = 0
        outputs = torch.Tensor([])
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                batch_size = x.shape[0]
                output_shape = y.shape[1]
                log_probs = self.forward(x.to(self.device))
                log_probs = log_probs.transpose(1, 2).transpose(0, 1)
                input_lengths = torch.full((batch_size,), log_probs.shape[0], dtype=torch.long)
                target_lengths = torch.full((batch_size,), output_shape, dtype=torch.long)
                loss = self.criterion(log_probs, y.to(self.device), input_lengths, target_lengths)
                total_loss += loss.item()
                num_samples += batch_size
                if out:
                    output = decoder(log_probs).cpu().float()
                    outputs = torch.cat([outputs, output.view(1, output.shape[0])], dim=0)

        if out:
            return outputs, total_loss/num_samples
        return total_loss/num_samples
    
    def plot_hist(self):
        plt.plot(self.history['train'])
        plt.plot(self.history['val'])
        plt.legend(['train', 'val'])
        plt.show()
        
    def get_n_params(self):
        pp=0
        for p in list(self.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp
    
    def save(self, arg):
        torch.save(self.state_dict(), 'weights/wav2letter_'+str(arg)+'.pkl')
        
    def load(self, path):
        self.load_state_dict(torch.load(path))