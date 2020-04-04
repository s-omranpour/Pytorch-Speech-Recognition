import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from Module.metrics import cer, wer

import torch
import torch.nn as nn
import torch.nn.functional as F


def fit(model, train_loader, epochs, optimizer, device, criterion, val_loader=None, train_steps=None,
            val_steps=None, verbose=10, checkpoint=0, save_path=None, save_name=None):
    """Trains model.

    Args:
        inputs (torch.Tensor): shape (sample_size, num_features, frame_len)
        output (torch.Tensor): shape (sample_size, seq_len)
        epochs (int): number of epochs
    """
    assert (checkpoint and save_path and save_name) or not checkpoint
    
    for t in tqdm(range(epochs)):
        total_loss = 0.
        num_samples = 0
        n = 0
        for batch in train_loader:
            if train_steps and n == train_steps:
                break
            model.train()
            optimizer.zero_grad()

            # reading data
            x, y, y_len = batch
            batch_size = x.shape[0]
            n += 1

            # log_probs shape (batch_size, num_classes, output_len)
            log_probs = model(x.to(device))
            assert log_probs.shape[2] >= y.shape[1]

            # CTC_Loss expects input shape
            # (input_length, batch_size, num_classes)
            log_probs = log_probs.permute(2,0,1)

            # CTC arguments
            # https://discuss.pytorch.org/t/ctcloss-with-warp-ctc-help/8788/3
            input_lengths = torch.full((batch_size,), log_probs.shape[0], dtype=torch.long)

            loss = criterion(log_probs, y.to(device), input_lengths, y_len)

            total_loss += loss.item()
            num_samples += batch_size

            loss.backward()
            optimizer.step()

        total_loss /= num_samples
        model.history['train'] += [total_loss]
        log = " train-loss : " + str(total_loss)
        if val_loader:
            val_loss = evaluate(model, val_loader, device, criterion, steps=val_steps)['loss']
            model.history['val'] += [val_loss]
            log += "  val-loss : " + str(val_loss)

        if (t+1) % verbose == 0:
            print("epoch", t + 1,log)
        if checkpoint and (t+1) % checkpoint == 0:
            save(model, save_path, save_name+'_'+str(len(model.history['train'])))
                

def evaluate(model, loader, device, criterion, steps=None, decoder=None, lm=None, convertor=None, metrics=False, display_out=False):
    num_samples = 0
    stats = {'loss':0.}
    if metrics:
        stats['cer'] = 0.
        stats['wer'] = 0.
    with torch.no_grad():
        n = 0
        model.eval()
        for batch in loader:
            if steps and n == steps:
                break
            n += 1
            x, y, y_len = batch
            batch_size = x.shape[0]
            log_probs = model(x.to(device))
            input_lengths = torch.full((batch_size,), log_probs.shape[2], dtype=torch.long)
            loss = criterion(log_probs.permute(2,0,1), y.to(device), input_lengths, y_len)

            ## metrics
            stats['loss'] += loss.item()
            num_samples += batch_size
            if metrics:
                outputs = decoder(log_probs, lm=lm)
                outputs = post_process(outputs)
                for i, (a,b) in enumerate(zip(outputs,y)):
                    res = convertor(a)
                    gt = convertor(b)
                    cer_ = cer(res,gt)
                    wer_ = wer(res,gt)
                    stats['cer'] += cer_
                    stats['wer'] += wer_
                    if display_out:
                        print('%d.\n  GT:'%(i),gt,'\n  PR:',res,'\n  cer:',cer_,' wer:',wer_,'\n')

    for key in stats.keys():
        stats[key] /= num_samples      
    return stats
    
def plot_hist(model, index=0):
    plt.plot(model.history['train'][index:])
    plt.plot(model.history['val'][index:])
    plt.legend(['train', 'val'])
    plt.show()

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def save(model, path, name):
    torch.save(model.state_dict(), path + str(name) + '.pkl')

def load(model, path, name):
    model.load_state_dict(torch.load(path + str(name) + '.pkl'))
    
    
def remove_dups_and_blanks(seq):
    res = [seq[0]]
    n = len(seq)
    for i in range(1,n):
        if seq[i] != res[-1]:
            res += [seq[i]]
    i = 0
    while i < len(res):
        if res[i] == 0:
            res.pop(i)
        else:
            i += 1
    return res


def post_process(seqs):
    seqs = [remove_dups_and_blanks(seq) for seq in seqs]
    return seqs