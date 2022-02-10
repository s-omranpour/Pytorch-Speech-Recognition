import torch

class Tokenizer:
    def __init__(self, alphabet):
        # reserve 0 for blank label
        alphabet = ['-'] + alphabet
        self.index2char = dict(enumerate(alphabet))
        self.char2index = {v: k for k, v in self.index2char.items()}
        self.vocab_size = len(self.char2index)

    def encode(self, label):
        y = []
        for char in label:
            y.append(self.char2index[char])
        return y
    
    def decode(self, indices, clean=True):
        res = ''
        for index in indices:
            if isinstance(index, torch.Tensor):
                index = index.item()
            res += self.index2char[index]
        if clean:
            return self.clean(res)
        return res

    def clean(self, seq):
        res = '-'
        for c in seq:
            if c != res[-1]:
                res += c
        return res.replace('-', '')