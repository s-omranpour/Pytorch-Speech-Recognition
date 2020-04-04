import torch
import torch.nn as nn
import torch.nn.functional as F


def get_same_pad(k, s, d):
    assert not (s > 1 and d > 1)
    if s > 1:
        return (k-s+1)//2
    return (k-1)*d//2


class ResBlock(nn.Module):

    def __init__(self, in_c, out_c, kernel, stride, dilation, dropout):
        
        super(ResBlock, self).__init__()
        conv1 = nn.Conv1d(in_c, out_c, kernel, padding=get_same_pad(kernel, 1, dilation), dilation=dilation)
        bn1 = nn.BatchNorm1d(out_c)
        relu1 = nn.ReLU()
        do1 = nn.Dropout(p=dropout)
        self.conv_block = nn.Sequential(conv1, bn1, relu1, do1)
        self.scale = nn.Conv1d(in_c, out_c, 1)
        self.pool = None
        if stride > 1:
            self.pool = nn.AvgPool1d(stride, padding=get_same_pad(stride, stride, 1))

    def forward(self, x):
        h = self.conv_block(x)
        scaled = self.scale(x)
        h += scaled
        return self.pool(h) if self.pool else h


class StreamBlock(nn.Module):
    def __init__(self, in_channel, out_channels, kernels, strides, dilation, dropout, n_attention_heads, n_hidden):
        super(StreamBlock, self).__init__()
        
        self.conv_blocks = nn.ModuleList()
        n = len(kernels)
        for i in range(n):
            in_c = in_channel if i == 0 else out_channels[i-1]
            block = ResBlock(in_c, out_channels[i], kernels[i], strides[i], dilation, dropout)
            self.conv_blocks.append(block)
                
        self.attention = nn.MultiheadAttention(out_channels[-1], n_attention_heads)
        self.lnorm1 = nn.LayerNorm([out_channels[-1]])
        self.ff = nn.Linear(out_channels[-1], n_hidden)
        self.lnorm2 = nn.LayerNorm([n_hidden])
#         self.relu = nn.ReLU()


    def forward(self, x):
        h = x
        for block in self.conv_blocks:
            h = block(h)
        att_in = h.permute(2,0,1)
        att_out, att_weight = self.attention(att_in,att_in,att_in)
        h = att_out.permute(1,0,2)
        h = self.lnorm1(h)
        h = self.lnorm2(self.ff(h))
        return h #self.relu(h)
        
        

class MultiStreamSelfAttentionModel(nn.Module):
    def __init__(self, n_streams, in_channels, num_classes , out_channels, kernels, 
                 strides, dilations, n_attention_heads, n_hidden, dropout):
        
        super(MultiStreamSelfAttentionModel, self).__init__()
        self.history = {'train':[], 'val':[]}
        
        self.streams = nn.ModuleList()
        for i in range(n_streams):
            stream = StreamBlock(in_channels, out_channels, kernels, strides, dilations[i],
                                 dropout, n_attention_heads, n_hidden)
            self.streams.append(stream)
            
        self.dense = nn.Linear(n_hidden*n_streams, num_classes, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_classes)
        self.do = nn.Dropout(p=dropout)
        
    def forward(self, x):
        encs = []
        for stream in self.streams:
            encs += [stream(x)]
        enc = torch.cat(encs, dim=2)
        h = self.dense(enc)
        h = h.permute(0,2,1)
        h = self.do(self.bn(self.relu(h)))
        return F.log_softmax(h, dim=1)