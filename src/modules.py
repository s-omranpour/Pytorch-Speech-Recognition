import torch
from torch import nn

def get_same_pad(k, s, d):
    assert not (s > 1 and d > 1)
    if s > 1:
        return (k-s+1)//2
    return (k-1)*d//2


class FactorizedConv(nn.Module):
    def __init__(self, in_c, d_bottleneck, out_c, kernel=2, dilation=1):
        super().__init__()
        pad = get_same_pad(kernel, 1, dilation)
        self.f1 = nn.Conv1d(in_c, d_bottleneck, kernel, dilation=dilation, padding=pad if dilation % 2 == 0 else pad + 1)
        self.f2 = nn.Conv1d(d_bottleneck, out_c, kernel, dilation=dilation, padding=pad)

    def semi_orth_obj(self):
        w = self.f1.weight
        m = w.reshape(w.shape[0], w.shape[1] * w.shape[2]).T
        p = torch.mm(m, m.T)
        q = p - torch.eye(p.shape[0])
        return torch.trace(torch.mm(q, q.T))

    def forward(self, x):
        h = self.f1(x)
        h = self.f2(h)
        o = self.semi_orth_obj()
        return h, o


class ResBlock(nn.Module):

    def __init__(self, in_c, d_bottleneck, out_c, kernel, stride, dilation, dropout):
        super().__init__()

        self.conv = FactorizedConv(in_c, d_bottleneck, out_c, kernel, dilation)
        self.post = nn.Sequential(
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.scale = nn.Conv1d(in_c, out_c, 1)
        # if stride > 1:
        self.pool = nn.AvgPool1d(stride, padding=get_same_pad(stride, stride, 1))

    def forward(self, x):
        h, l = self.conv(x)
        h = self.post(h)
        scaled = self.scale(x)
        h += scaled
        return self.pool(h), l



class StreamBlock(nn.Module):
    def __init__(self, in_channel, d_model, dilation, dropout=.1):
        super(StreamBlock, self).__init__()
        
        self.conv_blocks = nn.ModuleList([
            ResBlock(in_channel, 128, d_model, 2, 3, dilation, dropout),
            ResBlock(d_model, 128, d_model, 2, 1, dilation, dropout),
            ResBlock(d_model, 128, d_model, 2, 1, dilation, dropout),
        ])
                
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, dropout=dropout, batch_first=True)
        self.lnorm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model , 128)
        self.ff2 = nn.Linear(128, d_model)
        self.lnorm2 = nn.LayerNorm(d_model)


    def forward(self, x):
        h = x
        orth_losses = []
        for conv in self.conv_blocks:
            h, l = conv(h)
            orth_losses += [l]
        h = h.transpose(1, 2)
        att_out, _ = self.attention(h, h, h)
        h = self.lnorm1(h)
        h = self.ff2(self.ff1(h))
        h = self.lnorm2(h)
        return h, sum(orth_losses)