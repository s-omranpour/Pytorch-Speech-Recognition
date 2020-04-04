import torch

def GreedyDecoder(ctc_matrix):
    ## ctc_matrix shape = (N, C, T)
    seqs = torch.squeeze(torch.topk(ctc_matrix.cpu().detach(), k=1, dim=1)[1], dim=1)
    return seqs