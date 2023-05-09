import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.cuda as cuda
from torch.autograd import Variable
from torch.utils._contextlib import F

from decoder import Decoder
from layers import *
from encoder import Encoder






#构建Transformer模型
#我来组成头部!
class Transformer(nn.Module):
    def __init__(self,src_pad_idx,trg_pad_idx,trg_sos_idx,encode_vocab_size,decode_vocab_size,d_model,heads,max_seq_len,d_ffhidden,n_layers,drop=0.1):
        super(Transformer,self).__init__()
        self.encoder = Encoder(encode_vocab_size,max_seq_len,d_model,d_ffhidden,heads,n_layers,drop)
        self.decoder = Decoder(decode_vocab_size,max_seq_len,d_model,d_ffhidden,heads,n_layers,drop)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx

    def make_pad_mask(self,q,k,q_pad_idx,k_pad_idx):
        len_q,len_k = q.size(1),k.size(1)

        k = k.ne(k_pad_idx).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1,1,len_q,1)

        q = q.ne(q_pad_idx).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1,1,1,len_k)

        mask = k & q
        return mask

    def make_no_peak_mask(self,q,k):
        len_q,len_k = q.size(1),k.size(1)
        mask = torch.tril(torch.ones(len_q,len_k)).type(torch.BoolTensor).to('cuda:0')
        return mask

    def forward(self,src,trg):
        src_mask = self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        src_trg_mask = self.make_pad_mask(trg,src,self.trg_pad_idx,self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx) * self.make_no_peak_mask(trg,trg)
        enc_src = self.encoder(src,src_mask)
        out = self.decoder(trg,enc_src,trg_mask,src_trg_mask)
        return out
