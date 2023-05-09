import math

import torch
from torch import nn


class SingleDotProductAttention(nn.Module):
    def __init__(self):
        super(SingleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1) #-1就是横行

    def forward(self, q, k, v, mask=None):
        batch_size, head, length, d_model = k.size()
        kT = k.transpose(2, 3)
        attention = torch.matmul(q, kT) / math.sqrt(d_model)
        if mask is not None:
            attention = attention.masked_fill(mask==0, -1e9)
        attention = self.softmax(attention)
        context = attention @ v
        return context, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.attention = SingleDotProductAttention()
        self.Q_linear = nn.Linear(d_model, d_model)
        self.K_linear = nn.Linear(d_model, d_model)
        self.V_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)


    def split_tensor_by_head(self,tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.heads
        tensor = tensor.view(batch_size, length, self.heads, d_tensor).transpose(1, 2)
        return tensor

    def concat_tensor_by_head(self, tensor):
        batch_size, _, length, d_tensor = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_tensor * self.heads)
        return tensor


    def forward(self, q, k, v, mask=None):
        q ,k ,v = self.Q_linear(q), self.K_linear(k), self.V_linear(v)
        q,k,v = self.split_tensor_by_head(q), self.split_tensor_by_head(k), self.split_tensor_by_head(v)
        out,attention = self.attention(q,k,v,mask)
        out = self.concat_tensor_by_head(out)
        out = self.output_linear(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self,d_model,e=1e-12):
        super(LayerNorm, self).__init__()
        self.d_model = d_model
        self.e = e
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))


    def forward(self,x):
        mean = x.mean(-1,keepdim = True)
        var = x.var(-1, unbiased=False, keepdim=True) #-1对最后一维操作
        out = (x-mean)/torch.sqrt(var+self.e)
        out = self.alpha*out+self.bias
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.linear1 = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden, d_model)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model,d_ffhidden, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        #组装注意力机制和前馈神经网络 用残差连接
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model) # LayerNorm是对最后一维进行归一化
        self.attn = MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model,hidden=d_ffhidden,dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        #计算attention
        x2 = x
        x = self.attn(x,x,x,mask)
        #过一层norm
        x = self.dropout_1(x)
        x = self.norm_1(x+x2)
        #过一层ff
        x2 = x
        x = self.ff(x)
        #norm
        x = self.dropout_2(x)
        x = self.norm_2(x+x2)
        return x

class DecoderLayer(nn.Module):
    def __init__(self,d_model,d_ffhidden,heads,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.norm_3 = LayerNorm(d_model)

        self.attn_self = MultiHeadAttention(d_model,heads)
        self.attn_enc_dec = MultiHeadAttention(d_model,heads)
        self.ff = FeedForward(d_model,d_ffhidden,dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)


    def forward(self,decode,encode,trg_mask,src_mask=None):
        decode2 = decode
        decode =self.attn_self(decode,decode,decode,trg_mask)

        decode = self.dropout_1(decode)
        decode = self.norm_1(decode+decode2)

        if encode is not None:
            decode2=decode
            decode = self.attn_enc_dec(decode,encode,encode,src_mask)
            decode = self.dropout_2(decode)
            decode = self.norm_2(decode+decode2)


        decode2 = decode
        decode = self.ff(decode)
        decode = self.dropout_3(decode)
        decode = self.norm_3(decode+decode2)

        return decode
