import torch
from torch import nn


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=1000):
        super(PositionEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len  #最长句子长度
        self.pe = torch.zeros(max_seq_len, d_model,device='cuda:0')
        self.pe.requires_grad = False
        _2i = torch.arange(0, d_model, 2,device="cuda:0").float()
        self.pos = torch.arange(0, max_seq_len).float().unsqueeze(dim=1).to("cuda:0") #在第1维增加一个维度
        self.pe[:, 0::2] = torch.sin(self.pos / 10000**(_2i / d_model))
        self.pe[:, 1::2] = torch.cos(self.pos / 10000**(_2i / d_model))


    def forward(self, x):
        batch_size,seq_len = x.size()
        output = self.pe[:seq_len, :]
        return output

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class Transformerembedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_seq_len,drop):
        super(Transformerembedding,self).__init__()
        self.tok_embed = TokenEmbedding(vocab_size,d_model)
        self.pos_embed = PositionEncoding(d_model,max_seq_len)
        self.drop = nn.Dropout(drop)


    def forward(self,x):
        tok_embed = self.tok_embed(x)
        pos_embed = self.pos_embed(x)
        return self.drop(tok_embed + pos_embed)
