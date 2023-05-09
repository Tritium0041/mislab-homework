from torch import nn

from embeddings import Transformerembedding
from layers import EncoderLayer


class Encoder(nn.Module):
    def __init__(self,encode_vocab_size,max_seq_len, d_model,d_ffhidden, heads,n_layers,drop=0.1):
        super(Encoder, self).__init__()
        #组装N个EncoderLayer
        self.embed = Transformerembedding(d_model=d_model,vocab_size=encode_vocab_size,max_seq_len=max_seq_len,drop=drop)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,d_ffhidden=d_ffhidden,heads=heads,dropout=drop) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x