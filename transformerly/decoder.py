from torch import nn

from embeddings import Transformerembedding
from layers import DecoderLayer


class Decoder(nn.Module):
    def __init__(self,decode_vocab_size,max_seq_len,d_model,d_ffhidden, heads,n_layers,drop=0.1):
        super(Decoder, self).__init__()
        #组装N个DecoderLayer
        self.embed = Transformerembedding(d_model=d_model,vocab_size=decode_vocab_size,max_seq_len=max_seq_len,drop=drop)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,d_ffhidden=d_ffhidden,heads=heads,dropout=drop) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, decode_vocab_size).to('cuda:0')


    def forward(self, trg, src, src_mask=None, trg_mask=None):
        trg = self.embed(trg)

        for layer in self.layers:
            trg = layer(trg, src, src_mask, trg_mask)

        output = self.linear(trg)
        return output