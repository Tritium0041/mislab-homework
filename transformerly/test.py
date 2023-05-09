import random

from preprocess import *
from conf import *

def main():
    en_va_sens, zh_va_sens = read_file(
        'translation2019zh_valid.json')
    print(len(en_va_sens), len(zh_va_sens))
    en_vocab = create_vocab(en_va_sens, 30000)
    zh_vocab = create_vocab(zh_va_sens, 30000)
    model = Transformer(src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, trg_sos_idx=trg_sos_idx,
                        encode_vocab_size=len(en_vocab), decode_vocab_size=len(zh_vocab), d_model=d_model,
                        heads=n_heads, max_seq_len=max_len, d_ffhidden=ffn_hidden, n_layers=n_layers, drop=drop_prob
                        ).to("cuda:0")
    model.load_state_dict(torch.load('transformer-29.pt'))
    model.eval()
    while True:
        en_sen = random.choice(en_va_sens)
        en_sen = sentence_to_tensor(en_sen, en_vocab)
        print(tensor_to_sentence(en_sen, en_vocab))
        en_sen = en_sen[0]
        en_sen = torch.from_numpy(en_sen).unsqueeze(0).to("cuda:0")
        zh_sen = [0]
        zh_sen = torch.tensor(zh_sen).unsqueeze(0).to("cuda:0")
        for i in range(max_len):
            zh_sen = model(en_sen, zh_sen)
            zh_sen = zh_sen.argmax(2)
            if zh_sen[0][-1].item() == zh_vocab.index('<eos>'):
                break
        zh_sen = zh_sen.squeeze(0).cpu().numpy()
        zh_sen = [zh_vocab[word] for word in zh_sen]
        zh_sen = " ".join(zh_sen)
        print(zh_sen)


if __name__ == '__main__':
    main()