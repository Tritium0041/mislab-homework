import math

from torch import optim

from conf import *
from model import Transformer
from preprocess import *


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src = src.to("cuda:0")
        trg = trg.to("cuda:0")
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg1 = trg[:, 1:].contiguous().view(-1).type(torch.LongTensor).to("cuda:0")
        del src, trg
        loss = criterion(output, trg1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        if i % 100 == 0:
            print("batch {} loss {}".format(i, loss.item()))
    return epoch_loss / len(iterator)


def main():
    en_va_sens, zh_va_sens = read_file(
        'translation2019zh_valid.json')
    print(len(en_va_sens), len(zh_va_sens))
    en_vocab = create_vocab(en_va_sens, 30000)
    zh_vocab = create_vocab(zh_va_sens, 30000)
    print(len(en_vocab))
    print(len(zh_vocab))
    print(en_vocab[:10])
    print(zh_vocab[:10])

    loader = get_dataloader(sentence_to_tensor(en_va_sens, en_vocab), sentence_to_tensor(zh_va_sens, zh_vocab),
                            batch_size=batch_size)
    model = Transformer(src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx, trg_sos_idx=trg_sos_idx,
                        encode_vocab_size=len(en_vocab), decode_vocab_size=len(zh_vocab), d_model=d_model,
                        heads=n_heads, max_seq_len=max_len, d_ffhidden=ffn_hidden, n_layers=n_layers, drop=drop_prob
                        ).to("cuda:0")
    optimizer = optim.Adam(model.parameters(), lr=init_lr, eps=adam_eps, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    best_valid_loss = float('inf')
    epoch = 5
    for epoch in range(epoch):
        start_time = time.time()
        train_loss = train(model, loader, optimizer, criterion, clip)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print("Epoch {} | Time {}m {}s".format(epoch, epoch_mins, epoch_secs))
        print("Train Loss {}".format(train_loss))
        if train_loss < best_valid_loss:
            best_valid_loss = train_loss
            torch.save(model.state_dict(), f'transformer-{epoch}.pt')


if "__main__" == __name__:
    en_va_sens, zh_va_sens = read_file(
        'translation2019zh_valid.json')
    print(len(en_va_sens), len(zh_va_sens))
    en_vocab = create_vocab(en_va_sens, 30000)
    zh_vocab = create_vocab(zh_va_sens, 30000)
    print(len(en_vocab))
    print(len(zh_vocab))
    print(en_vocab[:10])
    print(zh_vocab[:10])
    a = sentence_to_tensor([en_va_sens[0]],en_vocab)
    print(tensor_to_sentence(a,en_vocab))
