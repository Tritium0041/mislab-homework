import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from model import Transformer
import json
from torchtext.data import get_tokenizer
import jieba
from collections import Counter


# 这边的数据集加载/清洗模块是直接抄的
def read_file(json_path):
    print("开始读取训练集")
    english_sentences = []
    chinese_sentences = []
    tokenizer = get_tokenizer('basic_english')
    with open(json_path, 'r', encoding="UTF-8") as fp:
        for line in fp:
            line = json.loads(line)
            english, chinese = line['english'], line['chinese']
            # Correct mislabeled data
            if not english.isascii():
                english, chinese = chinese, english
            # Tokenize
            english = tokenizer(english)
            chinese = list(jieba.cut(chinese))
            chinese = [x for x in chinese if x not in {' ', '\t'}]
            english_sentences.append(english)
            chinese_sentences.append(chinese)
        fp.close()
    return english_sentences, chinese_sentences


def create_vocab(sentences, max_element=None):
    print("开始创建字典")
    """Note that max_element includes special characters"""

    default_list = ['<sos>', '<eos>', '<unk>', '<pad>']

    char_set = Counter()
    for sentence in sentences:
        c_set = Counter(sentence)
        char_set.update(c_set)

    if max_element is None:
        return default_list + list(char_set.keys())
    else:
        max_element -= 4
        words_freq = char_set.most_common(max_element)
        # pair array to double array
        words, freq = zip(*words_freq)
        return default_list + list(words)


SOS_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3


class TranslationDataset(Dataset):

    def __init__(self, en_tensor: np.ndarray, zh_tensor: np.ndarray):
        super().__init__()
        assert len(en_tensor) == len(zh_tensor)
        self.length = len(en_tensor)
        self.en_tensor = en_tensor
        self.zh_tensor = zh_tensor

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = np.concatenate(([SOS_ID], self.en_tensor[index], [EOS_ID]))
        x = torch.from_numpy(x)
        y = np.concatenate(([SOS_ID], self.zh_tensor[index], [EOS_ID]))
        y = torch.from_numpy(y)
        return x, y


def get_dataloader(en_tensor: np.ndarray,
                   zh_tensor: np.ndarray,
                   batch_size=16):
    def collate_fn(batch):
        x, y = zip(*batch)
        x_pad = pad_sequence(x, batch_first=True, padding_value=PAD_ID)
        y_pad = pad_sequence(y, batch_first=True, padding_value=PAD_ID)

        return x_pad, y_pad

    dataset = TranslationDataset(en_tensor, zh_tensor)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)

    return dataloader


#字符串数组与序号数组的互相转换
def sentence_to_tensor(sentences, vocab):
    vocab_map = {k: i for i, k in enumerate(vocab)}

    def process_word(word):
        return vocab_map.get(word, UNK_ID)

    res = []
    for sentence in sentences:
        sentence = np.array(list(map(process_word, sentence)), dtype=np.int32)
        res.append(sentence)

    return np.array(res,dtype=object)

def tensor_to_sentence(tensor, mapping, insert_space=False):
    res = ''
    first_word = True
    for id in tensor[0]:
        word = mapping[id]

        if insert_space and not first_word:
            res += ' '
        first_word = False

        res += word

    return res


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

