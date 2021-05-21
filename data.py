import os
import torch
import argparse
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils import ensure_dir, str2bool


class SINGLE_DATASET(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.X = dataframe.text.tolist()
        self.y = torch.tensor(dataframe.label.tolist(), dtype=torch.long)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        text = self.X[index]
        label = self.y[index]
        tokenized_dict = self.tokenizer.encode_plus(text=text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_len, return_attention_mask=True)
        input_ids = tokenized_dict.get('input_ids')
        attention_mask = tokenized_dict.get('attention_mask')
        input = torch.tensor(input_ids)
        mask = torch.tensor(attention_mask, dtype=torch.long)
        return input, mask, label


class TWO_DATASET(Dataset):
    def __init__(self, source, target, tokenizer, max_len:int):
        self.source_X = source.text.tolist()
        self.source_y = torch.tensor(source.label.tolist(), dtype=torch.long)
        self.target_X = target.text.tolist()
        self.target_y = torch.tensor(target.label.tolist(), dtype=torch.long)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.source_X)

    def __getitem__(self, index):
        # source
        text = self.source_X[index]
        source_label = self.source_y[index]
        tokenized_dict = self.tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_len, return_attention_mask=True)
        input_ids = tokenized_dict.get('input_ids')
        attention_mask = tokenized_dict.get('attention_mask')
        source_input = torch.tensor(input_ids)
        source_mask = torch.tensor(attention_mask, dtype=torch.long)
        # target
        text = self.target_X[index]
        target_label = self.target_y[index]
        tokenized_dict = self.tokenizer.encode_plus(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_len, return_attention_mask=True)
        input_ids = tokenized_dict.get('input_ids')
        attention_mask = tokenized_dict.get('attention_mask')
        target_input = torch.tensor(input_ids)
        target_mask = torch.tensor(attention_mask, dtype=torch.long)
        return source_input, source_mask, source_label, target_input, target_mask, target_label


def init_bert_tokenizer(add_new_tokens:bool):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if add_new_tokens is True:
        emojis_path = 'data/EMOJI/INVT.txt'
        tags_path = 'data/TAG/INVT.txt'
        emojis = []
        tags = []
        with open(emojis_path, 'r') as f:
            for line in f:
                emojis += [line.strip('\n')]
        with open(tags_path, 'r') as f:
            for line in f:
                tags += [line.strip('\n')]
        tokenizer.add_tokens(emojis + tags)
        return tokenizer, len(emojis + tags)
    else:
        return tokenizer, 0