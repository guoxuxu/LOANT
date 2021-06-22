from torch.utils.data import Dataset
import torch


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