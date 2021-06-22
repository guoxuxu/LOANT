from torch.utils.data import Dataset
import torch


class DATASET(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.ids = dataframe.id.tolist()
        self.X = dataframe.text.tolist()
        self.y = torch.tensor(dataframe.label.tolist(), dtype=torch.long)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    @property
    def instance_ids(self):
        return self.ids

    @property
    def true_labels(self):
        return self.y.tolist()

    def __getitem__(self, index):
        ins_id = self.ids[index]  # instance ID
        text = self.X[index]
        label = self.y[index]
        tokenized_dict = self.tokenizer.encode_plus(text=text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_len, return_attention_mask=True)
        input_ids = tokenized_dict.get('input_ids')
        attention_mask = tokenized_dict.get('attention_mask')
        input = torch.tensor(input_ids)
        mask = torch.tensor(attention_mask, dtype=torch.long)
        return ins_id, input, mask, label