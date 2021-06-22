import os
import pandas as pd
import torch


def get_datasets(source, target:str, tokenizer, max_len:int, fold:str, upsample):
    if source is None:
        from .single_dataset import DATASET
        data_path = os.path.join('data', target)
        if upsample is True:
            train_dataframe = pd.read_csv(os.path.join(data_path, 'train_balanced.csv'))  # balanced to be 51512
        else:
            train_dataframe = pd.read_csv(os.path.join(data_path, 'train.csv'))  #
        dev_dataframe = pd.read_csv(os.path.join(data_path, 'dev.csv'))
        test_dataframe = pd.read_csv(os.path.join(data_path, 'test.csv'))
        train_set = DATASET(dataframe=train_dataframe, tokenizer=tokenizer, max_len=max_len)
        dev_set = DATASET(dataframe=dev_dataframe, tokenizer=tokenizer, max_len=max_len)
        test_set = DATASET(dataframe=test_dataframe, tokenizer=tokenizer, max_len=max_len)
        return train_set, dev_set, test_set
    else:
        from .two_datasets import TWO_DATASET
        from .single_dataset import DATASET
        source_data_path = os.path.join('data', source)
        target_data_path = os.path.join('data', target)
        source_train_dataframe = pd.read_csv(os.path.join(source_data_path, 'train_balanced.csv'))   # balanced to be 51512
        target_train_dataframe = pd.read_csv(os.path.join(target_data_path, 'train_balanced.csv'))
        source_dev_dataframe = pd.read_csv(os.path.join(source_data_path, 'dev.csv'))
        source_test_dataframe = pd.read_csv(os.path.join(source_data_path, 'test.csv'))
        target_dev_dataframe = pd.read_csv(os.path.join(target_data_path, 'dev.csv'))
        target_test_dataframe = pd.read_csv(os.path.join(target_data_path, 'test.csv'))

        train_set = TWO_DATASET(source=source_train_dataframe, target=target_train_dataframe, tokenizer=tokenizer, max_len=max_len)
        source_dev_set = DATASET(dataframe=source_dev_dataframe, tokenizer=tokenizer, max_len=max_len)
        source_test_set = DATASET(dataframe=source_test_dataframe, tokenizer=tokenizer, max_len=max_len)
        target_dev_set = DATASET(dataframe=target_dev_dataframe, tokenizer=tokenizer, max_len=max_len)
        target_test_set = DATASET(dataframe=target_test_dataframe, tokenizer=tokenizer, max_len=max_len)
        return train_set, source_dev_set, source_test_set, target_dev_set, target_test_set


def get_inputs(source, target:str, emoji:bool, fold, upsample):
    if source is None:
        if emoji is True:
            path = os.path.join('torch_loader', 'w_emojis', target, fold)
        else:
            path = os.path.join('torch_loader', 'wo_emojis', target, fold)

        if upsample is True:
            train_set = torch.load(os.path.join(path, 'train_set.pt'))
        else:
            train_set = torch.load(os.path.join(path, 'train_set_original.pt'))
        dev_set = torch.load(os.path.join(path, 'dev_set.pt'))
        test_set = torch.load(os.path.join(path, 'test_set.pt'))
        return train_set, dev_set, test_set

    else:

        if emoji is True:
            path = os.path.join('torch_loader', 'w_emojis', source + target, fold)
        else:
            path = os.path.join('torch_loader', 'wo_emojis', source + target, fold)

        train_set = torch.load(os.path.join(path, 'train_comb.pt'))
        source_dev = torch.load(os.path.join(path, 'source_dev.pt'))
        source_test = torch.load(os.path.join(path, 'source_test.pt'))
        target_dev = torch.load(os.path.join(path, 'target_dev.pt'))
        target_test = torch.load(os.path.join(path, 'target_test.pt'))
        return train_set, source_dev, source_test, target_dev, target_test
