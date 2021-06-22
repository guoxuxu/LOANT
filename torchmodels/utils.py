from transformers import BertModel, BertTokenizer
import torch.nn as nn


def get_bert(num_tokens, num_added_tokens):
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.embeddings.token_type_embeddings.weight.requires_grad_(False)
    bert_model.pooler.dense.weight.requires_grad_(False)
    bert_model.pooler.dense.bias.requires_grad_(False)
    bert_w_emoji = BertModel.from_pretrained("bert-base-uncased")
    bert_w_emoji.embeddings.token_type_embeddings.weight.requires_grad_(False)
    bert_w_emoji.pooler.dense.weight.requires_grad_(False)
    bert_w_emoji.pooler.dense.bias.requires_grad_(False)
    bert_w_emoji.resize_token_embeddings(num_tokens)
    bert_w_emoji_uniform = BertModel.from_pretrained("bert-base-uncased")
    bert_w_emoji_uniform.embeddings.token_type_embeddings.weight.requires_grad_(False)
    bert_w_emoji_uniform.pooler.dense.weight.requires_grad_(False)
    bert_w_emoji_uniform.pooler.dense.bias.requires_grad_(False)
    bert_w_emoji_uniform.resize_token_embeddings(num_tokens)
    bert_w_emoji_uniform.embeddings.word_embeddings.weight[-num_added_tokens:].data.uniform_(-0.01, 0.01)
    bert_w_emoji_xavier = BertModel.from_pretrained("bert-base-uncased")
    bert_w_emoji_xavier.embeddings.token_type_embeddings.weight.requires_grad_(False)
    bert_w_emoji_xavier.pooler.dense.weight.requires_grad_(False)
    bert_w_emoji_xavier.pooler.dense.bias.requires_grad_(False)
    bert_w_emoji_xavier.resize_token_embeddings(num_tokens)
    nn.init.xavier_normal_(bert_w_emoji_xavier.embeddings.word_embeddings.weight[-num_added_tokens:])

    return bert_model, bert_w_emoji, bert_w_emoji_uniform, bert_w_emoji_xavier


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



