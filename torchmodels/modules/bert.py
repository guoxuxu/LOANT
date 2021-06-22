import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, LayerList, model_name, use_emoji, initialization):
        self.model_name = model_name
        super(BERT, self).__init__()
        if use_emoji is True:
            if initialization == 'random':
                self.bert = LayerList.bert_w_emoji_random
            elif initialization == 'uniform':
                self.bert = LayerList.bert_w_emoji_uniform
            elif initialization == 'xavier':
                self.bert = LayerList.bert_w_emoji_xavier
            else:
                raise TypeError
        else:
            self.bert = LayerList.bert
        if model_name in ['bert', 'bert_lo']:
            self.pooler = LayerList.tgt_pooler
            self.classifier = LayerList.tgt_cls
        elif model_name in ['bert_mlp']:
            self.classifier = LayerList.tgt_cls
        else:
            raise TypeError

    def forward(self, inputs, mask):
        # last layer
        lastlayer_hidden_state, _ = self.bert(input_ids=inputs, attention_mask=mask)
        cls_repr = lastlayer_hidden_state[:, 0, :]

        if self.model_name in ['bert_mlp']:
            scores = self.classification(cls_repr)
        elif self.model_name in ['bert', 'bert_lo']:
            repr = self.pooler(cls_repr)
            scores = self.classification(repr)
        else:
            raise TypeError
        return cls_repr, scores

    def second_forward(self, z):
        scores = self.classification(z)
        return scores

    def classification(self, cls_repr):
        scores = self.classifier(cls_repr)
        return scores