import torch.nn as nn
import torch


class MTL(nn.Module):
    def __init__(self, LayerList, use_emoji, initialization):
        super(MTL, self).__init__()
        if use_emoji is True:
            if initialization == 'random':
                self.bert = LayerList.bert_w_emoji_random
            elif initialization == 'uniform':
                self.bert = LayerList.bert_w_emoji_uniform
            elif initialization == 'xavier':
                self.bert = LayerList.bert_w_emoji_xavier
        else:
            self.bert = LayerList.bert
        self.src_pooler = LayerList.src_pooler
        self.share_pooler = LayerList.share_pooler
        self.tgt_pooler = LayerList.tgt_pooler
        self.src_aggre_cls = LayerList.src_aggre_cls
        self.tgt_aggre_cls = LayerList.tgt_aggre_cls
        self.domain_disc = LayerList.domain_disc

    def forward(self, inputs, mask, data_type: str):
        # last layer
        lastlayer_hidden_state, _ = self.bert(input_ids=inputs, attention_mask=mask)
        cls_repr = lastlayer_hidden_state[:, 0, :]  # CLS hidden vector

        if data_type == 'source':
            return self.classification(cls_repr, data_type)
        elif data_type == 'target':
            return self.classification(cls_repr, data_type)
        else:
            raise TypeError

    def second_forward(self, z, data_type: str):

        if data_type == 'source':
            cls_repr, (src_scores, domain_scores) = self.classification(z, data_type)
            return src_scores
        elif data_type == 'target':
            cls_repr, (tgt_scores, domain_scores) = self.classification(z, data_type)
            return tgt_scores
        else:
            raise TypeError

    def classification(self, cls_repr, data_type:str):
        share_repr = self.share_pooler(cls_repr)

        if data_type == 'source':
            src_repr = self.src_pooler(cls_repr)
            src_scores = self.src_aggre_cls(torch.cat((src_repr, share_repr), dim=1))
            # ignore domain discriminator loss backpropagation
            domain_scores = self.domain_disc(share_repr.clone().detach())
            return cls_repr, (src_scores, domain_scores)
        elif data_type == 'target':
            tgt_repr = self.tgt_pooler(cls_repr)
            tgt_scores = self.tgt_aggre_cls(torch.cat((tgt_repr, share_repr), dim=1))
            # ignore domain discriminator loss backpropagation
            domain_scores = self.domain_disc(share_repr.clone().detach())
            return cls_repr, (tgt_scores, domain_scores)
        else:
            raise TypeError