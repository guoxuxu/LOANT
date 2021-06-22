
from . import utils
import torch.nn as nn


class _init_layers(nn.Module):
    def __init__(self, num_all_tokens, num_added_tokens):
        super(_init_layers, self).__init__()
        # pretrained word vector norm = 1.2, bert_w_emoji norm=0.5, uniform norm=0.1, xavier norm=1.2

        '''
        bert initialize added embeddings depends on random seeds
        '''
        self.bert, self.bert_w_emoji_random, self.bert_w_emoji_uniform, self.bert_w_emoji_xavier = utils.get_bert(num_all_tokens, num_added_tokens)

        '''
        The following initialization depends on random seeds
        these modules order cannot change, otherwise initialization changes
        add new modules at the end, if needed. (this can further affect the training epochs shuffle seed, which does not matter more than model initialization)
        '''

        self.src_pooler = nn.Sequential(
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Tanh()
        )
        self.share_pooler = nn.Sequential(
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Tanh()
        )
        self.tgt_pooler = nn.Sequential(
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Tanh()
        )

        # classify aggregated source/target and shared pooled features
        self.src_aggre_cls = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768 * 2, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=2, bias=True),
        )
        self.tgt_aggre_cls = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768 * 2, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=2, bias=True),
        )

        # classify only shared pooled features (No controversy on this module)
        self.domain_disc = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=2, bias=True),
        )

        # the following is for adda2disc
        # classify aggregated source/target and shared pooled features
        self.src_aggre_private_disc = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768 * 2, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=2, bias=True),
        )
        self.tgt_aggre_private_disc = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768 * 2, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=2, bias=True),
        )

        # the following is for mtl_sep, bert
        # classify only source/target pooled features
        self.src_cls = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=2, bias=True),
        )
        self.tgt_cls = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=2, bias=True),
        )

        # classify only source/target pooled features
        self.src_private_disc = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=2, bias=True),
        )
        self.tgt_private_disc = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=2, bias=True),
        )

        # https://github.com/kefirski/pytorch_Highway/blob/master/highway/highway.py
        self.src_highway_pooler = nn.ModuleList([
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Linear(in_features=768, out_features=768, bias=True),
        ])
        self.share_highway_pooler = nn.ModuleList([
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Linear(in_features=768, out_features=768, bias=True),
        ])
        self.tgt_highway_pooler = nn.ModuleList([
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.Linear(in_features=768, out_features=768, bias=True),
        ])
