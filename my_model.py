import torch.nn as nn
import torch
from transformers import BertModel


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None


def grad_reverse(x, constant):
    return GradReverse.apply(x, constant)


class ANT(nn.Module):
    def __init__(self, resize, num_tokens):
        super(ANT, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        if resize is True:
            self.bert_model.resize_token_embeddings(num_tokens)
        self.bert_model.embeddings.token_type_embeddings.weight.requires_grad_(False)
        self.bert_model.pooler.dense.weight.requires_grad_(False)
        self.bert_model.pooler.dense.bias.requires_grad_(False)

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

        self.domain_disc = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=768, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=768, out_features=2, bias=True),
        )

    def compute(self, src_cls, tgt_cls, src_labels, tgt_labels, ad_weight):
        src_repr = self.src_pooler(src_cls)  #
        src_share_repr = self.share_pooler(src_cls)  #
        src_outs = self.src_aggre_cls(torch.cat((src_repr, src_share_repr), dim=1))
        tgt_repr = self.tgt_pooler(tgt_cls)  #
        tgt_share_repr = self.share_pooler(tgt_cls)  #
        tgt_outs = self.tgt_aggre_cls(torch.cat((tgt_repr, tgt_share_repr), dim=1))
        domain_outs = self.domain_disc(grad_reverse(torch.cat((src_share_repr, tgt_share_repr), dim=0), ad_weight))  ##
        src_loss = nn.CrossEntropyLoss()(src_outs, src_labels)
        tgt_loss = nn.CrossEntropyLoss()(tgt_outs, tgt_labels)
        domain_labels = torch.cat((torch.zeros_like(src_labels), torch.ones_like(tgt_labels)), dim=0)
        dom_loss = nn.CrossEntropyLoss()(domain_outs, domain_labels)
        return src_cls, tgt_cls, src_loss, tgt_loss, dom_loss, src_outs, tgt_outs

    def first_forward(self, src_inputs, src_mask, src_labels, tgt_inputs, tgt_mask, tgt_labels, ad_weight):
        # last layer
        outs = self.bert_model(input_ids=src_inputs, attention_mask=src_mask)
        src_cls = outs[0][:, 0, :]
        outs = self.bert_model(input_ids=tgt_inputs, attention_mask=tgt_mask)
        tgt_cls = outs[0][:, 0, :]
        return self.compute(src_cls, tgt_cls, src_labels, tgt_labels, ad_weight)

    def second_forward(self, src_cls, tgt_cls, src_labels, tgt_labels, ad_weight):
        return self.compute(src_cls, tgt_cls, src_labels, tgt_labels, ad_weight)

    def evaluate(self, inputs, masks, labels, domain):
        outs = self.bert_model(input_ids=inputs, attention_mask=masks)
        cls = outs[0][:, 0, :]
        domain_specific_repr = self.src_pooler(cls) if domain == 'source' else self.tgt_pooler(cls)
        shared_repr = self.share_pooler(cls)
        outs = self.src_aggre_cls(torch.cat((domain_specific_repr, shared_repr), dim=1))
        loss = nn.CrossEntropyLoss()(outs, labels)
        return loss, outs

