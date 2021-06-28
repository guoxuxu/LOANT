import torch
import torch.nn as nn
import time


def train_batch_comb(model, batch, data_type, ad_weight:float, ad:bool):
    time_start = time.perf_counter()
    if ad is True:
        cls_repr, outs = model.forward(inputs=batch[0], mask=batch[1], data_type=data_type, ad_weight=ad_weight)
    else:
        cls_repr, outs = model.forward(inputs=batch[0], mask=batch[1], data_type=data_type)
    time_end = time.perf_counter()
    time_elapsed = time_end - time_start

    if data_type == 'source':
        domain_labels = torch.zeros_like(batch[2])
    else:
        domain_labels = torch.ones_like(batch[2])

    task_loss = nn.CrossEntropyLoss()(outs[0], batch[2])
    domain_loss = nn.CrossEntropyLoss()(outs[1], domain_labels)

    return cls_repr, outs, task_loss, domain_loss, time_elapsed


def train_batch_sin(model, batch):
    time_start = time.perf_counter()
    cls_repr, outs = model.forward(inputs=batch[0], mask=batch[1])
    time_end = time.perf_counter()
    time_elapsed = time_end - time_start

    task_loss = nn.CrossEntropyLoss()(outs, batch[2])
    return cls_repr, outs, task_loss, time_elapsed
