import torch.nn as nn
from .train_iter import train_batch_comb
from .latent_optimization import optimize_z
from .maml import maml_optimize_w
import time
from .metric import print_CM
import torch
import math


def train_batch(scaler, model, optimizer, batch, options, log:bool, ad_weight:float, ad:bool):
    cuda = options['cuda']
    batch = [B.cuda(cuda) for B in batch]
    batch_forward_time = 0
    LO_forward_time = 0
    MAML_forward_time = 0
    batch_size = batch[0].shape[0]
    train_log_line = ''

    domain_cls_loss = 0
    # source batch
    cls_repr, outs, source_cls_loss, domain_loss, time_elapsed = train_batch_comb(model, batch[0:3], data_type='source', ad_weight=ad_weight, ad=ad)
    batch_forward_time += time_elapsed
    src_preds = print_CM(y_true=batch[2].tolist(), y_pred=outs[0].max(1)[1].tolist())
    #
    # domain classification on source batch
    domain_cls_loss += domain_loss
    domain_preds = outs[1].max(1)[1].tolist()

    if options['optimize_cls_repr'] is True:
        # optimize source_cls
        time_start = time.perf_counter()
        epsilon = optimizer.param_groups[0]['lr']
        epsilon = epsilon * options['epsilon']
        lo_loss = source_cls_loss if options['only_adver'] is False else domain_loss
        ad_loss = domain_loss
        cls_repr = optimize_z(scaler, optimizer, epsilon, loss=lo_loss, ad_loss=ad_loss, z=cls_repr)
        if ad is True:
            outs = model.second_forward(cls_repr, data_type='source', ad_weight=ad_weight)
        else:
            outs = model.second_forward(cls_repr, data_type='source')
        source_cls_loss = nn.CrossEntropyLoss()(outs, batch[2].cuda(cuda))
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        LO_forward_time += time_elapsed

    if options['optimize_bert'] is True:
        # optimize bert parameters
        time_start = time.perf_counter()
        epsilon = optimizer.param_groups[0]['lr']
        epsilon = epsilon * options['epsilon']
        model = maml_optimize_w(scaler, model, optimizer, epsilon, loss=source_cls_loss)
        cls_repr, outs, source_cls_loss, domain_loss, time_elapsed = train_batch_comb(model, batch[0:3], data_type='source', ad_weight=ad_weight, ad=ad)
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        MAML_forward_time += time_elapsed

    if log is True:
        if math.isnan(source_cls_loss.item()):
            print('source_cls_loss is NaN')
            raise TypeError
        else:
            train_log_line += ','.join([
                str(int(source_cls_loss.item() * 10000) / 10000), src_preds
            ]) + ';'

    # target batch
    cls_repr, outs, target_cls_loss, domain_loss, time_elapsed = train_batch_comb(model, batch[3:6], data_type='target', ad_weight=ad_weight, ad=ad)
    batch_forward_time += time_elapsed
    tgt_preds = print_CM(y_true=batch[-1].tolist(), y_pred=outs[0].max(1)[1].tolist())
    #
    # domain classification
    domain_cls_loss += domain_loss
    domain_preds += outs[1].max(1)[1].tolist()
    domain_labels = [0 for _ in range(0, batch_size)] + [1 for _ in range(0, batch_size)]

    if options['optimize_cls_repr'] is True:
        # optimize target_cls
        time_start = time.perf_counter()
        epsilon = optimizer.param_groups[0]['lr']
        epsilon = epsilon * options['epsilon']
        lo_loss = target_cls_loss if options['only_adver'] is False else domain_loss
        ad_loss = domain_loss
        cls_repr = optimize_z(scaler, optimizer, epsilon, loss=lo_loss, ad_loss=ad_loss, z=cls_repr)
        if ad is True:
            outs = model.second_forward(cls_repr, data_type='target', ad_weight=ad_weight)
        else:
            outs = model.second_forward(cls_repr, data_type='target')
        target_cls_loss = nn.CrossEntropyLoss()(outs, batch[-1].cuda(cuda))
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        LO_forward_time += time_elapsed

    if options['optimize_bert'] is True:
        # optimize bert parameters
        time_start = time.perf_counter()
        epsilon = optimizer.param_groups[0]['lr']
        epsilon = epsilon * options['epsilon']
        model = maml_optimize_w(scaler, model, optimizer, epsilon, loss=target_cls_loss)
        cls_repr, outs, target_cls_loss, domain_loss, time_elapsed = train_batch_comb(model, batch[3:6], data_type='target', ad_weight=ad_weight, ad=ad)
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        MAML_forward_time += time_elapsed

    if log is True:
        if math.isnan(target_cls_loss.item()):
            print('target_cls_loss is NaN')
            raise TypeError
        elif math.isnan(domain_cls_loss.item()):
            print('domain_cls_loss is NaN')
            raise TypeError
        else:
            train_log_line += ','.join([
                str(int(target_cls_loss.item() * 10000) / 10000), tgt_preds
            ]) + ';'
            train_log_line += ','.join([
                str(int(domain_cls_loss.item() * 10000) / 10000),
                print_CM(y_true=domain_labels, y_pred=domain_preds)
            ]) + ';'

    # sum train loss
    train_loss = source_cls_loss + target_cls_loss + domain_cls_loss

    if options['model_name'] in ['mtl_lo', 'LOANT']:
        forward_time = (batch_forward_time, LO_forward_time)
    elif options['model_name'] in ['mtl_maml', 'ant_maml']:
        forward_time = (batch_forward_time, MAML_forward_time)
    else:
        forward_time = batch_forward_time
    print(train_log_line)
    return train_loss, forward_time, train_log_line