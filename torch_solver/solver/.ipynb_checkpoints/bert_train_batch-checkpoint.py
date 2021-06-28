from .train_iter import train_batch_sin
from .metric import print_CM
from .latent_optimization import optimize_z
import torch.nn as nn


def train_batch(scaler, model, optimizer, batch, options, log:bool, **kwargs):
    # batch: sample id, vocab id, mask, label

    cuda = options['cuda']
    batch = [B.cuda(cuda) for B in batch[1:]]
    batch_forward_time = 0
    train_log_line = ''

    cls_repr, outs, task_loss, time_elapsed = train_batch_sin(model, batch)
    batch_forward_time += time_elapsed

    if options['optimize_cls_repr'] is True:
        epsilon = optimizer.param_groups[0]['lr']
        epsilon = epsilon * options['epsilon']
        cls_repr = optimize_z(scaler, optimizer, epsilon, loss=task_loss, ad_loss=None, z=cls_repr, multi_obj=False)
        outs = model.second_forward(cls_repr)
        task_loss = nn.CrossEntropyLoss()(outs, batch[2])

    if log is True:
        train_log_line = ','.join([str(int(task_loss.item() * 10000) / 10000), print_CM(y_true=batch[2].tolist(), y_pred=outs.max(1)[1].tolist())]) + ';'
    return task_loss, batch_forward_time, train_log_line