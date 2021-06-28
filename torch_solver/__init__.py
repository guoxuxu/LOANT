from torch.utils.data import DataLoader
import torch
import numpy as np
from transformers import AdamW
from . import solver
from .utils import FileLogger, nvidia_free_mem

'''
train batch: 0-input ids, 1-mask, 2-label, ( 3-input ids, 4-mask, 5-label )
dev & test batch: 0-instance id, 1-input ids, 2-mask, 3-label
'''


def run(model, train_set, dev_set, test_set, options, mode: str):

    cuda = options['cuda']
    # options['init_free_mem'] = nvidia_free_mem(cuda)
    model = model.cuda(cuda)

    if mode == 'COPft':
        dev_loader = [DataLoader(dev_set[i], batch_size=options.get('batch_size'), shuffle=False, num_workers=4) for i in range(0, len(dev_set))]
        test_loader = [DataLoader(test_set[i], batch_size=options.get('batch_size'), shuffle=False, num_workers=4) for i in range(0, len(test_set))]
    else:
        dev_loader = DataLoader(dev_set, batch_size=options['batch_size'], shuffle=False, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=options['batch_size'], shuffle=False, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=options['lr'])
    total_steps = int(np.ceil(len(train_set) / options['batch_size'])) * options['training_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_steps, eta_min=0)

    metric_to_save = ['loss', 'TP', 'TN', 'FP', 'FN', 'P', 'N']

    if options['model_name'] in ['mtl_lo', 'adda_lo']:
        resource = ['forward_time', 'LO_time', 'backward_time', 'memory_allocated', 'memory_reserved', 'lr']
    elif options['model_name'] in ['mtl_maml']:
        resource = ['forward_time', 'MAML_time', 'backward_time', 'memory_allocated', 'memory_reserved', 'lr']
    else:
        resource = ['forward_time', 'backward_time', 'memory_allocated', 'memory_reserved', 'lr']

    if mode == 'COPft':
        resource += ['grad_norm_sh', 'weight_norm_sh']

    if mode == 'COPft':
        train_header = ['src_' + name for name in metric_to_save] + ['tgt_' + name for name in metric_to_save] + ['dom_' + name for name in metric_to_save] + resource
        val_header = ['src_' + name for name in metric_to_save] + ['src_dom_accurate_num'] + ['tgt_' + name for name in metric_to_save] + ['tgt_dom_accurate_num']
    else:
        train_header = metric_to_save + resource
        val_header = metric_to_save

    train_logger = FileLogger(
        path=options['path'],
        filename='train.txt',
        seed=options['seed'],
        header=train_header)

    dev_logger = FileLogger(
        path=options['path'],
        filename='validation.txt',
        seed=options['seed'],
        header=val_header)

    test_logger = FileLogger(
        path=options['path'],
        filename='test.txt',
        seed=options['seed'],
        header=val_header)

    if mode == 'COPft':
        header = [str(idx) for idx in dev_loader[0].dataset.instance_ids[0:100]]
        source_val_eyeball_logger = FileLogger(
            path=options['path'],
            filename='source_val_eyeball.txt',
            seed=options['seed'],
            header=header)
        first_line = 'True Label;domain Label: \n' + ','.join([str(idx) for idx in dev_loader[0].dataset.true_labels[0:100]]) + ';' + ','.join([str(0) for _ in range(0, 100)]) + '\nPred Label;domain pred:\n'
        source_val_eyeball_logger.log(first_line)
        target_val_eyeball_logger = FileLogger(
            path=options['path'],
            filename='target_val_eyeball.txt',
            seed=options['seed'],
            header=header)
        first_line = 'True Label;domain Label: \n' + ','.join([str(idx) for idx in dev_loader[1].dataset.true_labels[0:100]]) + ';' + ','.join([str(1) for _ in range(0, 100)]) + '\nPred Label;domain pred:\n'
        target_val_eyeball_logger.log(first_line)
        eyeball_logger = [source_val_eyeball_logger, target_val_eyeball_logger]
    else:
        header = [str(idx) for idx in dev_loader.dataset.instance_ids[0:100]]
        eyeball_logger = FileLogger(
            path=options['path'],
            filename='val_eyeball.txt',
            seed=options['seed'],
            header=header)
        first_line = 'True Label: \n' + ','.join([str(idx) for idx in dev_loader.dataset.true_labels[0:100]]) + '\nPred Label:\n'
        eyeball_logger.log(first_line)

    eval_solver = solver.evaluate(dev_loader, test_loader, dev_logger, test_logger, eyeball_logger, cuda, model_name=options['model_name'], mode=mode)
    solver.train(model, optimizer, scheduler, train_set, train_logger, eval_solver, options, mode)

