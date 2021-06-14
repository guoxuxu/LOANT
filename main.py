import argparse
from utils import str2bool, set_seed, ensure_dir
import os
import torch
from my_model import ANT
from torch.utils.data import DataLoader
import numpy as np
from transformers import AdamW
from sklearn.metrics import classification_report
import copy
import math
from data import SINGLE_DATASET, TWO_DATASET, init_bert_tokenizer
from tqdm import tqdm
import pandas as pd


def save_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


def training(save_path, train_set, dev_loader, test_loader, model, optimizer, scheduler, lr, batch_size, epochs, LO, epsilon, cuda):
    # Creates once at the beginning of training, mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    best_tgt_dev = -math.inf
    best_model_wgt = None
    best_epoch  =None
    best_step = None

    for epoch in range(0, epochs):
        print(epoch)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        start_step = epoch * len(train_loader)
        model.train()
        for i, batch in tqdm(enumerate(train_loader)):
            batch = [x.cuda(cuda) for x in batch]
            with torch.set_grad_enabled(True):
                p = float(i + start_step) / total_steps
                ad_weight = 2. / (1. + np.exp(-10 * p)) - 1

                with torch.cuda.amp.autocast():
                    src_cls, tgt_cls, src_loss, tgt_loss, dom_loss, src_outs, tgt_outs = \
                        model.first_forward(src_inputs=batch[0], src_mask=batch[1], src_labels=batch[2],
                                            tgt_inputs=batch[3], tgt_mask=batch[4], tgt_labels=batch[5], ad_weight=ad_weight)
                    joint_loss = src_loss + tgt_loss + dom_loss

                if LO is True:
                    # optimize source_cls
                    epsilon = lr * epsilon
                    optimizer.zero_grad()
                    handle_src = src_cls.register_hook(save_grad(src_cls))
                    handle_tgt = tgt_cls.register_hook(save_grad(tgt_cls))
                    scaler.scale(dom_loss).backward(retain_graph=True)
                    delta_zs = src_cls.grad
                    delta_zt = tgt_cls.grad
                    handle_src.remove()
                    handle_tgt.remove()
                    src_cls = src_cls - delta_zs * epsilon
                    tgt_cls = tgt_cls - delta_zt * epsilon

                    _, _, src_new_loss, tgt_new_loss, dom_new_loss, src_outs, tgt_outs = \
                        model.second_forward(src_cls=src_cls, tgt_cls=tgt_cls, src_labels=batch[2], tgt_labels=batch[5], ad_weight=ad_weight)

                    joint_loss = src_loss + tgt_loss + dom_loss + src_new_loss + tgt_new_loss

                optimizer.zero_grad()
                scaler.scale(joint_loss).backward()
                scaler.step(optimizer)
                scaler.update()  # Updates the scale for next iteration
                scheduler.step()

                if i % 100 == 0:
                    model.eval()
                    dev = {'source': 0, 'target': 0}
                    for domain in ['source', 'target']:
                        true_labels, predictions = [], []
                        for i, batch in enumerate(dev_loader[domain]):
                            batch = [x.cuda(cuda) for x in batch]
                            with torch.set_grad_enabled(False):
                                loss, outs = model.evaluate(inputs=batch[0], masks=batch[1], labels=batch[2], domain=domain)
                                predictions += outs.max(dim=1)[1].detach().tolist()
                                true_labels += batch[2].detach().tolist()
                        metrics = classification_report(y_true=true_labels, y_pred=predictions, output_dict=True)
                        dev[domain] = metrics['1']['f1-score']
                    if dev['target'] > best_tgt_dev:
                        best_model_wgt = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        best_step = i
                        best_tgt_dev = dev['target']

    print('training stop, best epoch:{}, step{}, best target dev:{:.4f}'.format(best_epoch, best_step, best_tgt_dev))
    model_name = 'ANT_LO.pt' if LO is True else 'ANT.pt'
    torch.save(best_model_wgt, os.path.join(save_path, model_name))
    test_model = ANT(resize=args.add_emoji, num_tokens=30647)
    test_model.load_state_dict(best_model_wgt)
    
    # test
    test_model.eval()
    with open(os.path.join(save_path, 'source_test.txt'), 'w') as out:
        print('pos_f1, pos_recall, pos_precision, neg_f1, neg_recall, neg_precision, acc', file=out)
    with open(os.path.join(save_path, 'target_test.txt'), 'w') as out:
        print('best epoch:{}, step{}, best target dev:{:.4f}'.format(best_epoch, best_step, best_tgt_dev), file=out)
        print('pos_f1, pos_recall, pos_precision, neg_f1, neg_recall, neg_precision, acc', file=out)

    for domain in ['source', 'target']:
        true_labels, predictions = [], []
        for i, batch in enumerate(test_loader[domain]):
            batch = [x.cuda(cuda) for x in batch]
            with torch.set_grad_enabled(False):
                loss, outs = model.evaluate(inputs=batch[0], masks=batch[1], labels=batch[2], domain=domain)
                predictions += outs.max(dim=1)[1].detach().tolist()
                true_labels += batch[2].detach().tolist()
        metrics = classification_report(y_true=true_labels, y_pred=predictions, output_dict=True)
        with open(os.path.join(save_path, domain + '_test.txt'), 'a') as out:
            print(', '.join([str(metrics['1']['f1-score']), str(metrics['1']['recall']), str(metrics['1']['precision']),
                            str(metrics['0']['f1-score']), str(metrics['0']['recall']), str(metrics['0']['precision']),
                            str(metrics['accuracy'])]), file=out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-cuda", type=int, default=0)
    parser.add_argument("-LO", type=str2bool, default=True)
    parser.add_argument("-epsilon", type=float, default=1.0)
    parser.add_argument("-source", type=str, default='Ghosh')  # Ghosh, Ptacek
    parser.add_argument("-target", type=str, default='SemEval18')  # iSarcasm, SemEval18
    parser.add_argument("-lr", type=float, default=8e-5)
    parser.add_argument("-wd", type=float, default=0)
    parser.add_argument("-batch_size", type=int, default=128)
    parser.add_argument("-epochs", type=int, default=4)
    parser.add_argument("-max_len", type=int, default=100)
    args = parser.parse_args()

    # get data
    tokenizer, num_added_tokens = init_bert_tokenizer(add_new_tokens=False)
    source_data_path = os.path.join('Data', args.source)
    target_data_path = os.path.join('Data', args.target)
    source_train_dataframe = pd.read_csv(os.path.join(source_data_path, 'train_balanced.csv'))  # balanced to be 51512
    target_train_dataframe = pd.read_csv(os.path.join(target_data_path, 'train_balanced.csv'))
    source_dev_dataframe = pd.read_csv(os.path.join(source_data_path, 'dev.csv'))
    source_test_dataframe = pd.read_csv(os.path.join(source_data_path, 'test.csv'))
    target_dev_dataframe = pd.read_csv(os.path.join(target_data_path, 'dev.csv'))
    target_test_dataframe = pd.read_csv(os.path.join(target_data_path, 'test.csv'))

    train_set = TWO_DATASET(source=source_train_dataframe, target=target_train_dataframe, tokenizer=tokenizer, max_len=args.max_len)
    source_dev = SINGLE_DATASET(dataframe=source_dev_dataframe, tokenizer=tokenizer, max_len=args.max_len)
    source_test = SINGLE_DATASET(dataframe=source_test_dataframe, tokenizer=tokenizer, max_len=args.max_len)
    target_dev = SINGLE_DATASET(dataframe=target_dev_dataframe, tokenizer=tokenizer, max_len=args.max_len)
    target_test = SINGLE_DATASET(dataframe=target_test_dataframe, tokenizer=tokenizer, max_len=args.max_len)

    dev_loader = {
        'source': DataLoader(source_dev, batch_size=100, shuffle=False, num_workers=4),
        'target': DataLoader(target_dev, batch_size=100, shuffle=False, num_workers=4),
    }
    test_loader = {
        'source': DataLoader(source_test, batch_size=100, shuffle=False, num_workers=4),
        'target': DataLoader(target_test, batch_size=100, shuffle=False, num_workers=4),
    }

    filename = '_'.join(['lr', str(args.lr), 'wd', str(args.wd), 'bs', str(args.batch_size), 'seed', str(args.seed)])
    method = 'ANT_LO' if args.LO is True else 'ANT'
    save_path = os.path.join('results', args.source + args.target, method, filename)
    print(save_path)
    ensure_dir(save_path)

    # init model
    set_seed(args.seed)
    model = ANT(resize=False, num_tokens=30647).cuda(args.cuda)

    no_decay = ['bias', 'LayerNorm.weight']
    # Separate the `weight` parameters from the `bias` parameters.
    # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
    # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.

    optimizer_grouped_parameters = [
        # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.wd},

        # Filter for parameters which *do* include those.
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    total_steps = int(np.ceil(len(train_set) / args.batch_size)) * args.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_steps, eta_min=0)

    training(save_path=save_path, train_set=train_set, dev_loader=dev_loader, test_loader=test_loader,
             model=model, optimizer=optimizer, scheduler=scheduler,
             lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
             LO=args.LO, epsilon=args.epsilon, cuda=args.cuda)

