import torch
import torch.nn as nn
from .metric import print_CM


class Evaluater(object):
    def __init__(self, dev_loader, test_loader, dev_logger, test_logger, eyeball_logger, cuda, model_name, mode:str):

        self.dev_logger = dev_logger
        self.test_logger = test_logger
        self.eyeball_logger = eyeball_logger
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model_name = model_name
        self.cuda = cuda
        self.mode = mode
        self.dev_lines = []
        self.test_lines = []

    def log_dev(self):
        for line in self.dev_lines:
            self.dev_logger.log(
                line
            )
            
    def log_test(self):
        for line in self.test_lines:
            self.test_logger.log(
                line
            )

    def run_dev(self, model):
        if self.mode == 'COPft':
            # source
            loss, y_true, y_pred, ins_ids, dom_loss, dom_acc_preds, dom_preds = self.evaluate_batch(model, self.dev_loader[0], data_type='source')
            self.eyeball_logger[0].log(
                ','.join([str(x) for x in y_pred[0:100]]) + ';' + ','.join([str(x) for x in dom_preds[0:100]])
            )
            dev_prints = ','.join([loss, print_CM(y_true, y_pred)])
            dev_prints = ';'.join([dev_prints, dom_acc_preds])
            # target
            loss, y_true, y_pred, ins_ids, dom_loss, dom_acc_preds, dom_preds = self.evaluate_batch(model, self.dev_loader[1], data_type='target')
            self.eyeball_logger[1].log(
                ','.join([str(x) for x in y_pred[0:100]]) + ';' + ','.join([str(x) for x in dom_preds[0:100]])
            )
            dev_prints = ';'.join([dev_prints, ','.join([loss, print_CM(y_true, y_pred)]), dom_acc_preds])
        else:
            loss, y_true, y_pred, ins_ids, _, _, _ = self.evaluate_batch(model, self.dev_loader, data_type=None)
            self.eyeball_logger.log(
                ','.join([str(x) for x in y_pred[0:100]])
            )
            dev_prints = ','.join([loss, print_CM(y_true, y_pred)])

        self.dev_lines += [dev_prints]

    def run_test(self, model):
        if self.mode == 'COPft':
            # source
            loss, y_true, y_pred, ins_ids, dom_loss, dom_acc_preds, _ = self.evaluate_batch(model, self.test_loader[0], data_type='source')
            test_prints = ','.join([loss, print_CM(y_true, y_pred)])
            test_prints = ';'.join([test_prints, dom_acc_preds])
            # target
            loss, y_true, y_pred, ins_ids, dom_loss, dom_acc_preds, _ = self.evaluate_batch(model, self.test_loader[1], data_type='target')
            test_prints = ';'.join([test_prints, ','.join([loss, print_CM(y_true, y_pred)]), dom_acc_preds])
        else:
            loss, y_true, y_pred, ins_ids, _, _, _ = self.evaluate_batch(model, self.test_loader, data_type=None)
            test_prints = ','.join([loss, print_CM(y_true, y_pred)])
        self.test_lines += [test_prints]

    def evaluate_batch(self, model, data_loader, data_type):
        model.eval()
        total_loss = 0
        dom_loss = 0
        ins_ids = []
        pred_labels = []
        true_labels = []
        dom_accurate_preds = 0
        dom_preds = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                batch_size = batch[0].shape[0]
                batch = [B.cuda(self.cuda) for B in batch]
                if self.mode == 'COPft':
                    if self.model_name in ['adda2disc', 'adda2disc_lo', 'adda', 'adda_lo', 'adda_lo_ss']:
                        _, outs = model(inputs=batch[1], mask=batch[2], data_type=data_type, ad_weight=1)
                    else:
                        _, outs = model(inputs=batch[1], mask=batch[2], data_type=data_type)
                    total_loss += nn.CrossEntropyLoss()(outs[0], batch[3]) * batch_size
                    task_preds = outs[1].data.max(1)[1]
                    dom_loss += nn.CrossEntropyLoss()(outs[1], torch.zeros_like(task_preds)) * batch_size
                    pred_labels += outs[0].data.max(1)[1].tolist()
                    if data_type == 'source':
                        dom_accurate_preds += (outs[1].data.max(1)[1] == torch.zeros_like(task_preds)).sum()
                    elif data_type == 'target':
                        dom_accurate_preds += (outs[1].data.max(1)[1] == torch.ones_like(task_preds)).sum()
                    else:
                        raise TypeError
                    dom_preds += outs[1].data.max(1)[1].tolist()
                else:
                    if self.model_name in ['bert_lo']:
                        _, outs = model(inputs=batch[1], mask=batch[2])   # batch[0]: sample id
                    else:
                        outs = model(inputs=batch[1], mask=batch[2])   # batch[0]: sample id
                    total_loss += nn.CrossEntropyLoss()(outs, batch[3]) * batch_size
                    pred_labels += outs.data.max(1)[1].tolist()
                ins_ids += batch[0].tolist()
                true_labels += batch[3].tolist()

        num_total = len(data_loader.dataset)
        average_loss = total_loss.item() / num_total
        average_loss = str(int(average_loss * 10000) / 10000)

        if self.mode == 'COPft':
            dom_loss = dom_loss.item() / num_total
            dom_accurate_preds = dom_accurate_preds.item() / num_total
            dom_loss = str(int(dom_loss * 10000) / 10000)
            dom_accurate_preds = str(int(dom_accurate_preds * 10000) / 10000)

        return average_loss, true_labels, pred_labels, ins_ids, dom_loss, dom_accurate_preds, dom_preds