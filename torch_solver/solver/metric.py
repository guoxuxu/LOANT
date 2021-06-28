from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
import torch
import numpy as np


def compute_metrics(y_true, y_pred):
    result = classification_report(y_true, y_pred, digits=4, output_dict=True)
    MCC = matthews_corrcoef(y_true, y_pred)
    R = dict()
    R['pos_f1'], R['pos_rec'] = result['1']['f1-score'], result['1']['recall']
    R['neg_f1'], R['neg_rec'] = result['0']['f1-score'], result['0']['recall']
    R['macro_f1'] = result['macro avg']['f1-score']
    R['macro_rec'] = result['macro avg']['recall']
    R['acc'] = result['accuracy']
    R['mcc'] = MCC
    return R


def compute_CM(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # tn, fp, fn, tp : the order is defined by sklearn
    p = np.sum(y_true)
    n = len(y_true) - p
    return tp, tn, fp, fn, p, n


def print_CM(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    P = np.sum(y_true)
    N = len(y_true) - P
    string = ','.join([str(tp), str(tn), str(fp), str(fn), str(P), str(N)])
    return string


class Metrics(object):
    def __init__(self, tp, tn, fp, fn, P, N):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.P = P
        self.N = N

    def _mcc(self):
        if (self.tp + self.fp == 0) or (self.tn + self.fn) == 0:
            return 0
        else:
            mcc = (self.tp * self.tn - self.fp * self.fn) / np.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
            return int(mcc * 10000)/10000

    def _pos_f1(self):
        val = 2 * self.tp / (2 * self.tp + self.fp + self.fn)
        return int(val * 10000)/10000

    def _neg_f1(self):
        val = 2 * self.tn / (2 * self.tn + self.fp + self.fn)
        return int(val * 10000)/10000

    def _acc(self):
        val = (self.tp + self.tn) / (self.P + self.N)
        return int(val * 10000)/10000

    # sensitivity, recall, hit rate, or true positive rate (TPR)
    def _tpr(self):
        val = self.tp / self.P
        return int(val * 10000)/10000

    # specificity, selectivity or true negative rate (TNR)
    def _tnr(self):
        val = self.tn / self.N
        return int(val * 10000)/10000

    # precision or positive predictive value (PPV)
    def _ppv(self):
        if self.tp + self.fp == 0:
            return 0
        else:
            val = self.tp / (self.tp + self.fp)  # type 1 error
            return int(val * 10000)/10000

    # negative predictive value (NPV)
    def _npv(self):
        if self.tn + self.fn == 0:
            return 0
        else:
            val = self.tn / (self.tn + self.fn)  # type 2 error
            return int(val * 10000)/10000

    # miss rate or false negative rate (FNR)
    def _fnr(self):
        val = self.fn / self.P
        return int(val * 10000)/10000

    # fall-out or false positive rate (FPR)
    def _fpr(self):
        val = self.fp / self.N
        return int(val * 10000)/10000

    def print_metrics(self):
        print('pos_f1 {}, neg_f1 {}, acc {}, mcc {}, pos_recall {}, neg_recall {}, pos_precision {}, negtive_precision {} '.format(
            self._pos_f1(),
            self._neg_f1(),
            self._acc(),
            self._mcc(),
            self._tpr(),
            self._tnr(),
            self._ppv(),
            self._npv()
        ))

    def compute(self):
        pos_f1 = self._pos_f1()
        neg_f1 = self._neg_f1()
        macro_f1 = (pos_f1 + neg_f1) / 2
        macro_f1 = int(macro_f1 * 10000)/10000
        acc = self._acc()
        mcc = self._mcc()
        pos_recall = self._tpr()
        neg_recall = self._tnr()
        macro_recall = (pos_recall + neg_recall) /2
        macro_recall = int(macro_recall * 10000)/10000
        pos_precision = self._ppv()
        neg_precision = self._npv()
        macro_precision = (pos_precision + neg_precision) / 2
        macro_precision = int(macro_precision * 10000)/10000
        return (pos_f1, neg_f1, macro_f1), (pos_recall, neg_recall, macro_recall), (pos_precision, neg_precision, macro_precision), (acc, mcc)