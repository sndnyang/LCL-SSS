import numpy as np
import torch
from fastai.callback.core import Callback
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from loguru import logger


def prec_recall_f1(epoch, y_true, y_pred):
    a, b, c, d = precision_recall_fscore_support(y_true, y_pred, average='micro')
    logger.info('Epoch %d' % epoch)
    results = ' %.4f | %.4f | %.4f' % (a, b, c)
    logger.info('precision recall  F1 score in    micro: %s' % results)
    a, b, c, d = precision_recall_fscore_support(y_true, y_pred, average='macro')
    results = ' %.4f | %.4f | %.4f' % (a, b, c)
    logger.info('precision recall  F1 score in    macro: %s' % results)
    a, b, c, d = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    results = ' %.4f | %.4f | %.4f' % (a, b, c)
    logger.info('precision recall  F1 score in weighted: %s' % results)


class FastAIF1(Callback):
    def on_epoch_begin(self, **kwargs):
        self.correct, self.total = 0, 0
        self.y_pred = torch.tensor([]).cuda()
        self.y_true = torch.tensor([]).cuda()

    def on_batch_end(self, last_output, last_target, **kwargs):
        preds = last_output.argmax(1)
        self.y_pred = torch.cat((self.y_pred, last_output.argmax(dim=1).float()))
        self.y_true = torch.cat((self.y_true, last_target.float()))
        self.correct += ((preds == 0) * (last_target == 0)).float().sum()
        self.total += (preds == 0).float().sum()

    def on_epoch_end(self, **kwargs):
        prec_recall_f1(self.epoch, self.y_true, self.y_pred)
        # self.metric = self.correct / self.total
        self.metric = f1_score(self.y_true, self.y_pred, average='weighted')
