import numpy as np
import tensorflow.keras as keras
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, recall_score, precision_score
from loguru import logger


def prec_recall_f1(epoch, y_true, y_pred):
    a, b, c, d = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print('Epoch %d' % epoch)
    results = ' %.4f | %.4f | %.4f' % (a, b, c)
    logger.info('precision recall  F1 score in    micro: %s' % results)
    a, b, c, d = precision_recall_fscore_support(y_true, y_pred, average='macro')
    results = ' %.4f | %.4f | %.4f' % (a, b, c)
    logger.info('precision recall  F1 score in    macro: %s' % results)
    a, b, c, d = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    results = ' %.4f | %.4f | %.4f' % (a, b, c)
    logger.info('precision recall  F1 score in weighted: %s' % results)


class KerasF1(keras.callbacks.Callback):
    # https://keras.io/examples/keras_recipes/sklearn_metric_callbacks/
    # https://developpaper.com/tf-keras-implements-f1-score-precision-recall-and-other-metrics/
    def __init__(self, valid_data):
        super(KerasF1, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        prec_recall_f1(epoch, val_targ, val_predict)
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return
