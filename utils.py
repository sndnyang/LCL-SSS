import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
from loguru import logger


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = hungarian(w.max() - w)
    print(row_ind, col_ind)
    print(w)
    from sklearn.metrics import precision_recall_fscore_support
    pred = np.copy(y_pred)
    for i in range(len(row_ind)):
        idx = np.where(y_pred == row_ind[i])
        pred[idx] = col_ind[i]
    logger.info('precision recall  F1 score in micro: %s' % str(precision_recall_fscore_support(y_true, pred, average='micro')))
    logger.info('precision recall  F1 score in macro: %s' % str(precision_recall_fscore_support(y_true, pred, average='macro')))
    logger.info('precision recall  F1 score in weighted: %s' % str(precision_recall_fscore_support(y_true, pred, average='weighted')))
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size
