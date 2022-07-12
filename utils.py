import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics import precision_recall_fscore_support
from loguru import logger


def prec_recall_f1(y_true, y_pred):
    a, b, c, d = precision_recall_fscore_support(y_true, y_pred, average='micro')
    results = ' %.4f | %.4f | %.4f' % (a, b, c)
    logger.info('precision recall  F1 score in    micro: %s' % results)
    a, b, c, d = precision_recall_fscore_support(y_true, y_pred, average='macro')
    results = ' %.4f | %.4f | %.4f' % (a, b, c)
    logger.info('precision recall  F1 score in    macro: %s' % results)
    a, b, c, d = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    results = ' %.4f | %.4f | %.4f' % (a, b, c)
    logger.info('precision recall  F1 score in weighted: %s' % results)


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
    prec_recall_f1(y_true, pred)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size, pred


def mine_nearest_neighbors(features, top_k):
    # mine the top k nearest neighbors for every sample
    import faiss
    # features = features.cpu().numpy()
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatL2(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, top_k + 1)  # Sample itself is included

    # evaluate
    return indices
