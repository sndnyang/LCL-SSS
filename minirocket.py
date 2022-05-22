import numpy as np
from tsai.basics import *
import sktime
import sklearn
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *


def normalization(array):
    # max_cols = array.max()
    # min_cols = array.min()
    normalizer_data = preprocessing.Normalizer().fit_transform(array)
    # return (array - min_cols) / (max_cols - min_cols)
    return normalizer_data


def data_clean_and_norm(data, target, dataset='eq'):
    x = data[:, 300:1800]  # [220??, 1500]
    # 0: 爆炸 blast
    # 1: noise 噪声
    # 5: nature 自然地震
    y = np.argmax(target, axis=1)

    if dataset == 'eq2' or dataset == 'ne':
        # eq2 -> only consider  noise 1 and nature earth 5
        # treat nature 5 as category 0
        y[y == 5] = 0
    elif dataset == 'nb':
        # eq2 -> only consider  noise 1 and blast 0
        # remove 0 blast
        index = y != 5
        x = x[index]
        y = y[index]
    elif dataset == 'eb':
        # eq2 -> only consider  blast 0 and nature earth 5
        y[y == 5] = 1
    else:
        # three classes
        y[y == 5] = 2

    norm_x = normalization(x)
    seed = 1
    x_train, x_test, y_train, y_test = train_test_split(norm_x, y, test_size=0.6, random_state=seed)
    seed = 2
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.6, random_state=seed)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == '__main__':
    train_data = loadmat('earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']

    x_train, x_valid, x_test, y_train, y_valid, y_test = data_clean_and_norm(all_data, all_target, dataset='eq')
    print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
    d = 1500
    ppt = x_train.reshape(-1, 1, 1500)

    print(ppt.shape, y_train.shape)
    print("training start")
    model = MiniRocketClassifier()
    timer.start(False)
    model.fit(ppt, y_train)
    pred_y = model.predict(x_valid.reshape(-1, 1, d))
    print(pred_y[:20])
    print(y_valid[:20])
    print("sum", np.sum(pred_y - y_valid))

    t = timer.stop()
    print(f'valid accuracy    : {model.score(x_valid.reshape(-1, 1, d), y_valid):.3%} time: {t}')
    print(f'test accuracy    : {model.score(x_test.reshape(-1, 1, d), y_test):.3%} time: {t}')

    torch.save(model, 'minirocket.pt')

    pred_y = model.pred(x_test)
    torch.save(pred_y, 'pred_y.pt')
    torch.save(y_test, 'real_y.pt')
