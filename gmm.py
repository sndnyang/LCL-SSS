import time
import datetime
from tsai.basics import *
import sktime
import sklearn

from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *

from scipy.io import loadmat
from data_utils import get_data
from matplotlib import pyplot as plt


if __name__ == '__main__':
    train_data = loadmat('earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']

    # load data
    x_train, x_valid, x_test, y_train, y_valid, y_test, splits, splits_test = get_data(all_data, all_target, dataset='eq', seed=1)
    # set contains training and validation
    X = np.concatenate([x_train, x_valid])
    y = np.concatenate([y_train, y_valid])

    training = True
    if len(sys.argv) > 1:
        training = False
    print("train or evaluate? ", 'Train!' if training is True else 'Evaluate!')
    start = time.time()
    X_feat = X.reshape(X.shape[0], -1)
    print(X_feat.shape)

    from sklearn.mixture import GaussianMixture
    model = GaussianMixture(n_components=3,
                            covariance_type="full",
                            random_state=2333,
                            )
    model.fit(X_feat)
    y_pred = model.fit_predict(X_feat)
    print(y_pred[:10])

    from utils import cluster_acc
    print('Valid accuracy')
    train_valid_acc = cluster_acc(y, y_pred)

    X_feat = x_test.reshape(x_test.shape[0], -1)
    y_pred = model.predict(X_feat)
    print('test accuracy')
    train_valid_acc = cluster_acc(y_test, y_pred)
    end = time.time()
    print("Datashape, train, valid, test: ", x_train.shape, x_valid.shape, x_test.shape)
    print("total time (GM training + evaluate cluster accuracy) takes %d seconds, " % (end - start), str(datetime.timedelta(seconds=end-start)))
