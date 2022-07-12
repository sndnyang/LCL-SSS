import time
import datetime
from tsai.basics import *
import sktime
import sklearn

from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *
from loguru import logger

from scipy.io import loadmat
from data_utils import get_data
from utils import cluster_acc


if __name__ == '__main__':
    train_data = loadmat('earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']

    # load data
    x_train, x_valid, x_test, y_train, y_valid, y_test, splits, splits_test = get_data(all_data, all_target, dataset='eq', seed=1)
    # set contains training and validation
    X = np.concatenate([x_train, x_valid])
    y = np.concatenate([y_train, y_valid])

    logger.add('kmeans_0.log')
    print = logger.info

    start = time.time()
    X_feat = X.reshape(X.shape[0], -1)
    print(X_feat.shape)

    from sklearn.cluster import KMeans
    # model = KMeans(n_clusters=3)
    # y_pred = model.fit_predict(X_feat)
    # print(y_pred[:10])
    #
    # print('Valid accuracy')
    # train_valid_acc = cluster_acc(y, y_pred)

    X_test_feat = x_test.reshape(x_test.shape[0], -1)

    iterations = 50
    centroids = None

    for i in range(iterations):
        kmeans = KMeans(
            max_iter=1,
            n_init=1,
            init=(centroids if centroids is not None else 'k-means++'),
            n_clusters=3,
            random_state=1)
        kmeans.fit(X_feat)
        centroids = kmeans.cluster_centers_
        y_pred = kmeans.predict(X_test_feat)
        print('Iteration %d' % i)
        train_valid_acc, y_pred = cluster_acc(y_test, y_pred)

    y_pred = kmeans.predict(X_test_feat)
    print('test accuracy')
    train_valid_acc, y_pred = cluster_acc(y_test, y_pred, matrix=True)
    end = time.time()
    print("total time (KMeans training + evaluate cluster accuracy) takes %d seconds, %s" % (end - start, str(datetime.timedelta(seconds=end-start))))

