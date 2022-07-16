# -*- coding: utf-8 -*-
"""
Created on Sat May 28 08:04:58 2022

@author: yin
"""

import time
import datetime
import sklearn
from data_utils import get_data

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from utils import cluster_acc


if __name__ == '__main__':
    train_data = loadmat('earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']

    shape = 3
    select_maps = {0: None, 1: [0, 2000], 2: [300, 1300], 3: [500, 1000]}
    shape_maps = {0: (-1, 1, 6000), 1: (-1, 1, 2000), 2: [-1, 1, 1000], 3: (-1, 1, 500)}
    # load data
    data = get_data(all_data, all_target, dataset='eq', seed=1, shape=shape_maps[shape], select=select_maps[shape], size=0.95)
    x_train, x_valid, x_test, y_train, y_valid, y_test, splits, splits_test = data
    # set contains training and validation
    X = np.concatenate([x_train, x_valid])
    y = np.concatenate([y_train, y_valid])

    start = time.time()
    X_feat = X.reshape(X.shape[0], -1)
    print(X_feat.shape)

    '''
    print("Euclidean k-means")
    km = TimeSeriesKMeans(n_clusters=3, verbose=True, random_state=1)

    from utils import cluster_acc
    print('Valid accuracy')
    train_valid_acc = cluster_acc(y, y_pred)

    X_feat = x_test.reshape(x_test.shape[0], -1)
    y_pred = km.fit_predict(X_feat)
    print('test accuracy')
    train_valid_acc = cluster_acc(y_test, y_pred)
    end = time.time()
    print("total time (Euclidean k-means training + evaluate cluster accuracy) takes %d seconds, %s" % (end - start, str(datetime.timedelta(seconds=end-start))))


    print("DBA k-means")
    dba_km = TimeSeriesKMeans(n_clusters=3,
                              n_init=2,
                              metric="dtw",
                              verbose=True,
                              max_iter_barycenter=10,
                              random_state=1)
    y_pred = dba_km.fit_predict(X_feat)
    print('test accuracy')
    train_valid_acc = cluster_acc(y_test, y_pred)
    end = time.time()
    print("total time (DBA k-means training + evaluate cluster accuracy) takes %d seconds, %s" % (end - start, str(datetime.timedelta(seconds=end-start))))
    '''

    print("Soft-DTW k-means, super slow, because no parallelization")
    sdtw_km = TimeSeriesKMeans(n_clusters=3,
                               metric="softdtw",
                               n_jobs=-1,
                               metric_params={"gamma": .01},
                               verbose=True,
                               max_iter_barycenter=10,
                               random_state=1)
    y_pred = sdtw_km.fit_predict(X_feat)

    print('train accuracy')
    train_valid_acc = cluster_acc(y, y_pred)

    y_pred = sdtw_km.fit_predict(x_test.reshape(X.shape[0], -1))
    print('test accuracy')
    test_acc = cluster_acc(y_test, y_pred)
    end = time.time()
    print("total time (Soft-DTW k-means training + evaluate cluster accuracy) takes %d seconds, %s" % (end - start, str(datetime.timedelta(seconds=end - start))))
