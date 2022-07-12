# -*- coding: utf-8 -*-
"""
Created on 2022.05.22

@author: Xiulong Yang
"""
import os
import time
import datetime
import argparse

import sklearn
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from matplotlib import pyplot as plt
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_utils import get_data
from models import *
from callbacks.KerasF1 import KerasF1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet', type=str, choices=['resnet', 'vgg', 'inception', 'alexnet', 'lenet'], help='resnet, vgg, inception, alexnet, lenet')
    parser.add_argument('--shape', default=0, type=int, choices=[0, 1, 2, 3, 4], help='series option, 0: 6000=3x2000, 1: 0:2000, 2: 300:1300, 3: 500:1000')
    parser.add_argument('--gpu-id', type=str, help='', default='1')
    args = parser.parse_args()

    # Use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # 载入数据
    train_data = loadmat('earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']
    shape = args.shape
    model_name = args.model
    logger.add('%s_shape_%d.log' % (model_name, shape))

    select_maps = {0: None, 1: [0, 2000], 2: [300, 1300], 3: [500, 1000], 4: [100, 1700]}
    shape_maps = {0: (-1, 3, 50, 40), 1: (-1, 1, 50, 40), 2: [-1, 1, 40, 25], 3: (-1, 1, 25, 20), 4: [-1, 40, 40, 1]}
    print = logger.info
    print('Model uses %s, data select %d, namely %s->%s' % (model_name, shape, str(select_maps[shape]), str(shape_maps[shape])))

    # load data
    data = get_data(all_data, all_target, dataset='eq', seed=1, shape=shape_maps[shape], select=select_maps[shape])
    x_train, x_valid, x_test, y_train, y_valid, y_test, splits, splits_test = data
    print('data shape %s' % str(x_train.shape) + str(x_valid.shape) + str(x_test.shape))
    # set contains training and validation
    X = np.concatenate([x_train, x_valid])
    y = np.concatenate([y_train, y_valid])

    training = True

    if model_name == 'resnet':
        # 成功
        model = ResNet18([2, 2, 2, 2], n_classes=3)
    elif model_name == 'inception':
        # 成功
        model = Inception10(num_blocks=2, num_classes=3)
    elif model_name == 'vgg':
        # 成功
        model = VGG16(num_classes=3)
    elif model_name == 'alexnet':
        # 不能用于任意长宽的数据
        model = AlexNet8(num_classes=3)
    elif model_name == 'lenet':
        # 不能用于任意长宽的数据
        model = LeNet5(num_classes=3)
    else:
        model = ResNet18([2, 2, 2, 2], n_classes=3)

    if model_name in ['alexnet']:
        model.compile(optimizer='sgd',
                      lr=0.01,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    else:
        model.compile(optimizer='adam',
                      lr=1e-4,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = "./checkpoint/%s_%d_baseline.ckpt" % (model_name, shape)
    if not training and os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    else:
        start = time.time()
        epoch = 30
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                         save_weights_only=True,
                                                         save_best_only=True)
        history = model.fit(x_train, y_train, batch_size=32, epochs=epoch, validation_data=(x_valid, y_valid), validation_freq=1,
                            callbacks=[KerasF1(valid_data=(x_test, y_test)), cp_callback])

        end = time.time()
        model.summary()

        print('data shape %s' % str(x_train.shape) + str(x_valid.shape) + str(x_test.shape))
        print("%d epoch takes %d seconds, %s" % (epoch, end - start, str(datetime.timedelta(seconds=end-start))))

        # show
        # 显示训练集和验证集的acc和loss曲线
        acc = history.history['sparse_categorical_accuracy']
        val_acc = history.history['val_sparse_categorical_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        # plt.show()
        plt.savefig('%s_%d_loss_acc.png' % (model_name, shape))
        plt.close()

    result = model.predict(x_test)
    pred = tf.argmax(result, axis=1)
    if tf.__version__[0] == '1':
        pred = pred.eval(session=tf.compat.v1.Session())
    elif tf.__version__[0] == '2':
        pred = pred.numpy()
    print('Test accuracy %.2f' % sklearn.metrics.accuracy_score(y_test, pred))

    print('precision recall  F1 score in micro: %s' % str(precision_recall_fscore_support(y_test, pred, average='micro')))
    print('precision recall  F1 score in macro: %s' % str(precision_recall_fscore_support(y_test, pred, average='macro')))
    print('precision recall  F1 score in weighted: %s' % str(precision_recall_fscore_support(y_test, pred, average='weighted')))

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot()
    plt.savefig('%s_%d_confusion_matrix.png' % (model_name, shape))
    plt.close()
