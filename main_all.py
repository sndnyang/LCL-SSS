# -*- coding: utf-8 -*-
"""
Created on 2022.05.22

@author: Xiulong Yang
"""

import datetime
import os
import sys
import sklearn
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from matplotlib import pyplot as plt
from data_utils import get_data
from models import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # 载入数据
    train_data = loadmat('../EQun/earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']

    # load data
    x_train, x_valid, x_test, y_train, y_valid, y_test, splits, splits_test = get_data(all_data, all_target, dataset='eq', seed=1, shape=(-1, 3, 50, 40))
    # set contains training and validation
    X = np.concatenate([x_train, x_valid])
    y = np.concatenate([y_train, y_valid])

    model_name = 'resnet'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    print("use model?  ", model_name)

    if model_name == 'resnet':
        # 成功
        model = ResNet18([2, 2, 2, 2], n_classes=3)
    elif model_name == 'inception':
        # 成功
        model = Inception10(num_blocks=2, num_classes=3)
    elif model_name == 'vgg':
        # 成功
        model = VGG16(num_classes=3)
    # elif model_name == 'alexnet':
    #     # 不能用于任意长宽的数据
    #     model = AlexNet8(num_classes=3)
    # elif model_name == 'lenet':
    #     # 不能用于任意长宽的数据
    #     model = LeNet5(num_classes=3)
    else:
        model = ResNet18([2, 2, 2, 2], n_classes=3)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = "./checkpoint/%s_baseline.ckpt" % model_name
    # if os.path.exists(checkpoint_save_path + '.index'):
    #     print('-------------load the model-----------------')
    #     model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)

    history = model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_valid, y_valid), validation_freq=1,
                        callbacks=[cp_callback])

    model.summary()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_/' + current_time + '/train'
    test_log_dir = 'logs/gradient_/' + current_time + '/test'

    #    show   ################

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
    plt.savefig('loss_acc.png')
    plt.close()

    result = model.predict(x_test)
    pred = tf.argmax(result, axis=1)
    if tf.__version__[0] == '1':
        pred = pred.eval(session=tf.compat.v1.Session())
    elif tf.__version__[0] == '2':
        pred = pred.numpy()
    print('Test accuracy', sklearn.metrics.accuracy_score(y_test, pred))

    from sklearn.metrics import precision_recall_fscore_support
    print('precision recall  F1 score in micro:', precision_recall_fscore_support(y_test, pred, average='micro'))
    print('precision recall  F1 score in macro:', precision_recall_fscore_support(y_test, pred, average='macro'))

    from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot()
    plt.savefig('resnet_confusion_matrix.png')
    plt.close()
