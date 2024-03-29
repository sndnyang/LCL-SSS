import time
import random
import datetime
import argparse
from tsai.basics import *
import sktime
import sklearn

from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *
from loguru import logger

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_utils import get_data
from matplotlib import pyplot as plt
from callbacks.FastAIF1 import FastAIF1
torch.set_num_threads(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--shape', default=0, type=int, choices=[0, 1, 2, 3], help='series option, 0: 6000=3x2000, 1: 0:2000, 2: 300:1300, 3: 500:1000')
    parser.add_argument('--norm', default=1, type=int, choices=[0, 1], help='normalization or not')
    parser.add_argument('--seed', type=int, help='', default=1)
    parser.add_argument('--gpu-id', type=str, help='', default='1')
    args = parser.parse_args()

    # Use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['PYTHONHASHSEED'] = '0'

    random.seed(0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 载入数据
    train_data = loadmat('earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']
    shape = args.shape
    model_name = 'MiniRocket_TSAI'
    logger.add('%s_shape_%d.log' % (model_name, shape))

    select_maps = {0: None, 1: [0, 2000], 2: [300, 1300], 3: [500, 1000]}
    shape_maps = {0: (-1, 1, 6000), 1: (-1, 1, 2000), 2: [-1, 1, 1000], 3: (-1, 1, 500)}
    print = logger.info
    print('Model uses %s, data select %d, namely %s->%s' % (model_name, shape, str(select_maps[shape]), str(shape_maps[shape])))

    # load data
    data = get_data(all_data, all_target, dataset='eq', seed=1, shape=shape_maps[shape], select=select_maps[shape], norm=args.norm)
    x_train, x_valid, x_test, y_train, y_valid, y_test, splits, splits_test = data
    print('data shape %s' % str(x_train.shape) + str(x_valid.shape) + str(x_test.shape))
    # set contains training and validation
    X = np.concatenate([x_train, x_valid])
    y = np.concatenate([y_train, y_valid])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training = not args.eval

    if training is True:
        start = time.time()
        mrf = MiniRocketFeatures(x_train.shape[1], x_train.shape[2], num_features=10000).to(device)
        mrf.fit(x_train)

        X_feat = get_minirocket_features(X, mrf, chunksize=512, to_np=True)
        print('Get the features')

        # As above, use tsai to bring X_feat into fastai, and train.
        tfms = [None, TSClassification()]
        # batch_tfms = TSStandardize(by_sample=True)
        dls = get_ts_dls(X_feat, y, splits=splits, tfms=tfms, batch_tfms=None)

        model = build_ts_model(MiniRocketHead, dls=dls)
        learn = Learner(dls, model, metrics=[accuracy, Precision(average='weighted'), Recall(average='weighted'), F1Score(average='weighted')])
        epoch = 30
        timer.start()
        learn.fit_one_cycle(epoch, 3e-4, cbs=FastAIF1())    # epoch 30 (20 ~ 100),  learning rate 3e-4
        timer.stop()

        end = time.time()
        X_feat = get_minirocket_features(X[splits[1]], mrf, chunksize=512, to_np=True)
        probas, _, pred = learn.get_X_preds(X_feat)
        print('Valid Accuracy %.4f' % sklearn.metrics.accuracy_score(y[splits[1]], pred.astype(int)))

        X_feat = get_minirocket_features(x_test, mrf, chunksize=512, to_np=True)
        probas, _, pred = learn.get_X_preds(X_feat)
        pred = pred.astype(int)
        print('Test accuracy %.4f' % sklearn.metrics.accuracy_score(y_test, pred))

        print('data shape %s' % str(x_train.shape) + str(x_valid.shape) + str(x_test.shape))
        print("Total time(feature init + %d epochs training) takes %d seconds, %s" % (epoch, end - start, str(datetime.timedelta(seconds=end-start))))
        PATH = Path("./models/MR_feature.pt")
        PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mrf.state_dict(), PATH)
        PATH = Path('./models/MR_learner.pkl')
        learn.export(PATH)
        print('model save path %s' % PATH)
    else:
        mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(device)
        PATH = Path("./models/MR_feature.pt")
        mrf.load_state_dict(torch.load(PATH))
        PATH = Path('./models/MR_learner.pkl')
        learn = load_learner(PATH, cpu=False)
        X_feat = get_minirocket_features(x_test, mrf, chunksize=1024, to_np=True)
        probas, _, pred = learn.get_X_preds(X_feat)
        pred = pred.astype(int)
        print('Test accuracy %.2f' % sklearn.metrics.accuracy_score(y_test, pred))

    print('precision recall  F1 score in micro: %s' % str(precision_recall_fscore_support(y_test, pred, average='micro')))
    print('precision recall  F1 score in macro: %s' % str(precision_recall_fscore_support(y_test, pred, average='macro')))
    print('precision recall  F1 score in weighted: %s' % str(precision_recall_fscore_support(y_test, pred, average='weighted')))

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot()
    plt.savefig('minirocket_sup_%d_confusion_matrix.png' % shape)
    plt.close()
