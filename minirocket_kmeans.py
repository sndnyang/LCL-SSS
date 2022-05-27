import time
import datetime
import argparse
from tsai.basics import *
import sktime
import sklearn
from sklearn.cluster import KMeans
from tsai.models.MINIROCKET_Pytorch import *
from tsai.models.utils import *
from loguru import logger

from scipy.io import loadmat
from data_utils import get_data
from matplotlib import pyplot as plt
from utils import cluster_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--shape', default=0, type=int, choices=[0, 1, 2, 3], help='series option, 0: 6000=3x2000, 1: 0:2000, 2: 300:1300, 3: 500:1000')
    parser.add_argument('--gpu-id', type=str, help='', default='1')
    args = parser.parse_args()

    # Use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # 载入数据
    train_data = loadmat('earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']
    shape = args.shape
    model_name = 'MiniRocket_KMeans'
    logger.add('%s_shape_%d.log' % (model_name, shape))

    select_maps = {0: None, 1: [0, 2000], 2: [300, 1300], 3: [500, 1000]}
    shape_maps = {0: (-1, 1, 6000), 1: (-1, 1, 2000), 2: [-1, 1, 1000], 3: (-1, 1, 500)}
    print = logger.info
    print('Model uses %s, data select %d, namely %s->%s' % (model_name, shape, str(select_maps[shape]), str(shape_maps[shape])))

    # load data
    data = get_data(all_data, all_target, dataset='eq', seed=1, shape=shape_maps[shape], select=select_maps[shape])
    x_train, x_valid, x_test, y_train, y_valid, y_test, splits, splits_test = data
    print('data shape %s' % str(x_train.shape) + str(x_valid.shape) + str(x_test.shape))
    # set contains training and validation
    X = np.concatenate([x_train, x_valid])
    y = np.concatenate([y_train, y_valid])

    start = time.time()
    mrf = MiniRocketFeatures(x_train.shape[1], x_train.shape[2]).to(default_device())
    mrf.fit(x_train)

    X_feat = get_minirocket_features(X, mrf, chunksize=512, to_np=True).reshape(X.shape[0], -1)
    print(X_feat.shape)

    model = KMeans(n_clusters=3)
    y_pred = model.fit_predict(X_feat)
    print(y_pred[:10])

    print('Valid accuracy')
    train_valid_acc = cluster_acc(y, y_pred)

    X_feat = get_minirocket_features(x_test, mrf, chunksize=512, to_np=True).reshape(x_test.shape[0], -1)
    y_pred = model.predict(X_feat)
    print('test accuracy')
    train_valid_acc = cluster_acc(y_test, y_pred)
    end = time.time()

    print('data shape %s' % str(x_train.shape) + str(x_valid.shape) + str(x_test.shape))
    print("total time(feature init + KMeans training + evaluate cluster accuracy) takes %d seconds, %s" % (end - start, str(datetime.timedelta(seconds=end-start))))

    PATH = Path("./models/MR_feature.pt")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mrf.state_dict(), PATH)
    # PATH = Path('./models/MR_learner.pkl')
    # learn.export(PATH)
    print('model save path %s' % PATH)