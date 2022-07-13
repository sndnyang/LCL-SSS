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
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn

from utils import prec_recall_f1
from data_utils import get_data
from matplotlib import pyplot as plt
torch.set_num_threads(1)


class Cls(nn.Module):
    """
    based on Caffe LeNet (https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt)
    """

    def __init__(self, feat=8888):
        super(Cls, self).__init__()
        self.dense = nn.Linear(feat, 3)

    def forward(self, x):
        x = self.dense(x)
        return x


def train(train_loader, model, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    t_loss = 0
    for input_x, label_y in train_loader:
        input_x = input_x.cuda()
        label_y = label_y.cuda()
        logits = model(input_x)
        loss = criterion(logits, label_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_loss = loss.item()
    print('loss %.4f data range (%.4f, %.4f)' % (t_loss, input_x.min().item(), input_x.max().item()))


def test(test_loader, model):
    model.eval()
    preds = []
    ys = []
    acc, n = 0, 0
    for input_x, label_y in test_loader:
        with torch.no_grad():
            input_x = input_x.cuda()
            logits = model(input_x)
            bs_preds = logits.argmax(1).detach().cpu()
            acc += (bs_preds == label_y).sum()
            n += input_x.shape[0]
            preds.extend(bs_preds.numpy())
            ys.extend(label_y.numpy())

    prec_recall_f1(ys, preds)
    acc = acc / n
    return acc, preds


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
    model_name = 'MiniRocket_CLS'
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

        X_feat = get_minirocket_features(X[splits[0]], mrf, chunksize=512, to_np=True)
        X_valid_feat = get_minirocket_features(X[splits[1]], mrf, chunksize=512, to_np=True)
        X_test_feat = get_minirocket_features(x_test, mrf, chunksize=512, to_np=True)

        feats = X_feat.shape[1]
        batch_size = 32
        train_set = TensorDataset(torch.Tensor(X_feat.reshape(-1, feats)), torch.LongTensor(y[splits[0]]))  # create your dataset
        valid_set = TensorDataset(torch.Tensor(X_valid_feat.reshape(-1, feats)), torch.LongTensor(y[splits[1]]))
        test_set = TensorDataset(torch.Tensor(X_test_feat.reshape(-1, feats)), torch.LongTensor(y_test))

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
        print('Get the features')

        # As above, use tsai to bring X_feat into fastai, and train.
        epoch = 30

        model = Cls(X_feat.shape[1])
        model.cuda()

        optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch)
        best_acc = 0
        for epoch in range(epoch):
            train(train_loader, model, optimizer)
            print('Epoch %d' % epoch)
            c_acc, preds = test(test_loader, model)
            if c_acc > best_acc:
                best_acc = c_acc
            sched.step()

        end = time.time()
        print('data shape %s' % str(x_train.shape) + str(x_valid.shape) + str(x_test.shape))
        print("Total time(feature init + %d epochs training) takes %d seconds, %s" % (epoch, end - start, str(datetime.timedelta(seconds=end-start))))
        PATH = Path("./models/MR_feature.pt")
        PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mrf.state_dict(), PATH)
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

    print("Final")
    c_acc, pred = test(test_loader, model)
    print('precision recall  F1 score in micro: %s' % str(precision_recall_fscore_support(y_test, pred, average='micro')))
    print('precision recall  F1 score in macro: %s' % str(precision_recall_fscore_support(y_test, pred, average='macro')))
    print('precision recall  F1 score in weighted: %s' % str(precision_recall_fscore_support(y_test, pred, average='weighted')))

    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
    disp.plot()
    plt.savefig('minirocket_sup_%d_confusion_matrix.png' % shape)
    plt.close()
