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
from utils import cluster_acc, mine_nearest_neighbors

EPS = 1e-8


def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight=2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight  # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss


class ClusterHead(nn.Module):
    """
    based on Caffe LeNet (https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt)
    """

    def __init__(self, feat=8888):
        super(ClusterHead, self).__init__()
        self.bn = nn.BatchNorm1d(feat)
        self.dense = nn.Linear(feat, 3)

    def forward(self, x):
        x = self.bn(x)
        x = self.dense(x)
        return x


def test(test_loader, model):
    model.eval()
    t_loss = 0
    preds = []
    y_t = []
    for x, y in test_loader:
        x = x.cuda()
        h = model(x)
        y_t.extend(y.numpy())
        preds.extend(h.argmax(1).detach().cpu().numpy())
    return preds, y_t


def train(train_loader, model, optimizer):
    model.train()
    t_loss = 0
    for x, y in train_loader:
        x = x.cuda()
#         i = np.random.randint(3)
#         x_y = torch.Tensor(X_feat)[y[:, i]].cuda()
        y = y.cuda()
        anchor = model(x)
        neighbor = model(y)
        loss = criterion(anchor, neighbor)
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()
        t_loss += loss[0].item()
    print(t_loss)


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

    # 载入数据
    train_data = loadmat('earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']
    shape = args.shape
    model_name = 'MiniRocket_ClusterHead'
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

    start = time.time()
    mrf = MiniRocketFeatures(x_train.shape[1], x_train.shape[2]).to(device)
    mrf.fit(x_train)

    X_feat = get_minirocket_features(X, mrf, chunksize=512, to_np=True).reshape(X.shape[0], -1)
    print(X_feat.shape)
    X_test_feat = get_minirocket_features(x_test, mrf, chunksize=512, to_np=True).reshape(x_test.shape[0], -1)

    mrf84 = MiniRocketFeatures(x_train.shape[1], x_train.shape[2], num_features=84).to(device)
    mrf84.fit(x_train)
    X_feat_84 = get_minirocket_features(X, mrf84, chunksize=512, to_np=True).reshape(X.shape[0], -1)
    id_84 = mine_nearest_neighbors(X_feat_84, top_k=3)

    criterion = SCANLoss()

    import torch.optim as optim

    from torch.utils.data import TensorDataset, DataLoader

    train_i = torch.arange(len(y_train))
    train_set = TensorDataset(torch.Tensor(X_feat), torch.Tensor(X_feat[id_84[:, 1]]))  # create your dataset

    test_i = torch.arange(len(y_test))
    test_set = TensorDataset(torch.Tensor(X_test_feat), torch.Tensor(y_test))

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ClusterHead(feat=X_feat.shape[1])
    model.cuda()

    optimizer = optim.Adam(params=model.parameters(), lr=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
    best_acc = 0
    for epoch in range(50):
        train(train_loader, model, optimizer)
        preds, y_t = test(test_loader, model)
        print('Epoch %d' % epoch)
        c_acc, _ = cluster_acc(np.array(y_t), np.array(preds))
        if c_acc > best_acc:
            best_acc = c_acc
        sched.step()

    end = time.time()
    print("Best test clustering accuracy %.4f" % best_acc)
    print('data shape %s' % str(x_train.shape) + str(x_valid.shape) + str(x_test.shape))
    print("total time(feature init + Cluster Head training + evaluate cluster accuracy) takes %d seconds, %s" % (end - start, str(datetime.timedelta(seconds=end-start))))

    PATH = Path("./models/MR_feature.pt")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mrf.state_dict(), PATH)
    # PATH = Path('./models/MR_learner.pkl')
    # learn.export(PATH)
    print('model save path %s' % PATH)