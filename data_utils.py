import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ShuffleSplit
from fastcore.foundation import L


def normalization(array):
    # max_cols = array.max()
    # min_cols = array.min()
    normalizer_data = preprocessing.Normalizer().fit_transform(array)
    # return (array - min_cols) / (max_cols - min_cols)
    return normalizer_data


def data_clean_and_norm(data, target, args):
    x = data[:, 300:1836]  # [11086, 1536]
    yy = target[:, :]
    y = []
    # 0: 爆炸 blast
    # 1: noise 噪声
    # 5: nature 自然地震
    for i in range(len(x)):
        j = len(yy[i]) - 1
        while j >= 0:
            if yy[i][j] == 1:
                # print('在{} {}的处找到key值'.format(i, j))
                y.append(j)
                break
            j -= 1
        else:
            print('没找到')

    y = np.array(y)
    if args.dataset == 'eq2' or args.dataset == 'ne':
        # eq2 -> only consider  noise 1 and nature earth 5
        ys = y
        # remove 0 blast
        index = y != 0
        x = x[index]
        y = ys[index]
        # treat nature 5 as category 0
        y[y == 5] = 0
    elif args.dataset == 'nb':
        # eq2 -> only consider  noise 1 and blast 0
        ys = y
        # remove 0 blast
        index = y != 5
        x = x[index]
        y = ys[index]
    elif args.dataset == 'eb':
        # eq2 -> only consider  blast 0 and nature earth 5
        ys = y
        # remove 0 blast
        index = y != 1
        x = x[index]
        y = ys[index]
        # treat nature 5 as category 1
        y[y == 5] = 1
    else:
        y[y == 5] = 2

    norm_x = normalization(x)

    tensor_x = torch.Tensor(norm_x)  # transform to torch tensor
    tensor_y = torch.Tensor(y)
    tensor_i = torch.arange(len(y))
    my_dataset = TensorDataset(tensor_x, tensor_y, tensor_i)  # create your dataset

    x_train, x_test, y_train, y_test = train_test_split(norm_x, y, test_size=0.1, random_state=args.seed)
    train_i = torch.arange(len(y_train))
    train_set = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train), train_i)  # create your dataset
    test_i = torch.arange(len(y_test))
    test_set = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test), test_i)  # create your dataset
    return my_dataset, train_set, test_set


def get_data(data, target, dataset='eq', seed=None, shape=None):
    if seed is None:
        seed = np.random.randint(100000)
    k = 2000
    if shape is None:
        shape = (-1, 3, k)
    x = data  # [:, 300:300 + k]  # [220??, 4096]  4096=64*64
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

    norm_x = normalization(x).reshape(shape).astype('float32')
    sss = ShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    sss.get_n_splits(norm_x, y)
    splits_test = next(sss.split(norm_x, y))

    train_index, valid_index = splits_test
    x_train, x_test = norm_x[train_index], norm_x[valid_index]
    y_train, y_test = y[train_index], y[valid_index]
    # x_train, x_test, y_train, y_test = train_test_split(norm_x, y, test_size=0.3, random_state=seed)

    # training and validation sets
    sss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    sss.get_n_splits(x_train, y_train)
    splits = next(sss.split(x_train, y_train))
    train_index, valid_index = splits
    x_train, x_valid = x_train[train_index], x_train[valid_index]
    y_train, y_valid = y_train[train_index], y_train[valid_index]

    splits2 = (L(list(splits[0])), L(list(splits[1])))
    return x_train, x_valid, x_test, y_train, y_valid, y_test, splits2, splits_test


def get_data2(args):
    train_data = loadmat('earthb.mat')
    all_data = train_data['images']
    all_target = train_data['labels']

    dataset, train_set, test_set = data_clean_and_norm(all_data, all_target, args)
    return dataset, train_set, test_set


def get_unlabeled_data(args):
    train_data = loadmat('strong_data.mat')
    all_data = train_data['out']
    norm_x = normalization(all_data)
    tensor_x = torch.Tensor(norm_x)  # transform to torch tensor
    tensor_i = torch.arange(len(norm_x))
    my_dataset = TensorDataset(tensor_x, tensor_i)  # create your dataset
    return my_dataset


def random_uniform_data(data_loader, args):
    batch_size = args.batch_size
    dataset = data_loader.dataset
    data, target, index = dataset.tensors
    n_classes = args.num_classes
    u_data = []
    u_target = []

    from torch.utils.data import TensorDataset, DataLoader
    for i in range(n_classes):
        idx = np.random.permutation(target == i)
        u_data.append(data[idx][:100])
        u_target.append(target[idx][:100])

    u_data = torch.cat(u_data)
    u_target = torch.cat(u_target).long()
    tensor_i = torch.arange(n_classes * 100)
    vis_dataset = TensorDataset(u_data, u_target, tensor_i)  # create your dataset
    vis_loader = torch.utils.data.DataLoader(vis_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return vis_loader
