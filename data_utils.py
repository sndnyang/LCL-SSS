import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
from fastcore.foundation import L


def normalization(array):
    # max_cols = array.max()
    # min_cols = array.min()
    normalizer_data = preprocessing.Normalizer().fit_transform(array)
    # return (array - min_cols) / (max_cols - min_cols)
    return normalizer_data


def get_data(data, target, dataset='eq', seed=None, select=None, shape=None, size=0.3):
    if seed is None:
        seed = np.random.randint(100000)
    k = 2000
    if shape is None:
        shape = (-1, 3, k)
    x = data  # [:, 300:300 + k]  # [220??, 4096]  4096=64*64
    if select is not None:
        x = data[:, select[0]:select[1]]
    # 0: 爆炸 blast
    # 1: 地震 earthquake
    # 5: noise 噪声 -> 2
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
    sss = ShuffleSplit(n_splits=1, test_size=size, random_state=seed)
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


def get_data_loader(data, target, args, shape=None):
    import torch
    from torch.utils.data import TensorDataset

    x_train, x_valid, x_test, y_train, y_valid, y_test, splits2, splits_test = get_data(data, target, dataset=args.dataset, seed=args.seed, select=None, shape=shape)
    # set contains training and validation
    X = np.concatenate([x_train, x_valid])
    y = np.concatenate([y_train, y_valid])

    tensor_x = torch.Tensor(X)  # transform to torch tensor
    tensor_y = torch.Tensor(y)
    tensor_i = torch.arange(len(y))
    my_dataset = TensorDataset(tensor_x, tensor_y, tensor_i)  # create your dataset

    train_i = torch.arange(len(y_train))
    train_set = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train), train_i)  # create your dataset
    test_i = torch.arange(len(y_test))
    test_set = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test), test_i)  # create your dataset
    return my_dataset, train_set, test_set


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
