"""
normalize
将数据归一化
"""
import numpy as np
import torch
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch.utils.data as Data


def load_data():
    # 导入空车转移概率数据表
    path = os.path.join(os.path.dirname(__file__) + '/../dataset/final_data')
    data = pd.read_table(path, header=None, engine='python', sep=',')
    return data


def data_normalization(method_type):
    """
    选择数值归一化的方法
    :param method_type: 归一化名称
    :return: feature, label
    """
    data = load_data()
    feature = data.loc[:, 0:6:1].values
    label = data.loc[:, 7:8:1].values
    if method_type is "MinMaxScaler":  # 归一化
        feature = MinMaxScaler().fit_transform(feature)
        label = MinMaxScaler().fit_transform(label)
    elif method_type is "StandardScaler":  # 标准化
        feature = StandardScaler().fit_transform(feature)
        label = MinMaxScaler().fit_transform(label)
    elif method_type is "z-score":
        feature = preprocessing.scale(feature)
        label = preprocessing.scale(label)
    elif method_type is "atan":
        feature = np.arctan(feature)
        label = np.arctan(label)
    elif method_type is "none":
        pass
    print("normlize={}".format(method_type))
    return feature, label


def divide_batch_mlp(method_type):
    x, y = data_normalization(method_type)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    # y = torch.unsqueeze(y, dim=1)
    kflod = KFold(n_splits=10)
    for train_index, test_index in kflod.split(x):
        # print("train_index: {} , test_index: {} ".format(train_index, test_index))
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    dataset_train = Data.TensorDataset(x_train, y_train)
    dataset_test = Data.TensorDataset(x_test, y_test)

    loader_train = Data.DataLoader(
        dataset=dataset_train,
        batch_size=2**7,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    loader_test = Data.DataLoader(
        dataset=dataset_test,
        batch_size=2**7,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    return loader_train, loader_test


class gru_dataset(torch.utils.data.Dataset):
    def __init__(self, input, target):
        super(gru_dataset, self).__init__()
        self.input = input
        self.target = target

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx, :, np.newaxis], self.target[idx, :, np.newaxis]


def divide_batch_gru(method_type):
    x, y = data_normalization(method_type)
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    # y = torch.unsqueeze(y, dim=1)

    kflod = KFold(n_splits=10)
    for train_index, test_index in kflod.split(x):
        # print("train_index: {} , test_index: {} ".format(train_index, test_index))
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # return x_train, y_train, x_test, y_test
    dataset_train = gru_dataset(x_train, y_train)
    dataset_test = gru_dataset(x_test, y_test)

    loader_train = Data.DataLoader(
        dataset=dataset_train,
        batch_size=2**7,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    loader_test = Data.DataLoader(
        dataset=dataset_test,
        batch_size=2**7,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    return loader_train, loader_test


loader_train, loader_test = divide_batch_gru("StandardScaler")
for i, data in enumerate(loader_train):
    if i is 0:
        x, y= data
        print(x.shape, y.shape)
