import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


def result_mlp():
    data = pd.read_table('./dataset/result.txt', header=None, engine='python', sep=',')
    epoch = len(data)
    data_train = data.loc[:, 0]
    data_test = data.loc[:, 1]

    # MinMaxScaler  StandardScaler
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(epoch), data_train, 'r', label='train', linewidth=1)  # np.arange()返回等差数组
    ax.plot(np.arange(epoch), data_test, 'b', label='test', linewidth=1)

    plt.text(0.1, 0.8,
             s='normlize=StandardScaler,\
optimizer=Adam, loss_func=MSELoss(), lr=1e-05\nepoch=5000,input_size=7,hidden_size=6,output_size=1'
             , transform=ax.transAxes)
    # 设置图例显示在图的上方
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    plt.grid()
    plt.show()


def result_gru():
    data = pd.read_table('./dataset/result.txt', header=None, engine='python', sep=',')
    epoch = len(data)
    data_train = data.loc[:, 0]
    data_test = data.loc[:, 1]
    data_temp = data.loc[:, 2]
    # MinMaxScaler  StandardScaler
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(epoch), data_train, 'r', label='loss', linewidth=0.5)  # np.arange()返回等差数组
    ax.plot(np.arange(epoch), data_test, 'b', label='loss_shape', linewidth=0.5)
    ax.plot(np.arange(epoch), data_temp, 'y', label='loss_temporal', linewidth=0.5)
    plt.text(0.3, 0.8,
             s='normlize=mine,optimizer=Adam, loss_func=MSELoss(),\n lr=0.00001,epoch=500,input_size=4,hidden_size=3,output_size=1'
             , transform=ax.transAxes)
    # 设置图例显示在图的上方
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    result_mlp()
