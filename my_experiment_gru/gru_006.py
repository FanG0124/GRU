import torch
import nerual_network.my_mlp as mlp
import nerual_network.my_gru as gru
from normalize.normalize import divide_batch_gru, divide_batch_mlp


def method_mlp():
    """ mlp方法
    :return:
    """

    normalize = "StandardScaler"
    loader_train, loader_test = divide_batch_mlp(normalize)
    if torch.cuda.is_available():
        my_mlp = mlp.my_mlp(loader_train, loader_test, n_result="041", epoch=500,
                            input_size=7, hidden_size=6, output_size=1,
                            optimizer="Adam", loss_func="MSE", lr=1e-4)
        my_mlp.train_my_mlp()


def method_gru():
    """ GRU方法
    :return:
    """
    # MinMaxScaler  StandardScaler
    normalize = "MinMaxScaler"
    loader_train, loader_test = divide_batch_gru(normalize)
    if torch.cuda.is_available():
        my_gru = gru.my_gru(batch_size=2**7, train_set=loader_train, test_set=loader_test, n_result="006",
                            epochs=500, input_size=7, hidden_size=128, output_size=1, num_gru_layers=1,
                            optimizer="Adam", loss_func='dilate', lr=1e-5)
        my_gru.train_model()


if __name__ == '__main__':
    method_gru()
