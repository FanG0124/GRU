import torch
import nerual_network.my_mlp as mlp
from normalize.normalize import divide_batch_gru, divide_batch_mlp
from train_model.train_model import run_model


def method_mlp():
    """
    mlp方法
    :return:
    """

    normalize = "StandardScaler"
    loader_train, loader_test = divide_batch_mlp(normalize)
    if torch.cuda.is_available():
        my_mlp = mlp.my_mlp(loader_train, loader_test, "041", epoch=500,
                            input_size=7, hidden_size=6, output_size=1,
                            optimizer="Adam", loss_func="SmoothL1Loss", lr=1e-3)
        my_mlp.train_my_mlp()


def method_gru():SmoothL1Loss
    # MinMaxScaler  StandardScaler
    normalize = "StandardScaler"
    run_model(normalize)


if __name__ == '__main__':
    method_mlp()
