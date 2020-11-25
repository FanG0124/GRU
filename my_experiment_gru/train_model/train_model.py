import os
import torch
import numpy as np
import logging
import time
import gc
import progressbar
from tslearn.metrics import dtw, dtw_path
from normalize.normalize import divide_batch_gru
from loss.dilate_loss import dilate_loss
from nerual_network.my_gru import EncoderRNN, DecoderRNN, Net_GRU

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(net, loss_type, learning_rate, normalize, epochs=10000, gamma=0.001,
                print_every=5, eval_every=50, verbose=1, Lambda=1, alpha=0.5):
    """
    训练模型
    :param normalize:
    :param net:             网络
    :param loss_type:       loss函数
    :param learning_rate:   学习率
    :param epochs:          训练次数
    :param gamma:
    :param print_every:
    :param eval_every:
    :param verbose:
    :param Lambda:
    :param alpha:
    :return:
    """
    # 选择数据归一化的方法
    trainloader, testloader = divide_batch_gru(normalize)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    loss_0, loss_1, loss_2, loss_3 = [], [], [], []
    for epoch in range(epochs):
        start = time.time()
        logging.info("---------------------------------------")
        logging.info("- 第[{}/{}]次epoch".format(epoch, epochs))
        for i, data in enumerate(trainloader, 0):
            inputs, target = data  # input.shape -> torch.size([batch_size, input_size, new_axis])
            inputs = inputs.clone().detach().to(device)
            target = target.clone().detach().to(device)

            # forward + backward + optimize
            outputs = net(inputs)
            loss_mse, loss_shape, loss_temporal = torch.tensor(0), torch.tensor(0), torch.tensor(0)

            if loss_type == 'mse':
                loss_mse = criterion(target, outputs)
                loss = loss_mse

            if loss_type == 'dilate':
                loss, loss_shape, loss_temporal = dilate_loss(target, outputs, alpha, gamma, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end = time.time()
        logging.info("— 训练用时{}".format(end - start))
        if loss_type is 'mse':
            loss_0.append(loss.item())
            export_data('mse', loss_0, None, None)
            logging.info("- loss={}".format(str(loss.item())))
        else:
            loss_1.append(loss.item())
            loss_2.append(loss_shape.item())
            loss_3.append(loss_temporal.item())
            export_data('dilate', loss_1, loss_2, loss_3)
            logging.info("- loss={}, loss shape={}, loss temporal={} ".
                         format(str(loss.item()), str(loss_shape.item()), str(loss_temporal.item())))

        if verbose:
            if epoch % print_every == 0:
                eval_model(net, testloader, gamma, verbose=1)

        if loss.item() < 1e-5:
            break


def eval_model(net, loader, gamma, verbose=1):
    """
    评价训练集
    :param net:
    :param loader:
    :param gamma:
    :param verbose:
    :return:
    """
    start = time.time()
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        # get the inputs
        inputs, target = data
        inputs = inputs.clone().detach().to(device)
        target = target.clone().detach().to(device)
        batch_size, N_output = target.shape[0:2]
        outputs = net(inputs)

        # MSE
        loss_mse = criterion(target, outputs)
        loss_dtw, loss_tdi = 0, 0
        # DTW and TDI
        for k in range(batch_size):
            target_k_cpu = target[k, :, 0:1].view(-1).detach().cpu().numpy()
            output_k_cpu = outputs[k, :, 0:1].view(-1).detach().cpu().numpy()

            loss_dtw += dtw(target_k_cpu, output_k_cpu)
            path, sim = dtw_path(target_k_cpu, output_k_cpu)

            Dist = 0
            for i, j in path:
                Dist += (i - j) * (i - j)
            loss_tdi += Dist / (N_output * N_output)

        loss_dtw = loss_dtw / batch_size
        loss_tdi = loss_tdi / batch_size

        # print statistics
        losses_mse.append(loss_mse.item())
        losses_dtw.append(loss_dtw)
        losses_tdi.append(loss_tdi)

    end = time.time()
    logging.info("- 测试集用时{}".format(end-start))
    logging.info('- Eval mse=' + str(np.array(losses_mse).mean()) + '\n' +
                 '- dtw= ' + str(np.array(losses_dtw).mean()) + '\n' +
                 '- tdi= ' + str(np.array(losses_tdi).mean()))


def export_data(loss_type, loss, loss_shape, loss_temporal):
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    path_gru = os.path.join(os.path.dirname(__file__) + "/../dataset/result_gru.txt")
    path_mse = os.path.join(os.path.dirname(__file__) + "/../dataset/result_mse.txt")

    if loss_type is 'dilate':
        data = [list(item) for item in zip(loss, loss_shape, loss_temporal)]
        np.savetxt(path_gru, data, fmt='%f,%f,%f', delimiter=',')
    else:
        data = loss
        np.savetxt(path_mse, data, fmt='%f', delimiter=',')

    del loss, loss_shape, loss_temporal
    gc.collect()


# 将训练好的网络保存
def export_nn_params(net):
    state = {'model': net.state_dict()}
    torch.save(state, os.path.join(os.path.dirname(__file__) + "/../dataset/sd_001"))
    logging.info("网络保存完成")


def run_model(method_gru):
    logging.info("- encoder&decoder -")
    encoder = EncoderRNN(input_size=1, hidden_size=128, num_gru_layers=1, batch_size=2**7).to(device)
    decoder = DecoderRNN(input_size=1, hidden_size=128, num_gru_layers=1, fc_units=16, output_size=1).to(device)
    logging.info("- encoder&decoder done -")
    net_gru_dilate = Net_GRU(encoder, decoder, 1, device).to(device)
    logging.info("- train_model -")
    train_model(net_gru_dilate, loss_type='dilate', learning_rate=1e-3, normalize=method_gru,
                epochs=500, gamma=0.01, print_every=50, eval_every=50, verbose=1)

    encoder = EncoderRNN(input_size=1, hidden_size=128, num_gru_layers=1, batch_size=2**7).to(device)
    decoder = DecoderRNN(input_size=1, hidden_size=128, num_gru_layers=1, fc_units=16, output_size=1).to(device)
    net_gru_mse = Net_GRU(encoder, decoder, 1, device).to(device)
    logger.info("-----test_model-----")
    train_model(net_gru_mse, loss_type='mse', learning_rate=0.001,
                epochs=500, gamma=0.01, print_every=50, eval_every=50, verbose=1)
