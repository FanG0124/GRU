import torch
import logging
import time
import gc
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tslearn.metrics import dtw, dtw_path
from loss.dilate_loss import dilate_loss


logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Log等级总开关


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_gru_layers, batch_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_gru_layers = num_gru_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_gru_layers,
                          batch_first=True)

    def forward(self, input, hidden):  # input [batch_size, length T, dimensionality d]
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def init_hidden(self, device):
        return torch.zeros(self.num_gru_layers, self.batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_gru_layers, fc_units, output_size):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_gru_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, fc_units)
        self.out = nn.Linear(fc_units, output_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = F.relu(self.fc(output))
        output = self.out(output)
        return output, hidden


class Net_GRU(nn.Module):
    def __init__(self, encoder, decoder, target_length, device):
        super(Net_GRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_length = target_length
        self.device = device

    def forward(self, x):
        input_length = x.shape[1]
        encoder_hidden = self.encoder.init_hidden(self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[:, ei:ei+1, :], encoder_hidden)

        decoder_input = x[:, -1, :].unsqueeze(1)  # first decoder input= last element of input sequence
        decoder_hidden = encoder_hidden

        outputs = torch.zeros([x.shape[0], self.target_length, x.shape[2]]).to(self.device)
        for di in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output
            outputs[:, di:di + 1, :] = decoder_output
        return outputs


class my_gru(object):
    def __init__(self, batch_size, train_set, test_set, n_result, epochs,
                 input_size, hidden_size, output_size, num_gru_layers, optimizer, loss_func, lr):
        super(my_gru, self)
        self.batch_size = batch_size
        self.train_set = train_set  # 训练集
        self.test_set = test_set  # 测试集
        self.n_result = n_result
        self.epochs = epochs  #
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_gru_layers = num_gru_layers
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr = lr
        self.alpha = 0.5
        self.gamma = 1e-2

        self.loss_0, self.loss_1, self.loss_2, self.loss_3 = [], [], [], []
        self.losses_mse, self.losses_dtw, self.losses_tdi = [], [], []

        self.net = self.set_gru_index()  # 初始化网络
        self.opt = self.set_gru_optim()  # 初始化优化函数

    # 查看是否使用GPU
    def set_device(self):
        n_gpu = 1
        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpu > 0) else "cpu")
        return device

    # 设置gru参数
    def set_gru_index(self):
        logging.info("- encoder&decoder")
        encoder = EncoderRNN(input_size=1, hidden_size=self.hidden_size,
                             num_gru_layers=self.num_gru_layers,
                             batch_size=self.batch_size).to(self.set_device())

        decoder = DecoderRNN(input_size=1, hidden_size=self.hidden_size,
                             num_gru_layers=self.num_gru_layers, fc_units=16,
                             output_size=self.output_size).to(self.set_device())
        return Net_GRU(encoder, decoder, self.output_size, self.set_device()).to(self.set_device())

    # 设置优化器
    def set_gru_optim(self):
        if self.optimizer == "Adam":
            return torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            return torch.optim.SGD(self.net.parameters(), lr=self.lr)
        elif self.optimizer == "rmsprop":
            return torch.optim.RMSprop(self.net.parameters(), lr=self.lr)

    # 训练训练集
    def train_train_set(self):
        start = time.time()  # 记录当前epoch起始时间
        for i, data in enumerate(self.train_set):
            inputs, target = data  # input.shape -> torch.size([batch_size, input_size, new_axis])
            inputs = inputs.clone().detach().to(self.set_device())
            target = target.clone().detach().to(self.set_device())

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss_mse, loss_shape, loss_temporal = torch.tensor(0), torch.tensor(0), torch.tensor(0)

            if self.loss_func == 'mse':
                criterion = torch.nn.MSELoss()
                loss_mse = criterion(target, outputs)
                loss = loss_mse

            if self.loss_func == 'dilate':
                loss, loss_shape, loss_temporal = dilate_loss(target, outputs, self.alpha, self.gamma, self.set_device())

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        end = time.time()
        logging.info("— 训练用时{}".format(end - start))
        if self.loss_func is 'mse':
            self.loss_0.append(loss.item())
            self.export_data('mse', self.loss_0, None, None)
            logging.info("- loss={}".format(str(loss.item())))
        else:
            self.loss_1.append(loss.item())
            self.loss_2.append(loss_shape.item())
            self.loss_3.append(loss_temporal.item())
            self.export_data('dilate', self.loss_1, self.loss_2, self.loss_3)
            logging.info("- loss={}, loss shape={}, loss temporal={} ".
                         format(str(loss.item()), str(loss_shape.item()), str(loss_temporal.item())))

    # 测试测试集
    def test_test_set(self):
        criterion = torch.nn.MSELoss()

        for i, data in enumerate(self.test_set, 0):
            # loss_mse, loss_dtw, loss_tdi = torch.tensor(0), torch.tensor(0), torch.tensor(0)
            inputs, target = data
            inputs = inputs.clone().detach().to(self.set_device())
            target = target.clone().detach().to(self.set_device())
            batch_size, n_output = target.shape[0:2]
            outputs = self.net(inputs)

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
                loss_tdi += Dist / (n_output * n_output)

            loss_dtw = loss_dtw / batch_size
            loss_tdi = loss_tdi / batch_size

            # print statistics
            self.losses_mse.append(loss_mse.item())
            self.losses_dtw.append(loss_dtw)
            self.losses_tdi.append(loss_tdi)

        logging.info(' Eval mse= ', np.array(self.losses_mse).mean(),
                     ' dtw= ', np.array(self.losses_dtw).mean(),
                     ' tdi= ', np.array(self.losses_tdi).mean())

    # 训练模型
    def train_model(self):
        logging.info("---------------------------------------")
        logging.info("- optimizer={}, loss function={}, lr={}".format(self.optimizer, self.loss_func, self.lr))
        for epoch in range(self.epochs):
            logging.info("---------------------------------------")
            logging.info("- 第[{}/{}]次epoch".format(epoch, self.epochs))
            self.train_train_set()  # 训练训练集
            self.test_test_set()  # 测试测试集
            if self.loss_func is 'mse':  # 若收敛，停止训练
                if self.loss_0[-1] < 1e-5:
                    self.export_nn_params()
                    break
            elif self.loss_func is 'dilate':
                if self.loss_1[-1] < 1e-5:
                    self.export_nn_params()
                    break

    # 将训练好的网络保存
    def export_nn_params(self):
        state = {'model': self.net.state_dict()}
        torch.save(state, os.path.join(os.path.dirname(__file__) + "/../dataset/gru_params_" + str(self.n_result)))
        logging.info("网络保存完成")

    # 导出数据
    def export_data(self, loss_type, loss, loss_shape, loss_temporal):
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        path_gru = os.path.join(os.path.dirname(__file__) + "/../dataset/result_gru.txt")
        path_mse = os.path.join(os.path.dirname(__file__) + "/../dataset/result_mse.txt")
        path_test = os.path.join(os.path.dirname(__file__) + "/../dataset/result_test.txt")

        if loss_type is 'dilate':
            data = [list(item) for item in zip(loss, loss_shape, loss_temporal)]
            np.savetxt(path_gru, data, fmt='%f,%f,%f', delimiter=',')
        elif loss_type is 'mse':
            data = loss
            np.savetxt(path_mse, data, fmt='%f', delimiter=',')
        else:
            data = [list(item) for item in zip(self.losses_mse, self.losses_dtw, self.losses_tdi)]
            np.savetxt(path_test, data, fmt='%%f,%f,%f', delimiter=',')

        del loss, loss_shape, loss_temporal
        gc.collect()