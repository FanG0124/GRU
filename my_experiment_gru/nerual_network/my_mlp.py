import torch
import os
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import datetime
from sklearn.metrics import explained_variance_score  # 可释方差也叫解释方差（explained_variance_score）
from sklearn.metrics import mean_absolute_error  # 平均绝对误差（mean_absolute_error）
from sklearn.metrics import mean_squared_error  # 均方误差（mean_squared_error）
from sklearn.metrics import median_absolute_error  # 中值绝对误差（median_absolute_error）
from sklearn.metrics import r2_score  # R方值，确定系数（r2_score）


class Net_mlp(torch.nn.Module):
    """
    定义mlp
    """
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net_mlp, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


class my_mlp(object):
    def __init__(self, train_set, test_set, n_result, epoch,
                 input_size, hidden_size, output_size, optimizer, loss_func, lr):
        self.train_set = train_set  # 训练集
        self.test_set = test_set  # 测试集
        self.n_result = n_result
        self.epoch = epoch  #
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr = lr

        self.net = self.set_mlp_index()  # 初始化网络
        self.opt = self.set_mlp_optim()  # 初始化优化函数
        self.loss_function = self.set_mlp_lossfunc()  # 初始化loss函数
        self.__converge_train = []
        self.__converge_test = []

    # 查看是否使用GPU
    def set_device(self):
        ngpu = 1
        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
        return device

    # 设置mlp参数
    def set_mlp_index(self):
        return Net_mlp(self.input_size, self.hidden_size, self.output_size).to(self.set_device())

    # 设置优化器
    def set_mlp_optim(self):
        if self.optimizer == "Adam":
            return torch.optim.Adam(self.net.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            return torch.optim.SGD(self.net.parameters(), lr=self.lr)
        elif self.optimizer == "rmsprop":
            return torch.optim.RMSprop(self.net.parameters(), lr=self.lr)

    # 设置loss函数
    def set_mlp_lossfunc(self):
        if self.loss_func == "MSE":
            return torch.nn.MSELoss()
        elif self.loss_func == "SmoothL1Loss":
            return torch.nn.SmoothL1Loss()
        elif self.loss_func == "RMSE":
            return mean_squared_error

    # 训练训练集
    def train_train_set(self):
        for step_train, (x_train, y_train) in enumerate(self.train_set):
            # 将数据放入cuda
            x = Variable(x_train).to(self.set_device())
            y = Variable(y_train).to(self.set_device())

            # net = self.set_mlp_index()
            prediction_train = self.net(x)

            # lossfunc = self.set_mlp_lossfunc()
            loss_train = self.loss_function(prediction_train, y)

            self.opt.zero_grad()
            loss_train.backward()
            self.opt.step()
        print("    训练集loss:{}".format(loss_train.item()))
        self.__converge_train.append(loss_train)

    # 测试测试集
    def test_test_set(self):
        for step_test, (x_test, y_test) in enumerate(self.test_set):
            # 将数据放入cuda
            x = Variable(x_test).to(self.set_device())
            y = Variable(y_test).to(self.set_device())
            # net = self.set_mlp_index()
            prediction_test = self.net(x)
            # lossfunc = self.set_mlp_lossfunc()
            loss_test = self.loss_function(prediction_test, y)
        print("    测试集loss:{}".format(loss_test.item()))
        self.__converge_test.append(loss_test)

    # 导出数据
    def export_data(self):
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        data = [list(item) for item in zip(self.__converge_train, self.__converge_test)]
        path = os.path.join(os.path.dirname(__file__) + "/../dataset/result_" + str(self.n_result) + ".txt")
        np.savetxt(path, data, fmt='%f,%f', delimiter=',')

    # 将训练好的网络保存
    def export_nn_params(self):
        state = {'model': self.net.state_dict(),
                 'optimizer': self.opt.state_dict(),
                 'epoch': self.epoch}
        torch.save(state, os.path.join(os.path.dirname(__file__) + "/../dataset/sd_" + str(self.n_result)))
        print("网络保存完成")

    # 训练mlp
    def train_my_mlp(self):
        print("optimizer=" + self.optimizer + ", loss_func=" + str(self.loss_function) + ", lr=" + str(self.lr) +
              ",epoch=" + str(self.epoch) + ",input_size=" + str(self.input_size) +
              ",hidden_size=" + str(self.hidden_size) + ",output_size=" + str(self.output_size))
        for t in range(self.epoch):
            print("-------------------------------------------")
            start_time = datetime.datetime.now()
            self.train_train_set()
            if self.__converge_train[-1] < 1e-5:
                print('- Epoch [{}/{}], Loss: {:.5f}'.format(t+1, self.epoch, self.__converge_train[-1]))
                print("- The loss value is reached")
                break
            elif (t + 1) % 10 == 0:
                self.test_test_set()
                print('- Epoch: [{}/{}], Loss:{:.5f}'.format(t+1, self.epoch, self.__converge_test[-1]))
            end_time = datetime.datetime.now()
            print('- Epoch: [{}/{}],用时{}'.format(t+1, self.epoch, end_time - start_time))
            self.export_data()
        self.export_nn_params()


# 勘察误差
def check_error(label, pre_y):
    label = label.cpu().detach().numpy()
    pre_y = pre_y.cpu().detach().numpy()
    print("可释方差(explained_variance_score)为: {}"
          .format(mean_absolute_error(label, pre_y)))
    print("平均绝对误差(mean_absolute_error)为: {}".format(mean_absolute_error(label, pre_y)))
    print("均方误差（mean_squared_error）为: {}".format(mean_squared_error(label, pre_y)))
    print("中值绝对误差（median_absolute_error）为: {}".format(median_absolute_error(label, pre_y)))
    print("R方值，确定系数（r2_score）为: {}".format(r2_score(label, pre_y)))


