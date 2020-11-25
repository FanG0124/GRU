import torch
import numpy as np


a = torch.arange(0, 256, 2).reshape(128, 1, 1)
b = torch.arange(128, 256, 1).reshape(128, 1, 1)
batch_size, n_output = a.shape[0:2]
print("batch_size:{}, n_output:{}".format(batch_size, n_output))
D = torch.zeros((batch_size, n_output, n_output))  # D.shape -> torch.size([batch_size, 7, 7])
for k in range(batch_size):
    x = a[k, :, :].view(-1, 1)
    y = b[k, :, :].view(-1, 1)

    x_norm = (x ** 2).sum(1).view(-1, 1)
    # print("x_norm:{}".format(x_norm))
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        # print("y_norm:{}".format(y_norm))
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    # print("dist:{}".format(dist))
    D[k:k + 1, :, :] = torch.clamp(dist, 0.0, float('inf'))


print("D:{}, D.shape:{}".format(D, D.shape))


def compute_softdtw(D, gamma):
    # D.shape -> torch.size([batch_size, 1, 1])
    N = D.shape[0]  # N -> k = 1
    M = D.shape[1]  # M -> input_size=1
    R = np.zeros((N + 2, M + 2)) + 1e8
    R[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            r0 = -R[i - 1, j - 1] / gamma
            r1 = -R[i - 1, j] / gamma
            r2 = -R[i, j - 1] / gamma
            r_max = max(max(r0, r1), r2)
            r_sum = np.exp(r0 - r_max) + np.exp(r1 - r_max) + np.exp(r2 - r_max)
            soft_min = - gamma * (np.log(r_sum) + r_max)
            R[i, j] = D[i - 1, j - 1] + soft_min
            print("D[i - 1, j - 1]={}".format(D[i - 1, j - 1]))
    return R


total_loss = 0
R = torch.zeros((batch_size, 1+2, 1+2))
for k in range(batch_size):
    Rk = torch.FloatTensor(compute_softdtw(D[k, :, :], 1e-4))
    R[k:k+1, :, :] = Rk
    # print(Rk)
    # print(Rk[-2, -2])
    total_loss = total_loss + Rk[-2, -2]

print(total_loss / batch_size)


