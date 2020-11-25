import torch
from . import soft_dtw
from . import path_soft_dtw 


def dilate_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	# input.shape -> torch.size([batch_size, input_size, new_axis])
	batch_size, n_output = outputs.shape[0:2]
	loss_shape = 0
	soft_dtw_batch = soft_dtw.SoftDTWBatch.apply
	dist = torch.zeros((batch_size, n_output, n_output)).to(device)  # D.shape -> torch.size([batch_size, 1, 1])
	for k in range(batch_size):
		dist_k = soft_dtw.pairwise_distances(targets[k, :, :].view(-1, 1), outputs[k, :, :].view(-1, 1))
		dist[k:k+1, :, :] = dist_k
	loss_shape = soft_dtw_batch(dist, gamma)
	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(dist, gamma)
	omega = soft_dtw.pairwise_distances(torch.arange(1, n_output+1).view(n_output, 1)).to(device)  # 修改过
	loss_temporal = torch.sum(path*omega) / (n_output*n_output)

	loss = alpha*loss_shape + (1-alpha)*loss_temporal
	return loss, loss_shape, loss_temporal
