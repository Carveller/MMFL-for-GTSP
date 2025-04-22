
import torch
import numpy as np


def get_random_problems(batch_size, problem_size):
    # torch.manual_seed(5)
    nodexy = torch.rand(size=(batch_size, problem_size+1, 2), device=torch.device('cuda'), requires_grad=False)

    if problem_size >= 20:
        cluster_num = int(problem_size/5)
    else:
        raise Exception('No such setting')

    cluster_idx = torch.randint(low=1, high=cluster_num+1, size=(batch_size, problem_size), device=torch.device('cuda'), requires_grad=False)
    cluster_idx = torch.cat((torch.zeros((batch_size, 1), device=torch.device('cuda'), requires_grad=False), cluster_idx), 1)

    return nodexy, cluster_idx


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data