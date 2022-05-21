import torch
import torch.utils.data
import numpy as np


def generate_torch_dataset(data, input_dimension, output_dimension, normalize=False, device="cpu"):
    if normalize:
        data /= np.max(data)

    train_X = []
    train_Y = []

    for i in range(len(data)-input_dimension-output_dimension):
        train_X.append(data[i:i+input_dimension])
        train_Y.append(
            data[i+input_dimension:i+input_dimension+output_dimension])

    return torch.utils.data.TensorDataset(torch.FloatTensor(np.array(train_X)).to(device), torch.FloatTensor(np.array(train_Y)).to(device))


def adjacency_to_edge_index(adjacency: np.ndarray):
    return torch.LongTensor([
        [i, j]
            for i in range(adjacency.shape[0])
            for j in range(adjacency.shape[1])
            if adjacency[i][j]
    ]).transpose(0, 1)