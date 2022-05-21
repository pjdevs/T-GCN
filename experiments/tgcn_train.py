from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM
import torch_geometric_temporal.nn as gtnn
from torch_geometric.data import Data
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split , StaticGraphTemporalSignal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(data, input_dimension, output_dimension, normalize=False, device="cpu"):
    if normalize:
        data /= np.max(data)

    train_X = []
    train_Y = []

    for i in range(len(data)-input_dimension-output_dimension):
        train_X.append(np.transpose(data[i:i+input_dimension]))
        train_Y.append(np.transpose(data[i+input_dimension:i+input_dimension+output_dimension]))
    return train_X,train_Y
def adjacency_to_edge_index(adjacency: np.ndarray):
    return torch.LongTensor([
        [i, j]
            for i in range(adjacency.shape[0])
            for j in range(adjacency.shape[1])
            if adjacency[i][j]
    ]).transpose(0, 1)

sequence_len = 12
prediction_len = 3
batch_size = 1
nb_epochs = 20

device = "cpu" if torch.cuda.is_available() else "cpu"

adjacency = pd.read_csv("data/sz_adj.csv", header=None).values
features = pd.read_csv("data/sz_speed.csv", header=0).values
max_feature_value = np.max(features)
edge_index = adjacency_to_edge_index(adjacency).to(device)
train_len = int(0.8*features.shape[0])
features = features / max_feature_value
X,Y = generate_dataset(features, sequence_len, prediction_len, device=device)
edge_weight=np.ones((edge_index.shape[1]))
print(edge_weight.shape)
print(edge_index.shape)
dataset=StaticGraphTemporalSignal(edge_index=edge_index,edge_weight=edge_weight,features=X,targets=Y)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.6)


def train():
    model = gtnn.TGCN(sequence_len, prediction_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()

    model.train()

    for epoch in range(nb_epochs):
        for data in tqdm(train_dataset, desc=f"Epoch {epoch}", total=len(train_dataset.features)):
            # Run model on input data
            H = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))

            # Calculate the loss
            loss = loss_function(H, data.y.to(device))

            # Perform backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("loss =", loss.item())

    torch.save(model, "model.pt")

def test():
    def accuracy(pred, y, norm="fro"):
        """
        :param pred: predictions
        :param y: ground truth
        :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
        """
        return 1 - torch.linalg.norm(y - pred, norm) / torch.linalg.norm(y, norm)

    model = torch.load("TGCN64.pt")
    acc = 0
    n = 0
    truth = []
    preds = []
    times=[]
    for data in test_dataset:
        x = data.x
        y = data.y

        pred = model(x, data.edge_index, data.edge_attr)

        truth.append(y[5,0].item())
        preds.append(pred[5,0].item())
        acc += accuracy(pred[5,], y[5,], 2)
        n += 1
        times.append(n)

        if n > 100:
            break


    print("Mean accuracy:", acc / n)
    plt.plot(times, preds, 'r', times, truth, 'b')
    #all_truth = torch.cat(truth, dim=1)
    #all_preds = torch.cat(preds, dim=1)
    plt.xlabel('time')
    plt.ylabel('vitesse')
    plt.show()

# train()
test()
