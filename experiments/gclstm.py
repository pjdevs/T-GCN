from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GCLSTM
from torch_geometric.data import Data
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split , StaticGraphTemporalSignal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sequence_len = 12
prediction_len = 3
batch_size = 1
nb_epochs = 100

device = "cpu" if torch.cuda.is_available() else "cpu"

adjacency = pd.read_csv("data/sz_adj.csv", header=None).values
features = pd.read_csv("data/sz_speed.csv", header=0).values
max_feature_value = np.max(features)
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

print("Adjacency:", adjacency.shape)
print("Features:", features.shape)
print("Batch size:", batch_size)
print("Sequence length:", sequence_len)
print("Prediction length:", prediction_len)
print()
print("Device:", device)
print("Epochs:", nb_epochs)
print()

edge_index = adjacency_to_edge_index(adjacency).to(device)
features = features / max_feature_value

X,Y = generate_dataset(features, sequence_len, prediction_len, device=device)

edge_weight=np.ones((edge_index.shape[1],sequence_len))
print(edge_weight.shape)
dataset=StaticGraphTemporalSignal(edge_index=edge_index,edge_weight=edge_weight,features=X,targets=Y)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.5)

class RecurrentGCN(torch.nn.Module):
    def __init__(self):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GCLSTM(12, 3, 1)
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0

def train():
    model = RecurrentGCN()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in range(nb_epochs):
        cost = 0
        h, c = None, None
        for time, snapshot in tqdm(enumerate(train_dataset), desc=f"Epoch {epoch}"):
            y_hat, h, c = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.save(model, "GCLSTM.pt")

@torch.no_grad()
def test():
    def accuracy(pred, y, norm="fro"):
        return 1 - torch.linalg.norm(y - pred, norm) / torch.linalg.norm(y, norm)

    model = torch.load("GCLSTM.pt").to(device).eval()
    acc = []
    n = 0
    preds = []
    max_time = 1000
    road_index = 123

    for data in test_dataset:
        x = data.x
        y = data.y
        h, c = None, None
        if n % prediction_len == 0:
            pred, h, c = model(x, data.edge_index, data.edge_attr, h, c)
            preds.append(pred)
            acc.append(accuracy(pred, y))

        n += 1

        if n >= max_time:
            break

    print("Mean accuracy:", np.mean(acc))

    all_preds = torch.cat(preds, dim=1)

    road_preds = all_preds[road_index,].numpy()
    times = list(range(len(road_preds)))

    plt.plot(times, [test_dataset.features[i][road_index,0] for i in range(len(road_preds))], c="blue")
    plt.plot(times, road_preds, c="red")
    plt.ylim(-1, 1)
    plt.show()


#train()
test()