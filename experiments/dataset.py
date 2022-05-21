import torch
import torch.nn as nn
import torch_geometric_temporal.nn as gtnn
from torch_geometric_temporal.dataset import PemsBayDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class TGCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()

        self.tgcn = gtnn.TGCN(in_channels, hidden_channels)
        self.reg = nn.Linear(hidden_channels, out_channels)

    def forward(self, inputs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        h = self.tgcn(inputs, edge_index, edge_attr)
        h = self.reg(h)
        
        return h

sequence_len = 12
hidden = 64
prediction_len = 3
nb_epochs = 20
train_ratio = 0.1

device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

loader = PemsBayDatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=sequence_len, num_timesteps_out=prediction_len)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=train_ratio)

def train():
    model = TGCN(sequence_len, hidden, prediction_len).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()

    model.train()

    loss = torch.Tensor([0.0])

    for epoch in range(nb_epochs):
        for data in tqdm(train_dataset, desc=f"Epoch {epoch}", total=len(train_dataset.features), postfix=f"loss={loss.item()}"):
            # Run model on input data
            H = model(data.x[:,0,:].to(device), data.edge_index.to(device), data.edge_attr.to(device))

            # Calculate the loss
            loss = loss_function(H, data.y[:,0,:].to(device))

            # Perform backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model, "model.pt")

@torch.no_grad()
def test():
    def accuracy(pred, y, norm="fro"):
        return 1 - torch.linalg.norm(y - pred, norm) / torch.linalg.norm(y, norm)

    model = torch.load("TGCN64.pt").to(device).eval()
    acc = []
    n = 0
    preds = []
    max_time = 1000
    road_index = 0

    for data in test_dataset:
        x = data.x[:,0,:]
        y = data.y[:,0,:]

        if n % prediction_len == 0:
            pred = model(x, data.edge_index, data.edge_attr)
            preds.append(pred)
            acc.append(accuracy(pred, y))

        n += 1

        if n >= max_time:
            break

    print("Mean accuracy:", np.mean(acc))

    all_preds = torch.cat(preds, dim=1)

    road_preds = all_preds[road_index,].numpy()
    times = list(range(len(road_preds)))

    plt.plot(times, [test_dataset.features[i][road_index,0,0] for i in range(len(road_preds))], c="blue")
    plt.plot(times, road_preds, c="red")
    plt.show()

# train()
test()
