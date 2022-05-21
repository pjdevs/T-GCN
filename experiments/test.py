import torch
import torch.nn
import torch.utils
import torch.utils.data
import torch_geometric_temporal.nn as gtnn
from utils.data import generate_torch_dataset, adjacency_to_edge_index
import pandas as pd
import numpy as np

sequence_len = 12
prediction_len = 3
batch_size = 4
nb_epochs = 10

device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

adjacency = pd.read_csv("data/sz_adj.csv", header=None).values
features = pd.read_csv("data/sz_speed.csv", header=0).values
max_feature_value = np.max(features)

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

train_len = int(0.8*features.shape[0])
train_features = features[:train_len,] / max_feature_value
# test_features = features[train_len:,] / max_feature_value

torch_train_dataset = generate_torch_dataset(train_features, sequence_len, prediction_len, device=device)
train_data_loader = torch.utils.data.DataLoader(torch_train_dataset, batch_size=batch_size)

# torch_test_dataset = generate_torch_dataset(test_features, 10, 20)
# test_data_loader = torch.utils.data.DataLoader(torch_test_dataset, batch_size=1)

model = gtnn.TGCN2(sequence_len, prediction_len, batch_size).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = torch.nn.MSELoss()

model.train()

for epoch in range(nb_epochs):
    print("=== epoch", epoch, "===")

    for (X, Y) in train_data_loader:
        if X.shape[0] != batch_size:
            break

        X = X.transpose(1, 2)
        Y = Y.transpose(1, 2)

        # Run model on input data
        H = model(X, edge_index)

        # Calculate the loss
        loss = loss_function(H, Y)

        # Perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("loss =", loss.item())

x = torch.Tensor(train_features.T[:,100:112].reshape(batch_size, 156, 12)).to(device)
y = model(x, edge_index)

print(x[0, 0, :] * max_feature_value)
print(y[0, 0, :] * max_feature_value)

# predictions = None
# y = None

# model.eval()

# with torch.no_grad:
#     for (data_inputs, data_labels) in test_data_loader:
#         predictions = model(torch.FloatTensor(data_inputs))
#         y = data_labels


# predictions = predictions.squeeze().transpose(0, 1)[:,0] * max_feature_value
# y = y.squeeze()[:,0] * max_feature_value

# print(predictions)
# print(y)