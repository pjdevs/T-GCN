""" pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html"""
import torch
import torch.nn
import torch.utils
import torch.utils.data
import torch.nn as nn
from utils.data import generate_torch_dataset
from torch import Tensor
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np


adjacency = pd.read_csv("data/sz_adj.csv", header=None).values
features = pd.read_csv("data/sz_speed.csv", header=0).values
max_feature_value = np.max(features)

train_len = int(0.8*features.shape[0])
train_features = features[:train_len,]
test_features = features[train_len:,]

torch_train_dataset = generate_torch_dataset(train_features, 10, 20)
train_data_loader = torch.utils.data.DataLoader(torch_train_dataset)

torch_test_dataset = generate_torch_dataset(test_features, 10, 20)
test_data_loader = torch.utils.data.DataLoader(torch_test_dataset)

list=[]
for line in range(adjacency.shape[0]):
    for column in range(adjacency.shape[1]):
        if(adjacency[line][column]):
            list.append((line,column))
x=np.array(list)
x = torch.from_numpy(x)
edge_index=Tensor.transpose(x,0,1)

model = GCNConv(10, 15)
model2 = GCNConv(15, 20)
gru = nn.GRU(20, 25, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_function = torch.nn.MSELoss()

for epoch in range(1):
        print ("epoch=",epoch)
        for (data_inputs, data_labels) in train_data_loader:
            data_inputs=data_inputs[-1,:,:]
            data_inputs = Tensor.transpose(data_inputs,1,0)
            predictions = model(data_inputs,edge_index)
            predictions = model2(predictions,edge_index)
            predictions = Tensor.transpose(predictions,1,0)
            loss = loss_function(predictions, data_labels[-1,:,:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
