import torch
from torch_geometric_temporal.dataset import PemsBayDatasetLoader
import numpy as np
import pandas as pd


sequence_len = 1
prediction_len = 3
device = "cpu"

def create_adjacency(dataset):
    adjacency= np.zeros((len(dataset.features[0]),len(dataset.features[0])))
    for i in range(dataset.edge_index.shape[1]):
        adjacency[dataset.edge_index[0][i],dataset.edge_index[1][i]]=1
    return adjacency


loader = PemsBayDatasetLoader()
feat = torch.transpose(loader.X[:,0,:5000],0,1)
dataset = loader.get_dataset(num_timesteps_in=sequence_len, num_timesteps_out=prediction_len)

adj = create_adjacency(dataset)
df = pd.DataFrame(adj)
df.to_csv('adj.csv',header=False,index=False)
df = pd.DataFrame(feat.detach().numpy() )
df.to_csv('feat.csv',header=False,index=False)