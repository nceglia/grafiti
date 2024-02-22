import scanpy as sc
import numpy as np
import seaborn as sns
import umap
import torch.nn.functional as F
import torch
from torch import Tensor
import torch_scatter
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import models
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import aggr
from torch_geometric.nn import MessagePassing
from sklearn import preprocessing
from torch_geometric.loader import DataLoader
import tqdm

class GrafitiEncoderLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GrafitiEncoderLayer, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def message(self, x_j, edge_attr):
        edge_attr = edge_attr.to(x_j.dtype) 
        return x_j / edge_attr.unsqueeze(-1) 

    def forward(self, x, edge_index, edge_attr):
        ret = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        ret = self.lin(ret) 
        return F.leaky_relu(ret, negative_slope=0.01)
    
class GrafitiDecoderLayer(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super(GrafitiDecoderLayer, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def message(self, x_j, edge_attr): 
        edge_attr = edge_attr.to(x_j.dtype)
        degree = x_j.size(0) 
        degree_normalized_message = x_j / edge_attr.unsqueeze(-1) 
        res = degree_normalized_message / degree
        return res

    def aggregate(self, inputs, index, dim_size=None):
        res = torch_scatter.scatter_mean(inputs, index, dim=0, dim_size=dim_size)
        return res

    def forward(self, x, edge_index, edge_attr):
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        transformed_features = x - aggr_out
        transformed_features = self.lin(transformed_features) 
        return F.leaky_relu(transformed_features, negative_slope=0.01)
    

class GrafitiEncoderModule(torch.nn.Module):
    def __init__(self, in_dim, layers=[10,10]):
        super(GrafitiEncoderModule, self).__init__()
        self.layers = layers
        self.conv = nn.ModuleList()
        lhidden_dim = self.layers[0]
        self.conv.append(GrafitiEncoderLayer(in_dim, lhidden_dim))
        for hidden_dim in self.layers[1:]:
            self.conv.append(GrafitiEncoderLayer(lhidden_dim, hidden_dim))
            lhidden_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        for conv in self.conv:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr).relu()
        return x

class GrafitiDecoderModule(torch.nn.Module):
    def __init__(self, in_dim, layers=[30,30]):
        super(GrafitiDecoderModule, self).__init__()
        self.layers = layers
        self.conv = nn.ModuleList()
        lhidden_dim = self.layers[0]
        self.conv.append(GrafitiDecoderLayer(in_dim, lhidden_dim))
        for hidden_dim in self.layers[1:]:
            self.conv.append(GrafitiDecoderLayer(lhidden_dim, hidden_dim))
            lhidden_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr):
        for conv in self.conv:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr).relu()
        return x

class GAE(object):

    def __init__(self, adata, layers=[10,10], lr=0.00001, distance_threshold=None, exponent=2, distance_scale=None, fov_key=False):
        self.lr = lr
        print("Generating PyTorch Geometric Dataset...")

        fovs = list(set(adata.obs[fov_key]))
        datas = []
        print("Building batches...")
        for f in tqdm.tqdm(fovs):
            fdata = adata[adata.obs[fov_key] == f].copy()
            if distance_threshold != None:
                distances = fdata.obsp["spatial_distances"]
                connectiv = fdata.obsp["spatial_connectivities"]
                rows, cols = distances.nonzero()
                for row, col in zip(rows, cols):
                    if distances[row, col] > distance_threshold:
                        connectiv[row, col] = 0
                fdata.obsp["spatial_connectivities"] = connectiv
            edges = fdata.obsp["spatial_connectivities"].nonzero()
            x = torch.from_numpy(fdata.X)
            x = x.float()
            e = torch.from_numpy(np.array(edges)).type(torch.int64)
            attrs = [fdata.obsp["spatial_distances"][x,y] for x,y in zip(*edges)]
            if distance_scale!=None:
                scaler = preprocessing.MinMaxScaler(feature_range=(0,distance_scale))
                attrs = scaler.fit_transform(np.array(attrs).reshape(-1,1)).reshape(1,-1)
                attrs = 1. / (np.array(attrs)**exponent)
                attrs = attrs[0]
            else:
                attrs = np.array(attrs)
            data = Data(x=x, edge_index=e, edge_attr=attrs)
            data.edge_attr = torch.from_numpy(data.edge_attr)
            datas.append(data)
        self.adata = adata
        self.encoder_layers = layers
        self.decoder_layers = list(reversed(layers[1:])) + [data.num_features]
        print("Setting up Model...")
        self.encoder = GrafitiEncoderModule(data.num_features,layers=self.encoder_layers, )
        self.decoder = GrafitiDecoderModule(layers[-1],layers=self.decoder_layers)
        self.gae = models.GAE(encoder=self.encoder,decoder=self.decoder)
        self.optimizer = torch.optim.Adadelta(self.gae.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.losses = []
        self.global_epoch = 0
        self.data = datas
        print("Ready to train!")

    def create_data_loader(self, batch_size=32):
        # Assuming self.data is a list of Data objects
        return DataLoader(self.data, batch_size=batch_size, shuffle=True)

    def train(self, epochs, update_interval=5, threshold=0, batch_size=None):
        if batch_size == None:
            batch_size = len(self.data)
        data_loader = self.create_data_loader(batch_size)
        prev_loss = 0.

        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                self.optimizer.zero_grad()
                z = self.gae.encode(batch.x, batch.edge_index, batch.edge_attr)
                reconstruction = self.gae.decode(z, batch.edge_index, batch.edge_attr)
                loss = self.loss(reconstruction, batch.x)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader)
            self.losses.append(avg_loss)

            if epoch % update_interval == 0:
                print(f"Epoch {epoch} Loss: {avg_loss}")

            if abs(avg_loss - prev_loss) < threshold:
                print("Minimum threshold reached!")
                break

            prev_loss = avg_loss

        print("Training Complete.")

    def __str__(self):
        fmt += "GAE Architecture\n\n"
        fmt += str(self.gae) + "\n"
        return fmt

    def plot(self):
        sns.lineplot(self.losses)

    def save(self, path):
        torch.save(self.gae.state_dict(), path)
    
    def load(self, path):
        state_dict = torch.load(path)
        self.gae.load_state_dict(state_dict)

    def load_embedding(self, adata, encoding_key="X_grafiti"):
        with torch.no_grad():
            zcpus = []
            for d in self.data:
                z = self.gae.encode(d.x, d.edge_index, d.edge_attr)
                zcpu = z.detach().numpy()
                zcpus.append(zcpu.T)
            zcpu = np.hstack(zcpus)
            adata.obsm[encoding_key] = zcpu.T