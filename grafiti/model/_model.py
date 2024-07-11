import scanpy as sc
import scipy.sparse
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

    def __init__(self, adata, layers=[10,10], lr=0.00001, distance_threshold=None, exponent=2, distance_scale=None, device='cpu'):
        self.lr = lr
        self.device = torch.device(device)
        print("Generating PyTorch Geometric Dataset...")
        if distance_threshold != None:
            distances = adata.obsp["spatial_distances"]
            connectiv = adata.obsp["spatial_connectivities"]
            rows, cols = distances.nonzero()
            for row, col in zip(rows, cols):
                if distances[row, col] > distance_threshold:
                    connectiv[row, col] = 0
            adata.obsp["spatial_connectivities"] = connectiv
        edges = adata.obsp["spatial_connectivities"].nonzero()

        # Check if adata.X is a scipy.sparse.csr_matrix
        if scipy.sparse.issparse(adata.X):
            x = torch.from_numpy(adata.X.toarray())
        else:
            x = torch.from_numpy(adata.X)

        x = x.float().to(self.device)
        e = torch.from_numpy(np.array(edges)).type(torch.int64).to(self.device)
        attrs = [adata.obsp["spatial_distances"][x,y] for x,y in zip(*edges)]
        if distance_scale!=None:
            scaler = preprocessing.MinMaxScaler(feature_range=(0,distance_scale))
            attrs = scaler.fit_transform(np.array(attrs).reshape(-1,1)).reshape(1,-1)
            attrs = 1. / (np.array(attrs)**exponent)
            attrs = attrs[0]
        else:
            attrs = np.array(attrs)
        data = Data(x=x, edge_index=e, edge_attr=attrs)
        self.adata = adata
        data.edge_attr = torch.from_numpy(data.edge_attr).to(self.device)
        self.encoder_layers = layers
        self.decoder_layers = list(reversed(layers[1:])) + [data.num_features]
        print("Setting up Model...")
        self.encoder = GrafitiEncoderModule(data.num_features,layers=self.encoder_layers).to(self.device)
        self.decoder = GrafitiDecoderModule(layers[-1],layers=self.decoder_layers).to(self.device)
        self.gae = models.GAE(encoder=self.encoder,decoder=self.decoder).to(self.device)
        self.optimizer = torch.optim.Adadelta(self.gae.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.losses = []
        self.global_epoch = 0
        self.data = data
        print("Ready to train!")

    def train(self, epochs, update_interval=5, threshold=0.001, patience=10):
        prev_loss = np.inf
        best_loss = np.inf
        patience_counter = 0 # Counter to track the number of epochs without improvement

        for i in range(epochs):
            self.optimizer.zero_grad()  
            z = self.gae.encode(self.data.x, self.data.edge_index, self.data.edge_attr)
            reconstruction = self.gae.decode(z, self.data.edge_index, self.data.edge_attr)
            loss = self.loss(reconstruction, self.data.x)
            loss.backward()
            self.optimizer.step() 
            self.losses.append(loss.item())
            if i % update_interval == 0:
                print("Epoch {} ** iteration {} ** Loss: {}".format(self.global_epoch, i, np.mean(self.losses[-update_interval:])))
            self.global_epoch += 1
            curr_loss = loss.item()

            # Check for improvement
            if curr_loss < best_loss - threshold:
                best_loss = curr_loss
                patience_counter = 0 # Reset the counter if there is an improvement
            else:
                patience_counter += 1

            # Early stopping condition
            if patience_counter >= patience:
                print("Early stopping due to no improvement over {} epochs.".format(patience))
                break

            prev_loss = curr_loss # Update previous loss

        print("Training Complete.")

    def __str__(self):
        fmt = "Pytorch Dataset\n\n"
        fmt += str(self.data) + "\n\n"
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
            z = self.gae.encode(self.data.x, self.data.edge_index, self.data.edge_attr)
            zcpu = z.detach().cpu().numpy()
            adata.obsm[encoding_key] = zcpu
