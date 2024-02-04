from os.path import join
from collections import OrderedDict

import pdb
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer, TransformerConv
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gavgp, global_max_pool as gmp, global_add_pool as gap
from torch_geometric.transforms.normalize_features import NormalizeFeatures

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class NormalizeFeaturesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x[:, :12] = data.x[:, :12] / data.x[:, :12].max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class NormalizeEdgesV2(object):
    r"""Column-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.edge_attr = data.edge_attr.type(torch.cuda.FloatTensor)
        data.edge_attr = data.edge_attr / data.edge_attr.max(0, keepdim=True)[0]
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class DeepGraphConv_Surv(torch.nn.Module):
    def __init__(self, edge_agg='latent', resample=0, num_features=1024, hidden_dim=256, 
        linear_dim=256, use_edges=False, dropout=0.25, n_classes=4):
        super(DeepGraphConv_Surv, self).__init__()
        self.use_edges = use_edges
        self.resample = resample
        self.edge_agg = edge_agg

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample)])

        self.conv1 = GINConv(Seq(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(Seq(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        
        self.path_attention_head = Attn_Net_Gated(L=hidden_dim, D=hidden_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        self.classifier = torch.nn.Linear(hidden_dim, n_classes)

    def forward(self, **kwargs):
        data = kwargs['x_path']
        x = data.x
        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            edge_index = data.edge_latent
        batch = data.batch
        edge_attr = None

        if self.resample:
            x = self.fc(x)

        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        if self.pool:
            x1, edge_index, _, batch, perm, score = self.pool1(x1, edge_index, None, batch)
            x1_cat = torch.cat([gmp(x1, batch), gap(x1, batch)], dim=1)

        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        if self.pool:
            x2, edge_index, _, batch, perm, score = self.pool2(x2, edge_index, None, batch)
            x2_cat = torch.cat([gmp(x2, batch), gap(x2, batch)], dim=1)

        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))
        

        h_path = x3

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path)
        h_path = self.path_rho(h_path).squeeze()
        h = h_path # [256] vector
        
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards, S, Y_hat, A_path, None



class PatchGCN_Subtype(torch.nn.Module):
    def __init__(self, input_dim=512, n_classes=4, num_layers=1, multires=False, resample=0,
        fusion=None, hidden_dim=128, linear_dim=64, pool=False, dropout=0., **kwargs):
        super(PatchGCN_Subtype, self).__init__()
        self.fusion = fusion
        self.pool = pool
        self.multires = multires
        self.num_layers = num_layers-1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers+1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            # conv = TransformerConv(hidden_dim, hidden_dim)
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=False)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim*num_layers, hidden_dim*num_layers), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim*num_layers, D=hidden_dim*num_layers, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim*num_layers, hidden_dim*num_layers), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim*num_layers, n_classes)    

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self = self.to(device)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = None

        x = self.fc(x)
        x_ = x 
        
        if len(self.layers)>0:
            x = self.layers[0].conv(x_, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)
            for layer in self.layers[1:]:
                x = layer(x, edge_index, edge_attr)
                x_ = torch.cat([x_, x], axis=1)
        
        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        
        return logits, Y_prob, Y_hat, None, None