import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power
from torch.nn import init

from Models.MLP import MLP


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def Comp_degree(A):
    out_degree = torch.sum(A, dim=0)
    in_degree = torch.sum(A, dim=1)
    if torch.cuda.is_available():
        diag = torch.eye(A.size()[0]).cuda()
    else:
        diag = torch.eye(A.size()[0])
    degree_matrix = diag * in_degree + diag * out_degree - torch.diagflat(torch.diagonal(A))
    return degree_matrix


class GraphConv_Ortega(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=128):
        super(GraphConv_Ortega, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)

        if num_layers == 1:
            init.xavier_uniform_(self.MLP.linear.weight)
            init.constant_(self.MLP.linear.bias, 0)
        else:
            for i in range(num_layers):
                init.xavier_uniform_(self.MLP.linears[i].weight)
                init.constant_(self.MLP.linears[i].bias, 0)

    def forward(self, features, A):
        b, n, d = features.shape
        assert (d == self.in_dim)
        A_norm = A
        deg_mat = Comp_degree(A_norm)
        if torch.cuda.is_available():
            frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.detach().cpu(),
                                                                    -0.5)).cuda()
        else:
            frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.detach().cpu(),
                                                                    -0.5))
        Laplacian = deg_mat - A_norm
        Laplacian_norm = frac_degree.matmul(Laplacian.matmul(frac_degree))
        _, U = torch.eig(Laplacian_norm, eigenvectors=True)
        repeated_U_t = U.t().repeat(b, 1, 1)
        repeated_U = U.repeat(b, 1, 1)
        agg_feats = torch.bmm(repeated_U_t, features)
        out = self.MLP(agg_feats.view(-1, d)).view(b, -1, self.out_dim)
        out = torch.bmm(repeated_U, out)
        return out


class Graph_CNN_ortega(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, final_dropout,
                 graph_pooling_type, device, adj):

        super(Graph_CNN_ortega, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.Adj = adj

        self.GCNs = torch.nn.ModuleList()
        self.GCNs.append(GraphConv_Ortega(self.input_dim, self.hidden_dim))
        for i in range(self.num_layers - 1):
            self.GCNs.append(GraphConv_Ortega(self.hidden_dim, self.hidden_dim))

        # Linear functions that maps the hidden representations to labels
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.Dropout(p=self.final_dropout),
            nn.PReLU(256),
            nn.Linear(256, output_dim))

    def forward(self, h):
        A = F.relu(self.Adj)
        for layer in self.GCNs:
            h = F.relu(layer(h, A))
        pooled = None
        if (self.graph_pooling_type == 'mean'):
            pooled = torch.mean(h, dim=1)
        if (self.graph_pooling_type == 'max'):
            pooled = torch.max(h, dim=1)[0]
        if (self.graph_pooling_type == 'sum'):
            pooled = torch.sum(h, dim=1)
        score = self.classifier(pooled)
        return score
