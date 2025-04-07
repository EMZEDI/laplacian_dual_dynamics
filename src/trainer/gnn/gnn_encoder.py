from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class GNNEncoder(nn.Module):
    """GNN encoder with configurable architecture using SAGE, GCN, GAT layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 10,
        gnn_type: str = 'sage',
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        # Initialize GNN layers
        self.convs = nn.ModuleList()

        # Input layer
        if gnn_type == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        elif gnn_type == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim))
        else:
            raise ValueError(f'Unknown GNN type: {gnn_type}')

        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim))

        # Output layer
        if gnn_type == 'sage':
            self.convs.append(SAGEConv(hidden_dim, output_dim))
        elif gnn_type == 'gcn':
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(hidden_dim, output_dim))

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
