from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class GNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        gnn_type: str = 'gcn',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim))

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Initial projection
        h = F.relu(self.input_proj(x))

        # GNN layers with skip connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            # Message passing
            h_conv = conv(h, edge_index)
            # Apply nonlinearity and dropout
            h_conv = F.relu(h_conv)
            h_conv = F.dropout(h_conv, p=self.dropout, training=self.training)
            # Skip connection
            h = h + h_conv
            # Layer normalization
            h = norm(h)

        # Final projection
        out = self.output_proj(h)
        return out
