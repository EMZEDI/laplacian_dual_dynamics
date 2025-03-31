import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
import numpy as np
import time
from tqdm import tqdm

from src.trainer.al_torch import ALLOLoss


class GraphSAGE(nn.Module):
    """Standard GraphSAGE implementation."""
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        self.convs.append(SAGEConv(hidden_channels, out_channels))
    
    def forward(self, x, edge_index):
        """Forward pass through GraphSAGE."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGEWithALLOHeads(nn.Module):
    """GraphSAGE with multiple heads for ALLO loss."""
    
    def __init__(self, in_channels, hidden_channels, emb_dim, num_layers=2, dropout=0.2):
        super().__init__()
        # Base GraphSAGE encoder
        self.encoder = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # Base outputs hidden_channels dimensions
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Projection heads for ALLO loss
        self.start_head = nn.Linear(hidden_channels, emb_dim)
        self.end_head = nn.Linear(hidden_channels, emb_dim)
        self.constraint_head1 = nn.Linear(hidden_channels, emb_dim)
        self.constraint_head2 = nn.Linear(hidden_channels, emb_dim)
        
        self.hidden_dim = hidden_channels
        self.emb_dim = emb_dim
    
    def forward(self, x, edge_index):
        """Compute base node embeddings."""
        node_embeddings = self.encoder(x, edge_index)
        return node_embeddings
    
    def get_edge_embeddings(self, node_embeddings, edge_index):
        """Extract source and target embeddings for edges."""
        src_nodes, dst_nodes = edge_index
        src_emb = node_embeddings[src_nodes]
        dst_emb = node_embeddings[dst_nodes]
        return src_emb, dst_emb
    
    def forward_with_allo_heads(self, x, edge_index, edge_batch=None):
        """Forward pass for ALLO loss computation, returning all required representations."""
        # Get base node embeddings
        node_embeddings = self.forward(x, edge_index)
        
        # Use provided edge batch or all edges
        if edge_batch is None:
            edge_batch = edge_index
            
        # Get source and target embeddings for the edge batch
        src_nodes, dst_nodes = edge_batch
        
        # Apply projection heads to get the required representations
        start_rep = self.start_head(node_embeddings[src_nodes])
        end_rep = self.end_head(node_embeddings[dst_nodes])
        
        # For constraint representations, use all nodes
        constraint_rep1 = self.constraint_head1(node_embeddings)
        constraint_rep2 = self.constraint_head2(node_embeddings)
        
        return start_rep, end_rep, constraint_rep1, constraint_rep2


def train_epoch(model, data, optimizer, allo_loss, duals, barrier_coef, dual_velocities, batch_size=1024):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Create data loader for edges
    edge_indices = torch.randperm(data.edge_index.size(1))
    num_edges = edge_indices.size(0)
    
    # Process in batches
    for i in range(0, num_edges, batch_size):
        optimizer.zero_grad()
        
        # Get batch of edges
        batch_indices = edge_indices[i:min(i+batch_size, num_edges)]
        edge_batch = data.edge_index[:, batch_indices]
        
        # Forward pass with all heads
        start_rep, end_rep, constraint_rep1, constraint_rep2 = model.forward_with_allo_heads(
            data.x, data.edge_index, edge_batch
        )
        
        # Compute ALLO loss
        loss = allo_loss(
            start_rep,
            end_rep,
            constraint_rep1,
            constraint_rep2,
            duals=duals,
            barrier_coef=barrier_coef
        )
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
    
    # Update duals and barrier coefficient
    with torch.no_grad():
        barrier_coef = allo_loss.update_barrier_coefficient(barrier_coef)
        duals, dual_velocities = allo_loss.update_duals(
            duals, dual_velocities, barrier_coef
        )
    
    avg_loss = total_loss / num_batches
    return avg_loss, duals, barrier_coef, dual_velocities


def train(model, data, epochs=100, lr=0.01, weight_decay=5e-4, batch_size=1024):
    """Complete training procedure for GraphSAGE with ALLO Loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize ALLO loss
    allo_loss = ALLOLoss(d=model.emb_dim)
    
    # Initialize augmented Lagrangian parameters
    duals = torch.zeros((model.emb_dim, model.emb_dim), device=device)
    barrier_coef = torch.tensor(0.1, device=device)
    dual_velocities = torch.zeros((model.emb_dim, model.emb_dim), device=device)
    
    # For tracking metrics
    losses = []
    graph_losses = []
    constraint_losses = []
    
    print(f"Starting training on {device}")
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train for one epoch
        loss, duals, barrier_coef, dual_velocities = train_epoch(
            model, data, optimizer, allo_loss, duals, barrier_coef, dual_velocities, batch_size
        )
        
        # Get detailed metrics
        metrics = allo_loss.get_metrics()
        graph_loss = metrics['graph_loss']
        constraint_loss = metrics['dual_loss'] + metrics['barrier_loss']
        
        # Track metrics
        losses.append(loss)
        graph_losses.append(graph_loss)
        constraint_losses.append(constraint_loss)
        
        # Print progress
        epoch_time = time.time() - start_time
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Graph Loss: {graph_loss:.4f} | "
                  f"Constraint Loss: {constraint_loss:.4f} | "
                  f"Time: {epoch_time:.2f}s")
    
    # Final metrics
    print(f"\nTraining complete!")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Final Graph Loss: {graph_losses[-1]:.4f}")
    print(f"Final Constraint Loss: {constraint_losses[-1]:.4f}")
    
    return model, {
        'losses': losses,
        'graph_losses': graph_losses,
        'constraint_losses': constraint_losses,
        'final_duals': duals,
        'final_barrier_coef': barrier_coef
    }


def evaluate(model, data):
    """Evaluate the model and extract node embeddings."""
    model.eval()
    with torch.no_grad():
        # Get base node embeddings
        node_embeddings = model(data.x, data.edge_index)
        
        # Get embeddings through each head
        start_embeddings = model.start_head(node_embeddings)
        end_embeddings = model.end_head(node_embeddings)
        constraint_embeddings1 = model.constraint_head1(node_embeddings)
        constraint_embeddings2 = model.constraint_head2(node_embeddings)
        
        # Check orthogonality
        inner_products = torch.mm(constraint_embeddings1.t(), constraint_embeddings1) / constraint_embeddings1.size(0)
        diag_diff = torch.norm(inner_products - torch.eye(inner_products.size(0), device=inner_products.device))
        off_diag_norm = torch.norm(inner_products - torch.diag(torch.diag(inner_products)))
        
        # Return metrics
        metrics = {
            'diag_diff': diag_diff.item(),
            'off_diag_norm': off_diag_norm.item(),
            'node_embeddings': node_embeddings,
            'start_embeddings': start_embeddings,
            'end_embeddings': end_embeddings,
            'constraint_embeddings1': constraint_embeddings1,
            'constraint_embeddings2': constraint_embeddings2
        }
        
        return metrics


def create_sparse_graph(num_nodes, max_neighbors=2, seed=42):
    """Create a sparse graph where each node has at most 'max_neighbors' connections."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize edges
    edge_list = []
    
    # Step 1: Create several disconnected chains/components
    # This ensures the graph is very sparse
    num_components = num_nodes // 10  # About 10 nodes per component on average
    nodes_per_component = num_nodes // num_components
    
    for c in range(num_components):
        start_idx = c * nodes_per_component
        end_idx = min((c+1) * nodes_per_component, num_nodes)
        
        # Create a chain within this component
        for i in range(start_idx, end_idx - 1):
            edge_list.append([i, i+1])
            edge_list.append([i+1, i])  # Bidirectional edges
    
    # Step 2: Add some random edges to ensure connectivity but maintain sparsity
    neighbor_count = {i: 0 for i in range(num_nodes)}
    
    # Count existing neighbors
    for src, dst in edge_list:
        neighbor_count[src] = neighbor_count.get(src, 0) + 1
        neighbor_count[dst] = neighbor_count.get(dst, 0) + 1
    
    # Add a few more random edges, ensuring no node exceeds max_neighbors
    for _ in range(num_nodes // 5):  # Add about num_nodes/5 additional edges
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        
        # Skip if would exceed max neighbors or self-loop
        if (src == dst or 
            neighbor_count[src] >= max_neighbors or 
            neighbor_count[dst] >= max_neighbors):
            continue
            
        edge_list.append([src, dst])
        edge_list.append([dst, src])  # Add bidirectional
        
        neighbor_count[src] += 1
        neighbor_count[dst] += 1
    
    # Convert to tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    # Check sparsity and max neighbors
    degree = torch.zeros(num_nodes)
    for src, dst in edge_list:
        degree[src] += 1
    
    print(f"Graph statistics:")
    print(f"- Nodes: {num_nodes}")
    print(f"- Edges: {len(edge_list)}")
    print(f"- Average degree: {degree.mean().item():.2f}")
    print(f"- Max degree: {degree.max().item()}")
    print(f"- Nodes with 0 neighbors: {(degree == 0).sum().item()}")
    print(f"- Nodes with 1 neighbor: {(degree == 1).sum().item()}")
    print(f"- Nodes with 2 neighbors: {(degree == 2).sum().item()}")
    
    return edge_index


if __name__ == '__main__':
    # Generate sparse graph data
    num_nodes = 5000
    in_channels = 160
    hidden_channels = 64
    emb_dim = 80
    
    # Create random features
    x = torch.randn(num_nodes, in_channels)
    
    # Create sparse graph where each node has max 2 neighbors
    edge_index = create_sparse_graph(num_nodes, max_neighbors=2)
    
    # Create graph data
    data = Data(x=x, edge_index=edge_index)
    
    # Initialize model
    model = GraphSAGEWithALLOHeads(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        emb_dim=emb_dim,
        num_layers=2
    )
    
    # Train model
    model, history = train(model, data, epochs=30, batch_size=64)
    
    # Evaluate model
    eval_metrics = evaluate(model, data)
    print(f"Orthogonality diagonal difference: {eval_metrics['diag_diff']:.4f}")
    print(f"Orthogonality off-diagonal norm: {eval_metrics['off_diag_norm']:.4f}")
    
    # Visualize a few embeddings
    node_embeddings = eval_metrics['start_embeddings']
    print(f"Sample embeddings (first 5 nodes):")
    print(node_embeddings[:5])
    
    # Optional: save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, 'allo_graphsage_sparse.pt')