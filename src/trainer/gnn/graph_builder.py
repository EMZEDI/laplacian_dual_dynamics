import numpy as np
import torch


class GraphBuilder:
    """Helper class to construct graphs from state batches."""

    def __init__(self, feature_dim, max_nodes=10000, device='cpu'):
        self.feature_dim = feature_dim
        self.max_nodes = max_nodes
        self.device = device

        # Initialize node storage
        self.node_features = torch.zeros(max_nodes, feature_dim, device=device)
        self.node_count = 0
        self.node_map = {}  # Maps state hash to node index

        # Edge storage
        self.edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        self.edge_weights = torch.zeros(0, device=device)

    def _hash_state(self, state):
        """Create a hash for a state to check if it exists."""
        if isinstance(state, np.ndarray):
            return hash(state.tobytes())
        elif isinstance(state, torch.Tensor):
            return hash(state.cpu().numpy().tobytes())
        else:
            return hash(state)

    def add_state(self, state_feature):
        """Add a state to the graph if it doesn't exist."""
        state_hash = self._hash_state(state_feature)
        if state_hash in self.node_map:
            return self.node_map[state_hash]
        else:
            idx = self.node_count
            if idx < self.max_nodes:
                self.node_features[idx] = torch.tensor(
                    state_feature, device=self.device
                )
                self.node_map[state_hash] = idx
                self.node_count += 1
                return idx
            else:
                # If we're out of space, reuse an existing node (not ideal)
                return 0

    def add_transition(self, state1, state2):
        """Add a transition between two states."""
        idx1 = self.add_state(state1)
        idx2 = self.add_state(state2)

        # Add edge in both directions (undirected graph)
        edge = torch.tensor(
            [[idx1, idx2], [idx2, idx1]], dtype=torch.long, device=self.device
        )
        self.edge_index = torch.cat([self.edge_index, edge], dim=1)

        # Weight = 1.0 for transitions from data
        weight = torch.tensor([1.0, 1.0], device=self.device)
        self.edge_weights = torch.cat([self.edge_weights, weight])

    def build_graph_from_batch(self, state_batch, next_state_batch):
        """Build graph from state transition batch."""
        self.node_count = 0
        self.node_map = {}
        self.edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
        self.edge_weights = torch.zeros(0, device=self.device)

        # Add all states as nodes and transitions as edges
        for state, next_state in zip(state_batch, next_state_batch):
            self.add_transition(state, next_state)

        # Return a compact version of the graph
        return self.get_graph()

    def get_graph(self):
        """Return the current graph."""
        return {
            'x': self.node_features[: self.node_count],
            'edge_index': self.edge_index,
            'edge_attr': self.edge_weights,
            'num_nodes': self.node_count,
        }
