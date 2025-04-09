import numpy as np
import torch


class GraphBuilder:
    """Helper class to construct graphs from state batches without duplicate transitions."""

    def __init__(self, feature_dim, max_nodes=20000, device='cpu'):
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

        # Set to track unique undirected edges (stored as sorted tuples)
        self.edge_set = set()

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
                # Save the state as a tensor on the desired device
                self.node_features[idx] = torch.tensor(
                    state_feature, device=self.device
                )
                self.node_map[state_hash] = idx
                self.node_count += 1
                return idx
            else:
                raise RuntimeError('Maximum number of nodes reached.')

    def add_transition(self, state1, state2):
        """Add a transition between two states while avoiding duplicate transitions."""
        idx1 = self.add_state(state1)
        idx2 = self.add_state(state2)

        # Create a sorted tuple to represent the undirected edge uniquely
        edge_tuple = tuple(sorted((idx1, idx2)))
        if edge_tuple in self.edge_set:
            return  # Skip adding duplicate edge

        self.edge_set.add(edge_tuple)

        # Add edge in both directions for undirected graph
        edge = torch.tensor(
            [[idx1, idx2], [idx2, idx1]], dtype=torch.long, device=self.device
        )
        self.edge_index = torch.cat([self.edge_index, edge], dim=1)

        # Set the weights for the new edges
        weight = torch.tensor([1.0, 1.0], device=self.device)
        self.edge_weights = torch.cat([self.edge_weights, weight])

    def build_graph_from_batch(self, state_batch, next_state_batch):
        """Build graph from state transition batch."""
        for state, next_state in zip(state_batch, next_state_batch):
            self.add_transition(state, next_state)

    def get_full_graph(self):
        """Return the complete graph built so far."""
        x = self.node_features[: self.node_count].clone()
        edge_index = self.edge_index
        return {'x': x, 'edge_index': edge_index}

    def visualize_graph(self):
        """Visualize the graph using NetworkX."""
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.Graph()
        for i in range(self.node_count):
            G.add_node(i, label=str(i))

        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i].item()
            dst = self.edge_index[1, i].item()
            G.add_edge(src, dst)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        plt.savefig('graph.png')

    def test_graph_builder(self):
        # Create some synthetic states (using numpy arrays)
        state1 = np.array([0.1, 0.2, 0.3])
        state2 = np.array([0.4, 0.5, 0.6])
        state3 = np.array([0.7, 0.8, 0.9])

        state_batch = [state1, state2, state3]
        # Setting transitions that would form a cycle (state1 -> state2, state2 -> state3, state3 -> state1)
        next_state_batch = [state2, state3, state1]

        self.build_graph_from_batch(state_batch, next_state_batch)

        graph = self.get_full_graph()

        print('Node Features (x):')
        print(graph['x'])

        print('\nEdge Index (edges):')
        print(graph['edge_index'])

        print('\nEdge Weights:')
        print(self.edge_weights)

        # Visualize the graph (this will save the figure as "graph.png")
        self.visualize_graph()


if __name__ == '__main__':
    # Using a small maximum node count for testing
    obj = GraphBuilder(feature_dim=3, max_nodes=10, device='cpu')
    obj.test_graph_builder()
