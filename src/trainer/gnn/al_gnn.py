import math
import os
import random
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
from torch_geometric.data import Data, Batch
from scipy.interpolate import Rbf
import wandb

from .gnn_encoder import GNNEncoder
from .graph_builder import GraphBuilder
from .al_torch import ALLOLoss


class LaplacianGNNTrainer:
    def __init__(
        self,
        # Environment settings
        env_name: str,
        env_family: str,
        # Model architecture
        input_dim: int,
        hidden_dim: int = 128,
        d: int = 16,  # Output dimension
        gnn_type: str = 'gcn',
        num_layers: int = 5,
        dropout: float = 0.0,
        # Training settings
        batch_size: int = 128,
        learning_rate: float = 5e-5,
        total_train_steps: int = 100000,
        discount: float = 0.99,
        # Augmented Lagrangian parameters
        use_barrier_normalization: bool = True,
        lr_barrier_coefs: float = 1e-3,
        min_barrier_coefs: float = 1e-5,
        max_barrier_coefs: float = 1e5,
        lr_duals: float = 1e-3,
        lr_dual_velocities: float = 1e-3,
        # Data and replay settings
        replay_buffer=None,
        # Logging and saving
        print_freq: int = 100,
        save_model: bool = True,
        save_model_every: int = 10000,
        do_plot_eigenvectors: bool = False,
        log_eigenvectors: bool = True,
        use_wandb: bool = True,
        # Other
        seed: int = 42,
        device: str = None,
        env=None,
        eigvec_dict=None,
        eigval_precision_order: int = 6,
    ):
        # Store parameters
        self.reset_counters()
        self.env_name = env_name
        self.env_family = env_family
        self.d = d  # Number of laplacian eigenfunctions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_steps = total_train_steps
        self.discount = discount
        self.print_freq = print_freq
        self.save_model = save_model
        self.save_model_every = save_model_every
        self.do_plot_eigenvectors = do_plot_eigenvectors
        self.log_eigenvectors = log_eigenvectors
        self.use_wandb = use_wandb
        self.replay_buffer = replay_buffer
        self.seed = seed
        self.env = env
        self.eigvec_dict = eigvec_dict
        self.eigval_precision_order = eigval_precision_order
        self.train_loader = (
            None  # to be later replaced with the full PyG Graph train loader
        )

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f'Using device: {self.device}')

        # Build environment and collect experience
        self.build_environment()
        self.collect_experience()

        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)

        # Create graph builder
        self.graph_builder = GraphBuilder(feature_dim=input_dim, device=self.device)
        self.build_complete_graph()

        # Create GNN encoder
        self.gnn = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=d,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout,
        )
        self.gnn = self.gnn.to(self.device)

        # Create ALLO loss
        self.allo_loss = ALLOLoss(
            d=d,
            barrier_initial_val=2.0,
            lr_duals=lr_duals,
            lr_barrier_coefs=lr_barrier_coefs,
            min_duals=-1000.0,
            max_duals=1000.0,
            min_barrier_coefs=min_barrier_coefs,
            max_barrier_coefs=max_barrier_coefs,
            error_update_rate=0.05,
            device=self.device,
        )

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_train_steps,
            eta_min=self.learning_rate / 10,
        )

        # Initialize parameters
        self.dual_params = torch.zeros((d, d), device=self.device)
        self.dual_params = torch.tril(self.dual_params)  # Use lower triangular part
        self.barrier_coef = torch.tensor(1.0, device=self.device)
        self.dual_velocities = torch.zeros((d, d), device=self.device)
        self.dual_velocities = torch.tril(self.dual_velocities)

        # Create tracking variables
        self._global_step = 0
        self._best_cosine_similarity = -1
        self.train_info = OrderedDict()
        self._date_time = datetime.now().strftime('%Y%m%d%H%M%S')
        self.permutation_array = torch.arange(d, device=self.device)
        self.past_permutation_array = torch.arange(d, device=self.device)

        # Print multiplicity info for eigenvectors if available
        if self.eigvec_dict is not None:
            print('Eigenvalue multiplicities:')
            for eigval in sorted(self.eigvec_dict.keys(), reverse=True):
                multiplicity = len(self.eigvec_dict[eigval])
                print(f'Eigenvalue {eigval} has multiplicity {multiplicity}')

        # Initialize wandb if required
        if self.use_wandb:
            self.logger = wandb.init(
                project=f'laplacian-gnn-{env_name}',
                name=f'{gnn_type}_{hidden_dim}_{d}_{learning_rate}',
                config={
                    'env_name': env_name,
                    'env_family': env_family,
                    'gnn_type': gnn_type,
                    'hidden_dim': hidden_dim,
                    'dimensions': d,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'total_steps': total_train_steps,
                    'discount': discount,
                    'seed': seed,
                },
            )

            # Log eigenvalues if available
            if self.env is not None and hasattr(self.env.unwrapped, 'get_eigenvalues'):
                eigenvalues = self.env.unwrapped.get_eigenvalues()
                eigval_dict = {
                    f'eigval_{i}': eigenvalues[i]
                    for i in range(min(len(eigenvalues), self.d))
                }
                self.logger.log(eigval_dict)

    def build_complete_graph(self):
        """Build graph from actual replay buffer transitions."""
        print('Building transition-based graph from replay buffer...')
        start_time = time.time()

        # Sample transitions from replay buffer
        n_samples = min(50000, self.replay_buffer.current_size)
        states, future_states = self.replay_buffer.sample_pairs(
            batch_size=n_samples,
            discount=1.0,  # Use direct transitions only
        )

        # Convert transitions to tensors
        state_tensors = self._convert_steps_to_tensor(states)
        future_state_tensors = self._convert_steps_to_tensor(future_states)

        # Add transitions to graph builder
        for state, future_state in zip(state_tensors, future_state_tensors):
            self.graph_builder.add_transition(state, future_state)

        # Get final graph data
        self.full_graph = self.graph_builder.get_full_graph()
        self.graph_builder.visualize_graph()

        # Report statistics
        total_nodes = self.full_graph['x'].shape[0]
        total_edges = self.full_graph['edge_index'].shape[1]
        print(f'Graph built with {total_nodes} nodes and {total_edges} edges')
        print(f'Graph building time: {time.time() - start_time:.2f} seconds')

        # Create PyG Data object
        from torch_geometric.data import Data

        x = self.full_graph['x']
        edge_index = self.full_graph['edge_index']

        print(f'x shape: {x.shape}')
        print(f'edge_index shape: {edge_index.shape}')

        self.graph_data = Data(x=x, edge_index=edge_index)

        # Create dataloader
        from torch_geometric.loader import NeighborLoader

        self.train_loader = NeighborLoader(
            self.graph_data,
            num_neighbors=[10, 10],
            batch_size=self.batch_size,
            shuffle=True,
        )

        print(f'DataLoader created with {len(self.train_loader)} batches')

    def _get_train_batch(self):
        """Get a training batch from the replay buffer."""
        state, future_state = self.replay_buffer.sample_pairs(
            batch_size=self.batch_size,
            discount=self.discount,
        )
        uncorrelated_state_1 = self.replay_buffer.sample_steps(self.batch_size)
        uncorrelated_state_2 = self.replay_buffer.sample_steps(self.batch_size)

        # Convert to tensors based on observation mode (specific to your env)
        # This is a simplified example - adapt based on your observation format
        state_tensor = self._convert_steps_to_tensor(state)
        future_tensor = self._convert_steps_to_tensor(future_state)
        uncorr_tensor_1 = self._convert_steps_to_tensor(uncorrelated_state_1)
        uncorr_tensor_2 = self._convert_steps_to_tensor(uncorrelated_state_2)

        return {
            'state': state_tensor.to(self.device),
            'future_state': future_tensor.to(self.device),
            'uncorrelated_state_1': uncorr_tensor_1.to(self.device),
            'uncorrelated_state_2': uncorr_tensor_2.to(self.device),
        }

    def _convert_steps_to_tensor(self, steps):
        """Convert replay buffer steps to tensors based on observation type."""
        if hasattr(self, 'obs_mode'):
            # Handle based on configured observation mode
            if self.obs_mode in ['xy']:
                obs_batch = [
                    s.step.agent_state['xy_agent'].astype(np.float32) for s in steps
                ]
                return torch.tensor(np.stack(obs_batch, axis=0), device=self.device)
            elif self.obs_mode in ['pixels', 'both']:
                obs_batch = [s.step.agent_state['pixels'] for s in steps]
                return (
                    torch.tensor(
                        np.stack(obs_batch, axis=0), device=self.device
                    ).float()
                    / 255
                )
            elif self.obs_mode in ['grid', 'both-grid']:
                obs_batch = [
                    s.step.agent_state['grid'].astype(np.float32) / 255 for s in steps
                ]
                return torch.tensor(np.stack(obs_batch, axis=0), device=self.device)

        # If obs_mode not set or not one of the above, try to infer from data structure
        if hasattr(steps[0].step, 'agent_state'):
            if isinstance(steps[0].step.agent_state, dict):
                if 'xy_agent' in steps[0].step.agent_state:
                    obs_batch = [
                        s.step.agent_state['xy_agent'].astype(np.float32) for s in steps
                    ]
                    return torch.tensor(np.stack(obs_batch, axis=0), device=self.device)
                elif 'grid' in steps[0].step.agent_state:
                    obs_batch = [
                        s.step.agent_state['grid'].astype(np.float32) / 255
                        for s in steps
                    ]
                    return torch.tensor(np.stack(obs_batch, axis=0), device=self.device)
                elif 'pixels' in steps[0].step.agent_state:
                    obs_batch = [s.step.agent_state['pixels'] for s in steps]
                    return (
                        torch.tensor(
                            np.stack(obs_batch, axis=0), device=self.device
                        ).float()
                        / 255
                    )

        # Fallback - try to convert directly assuming basic structure
        return torch.tensor(
            [s.step.observation for s in steps], device=self.device
        ).float()

    def train(self):
        """Run the full training loop with improved GNN-specific adaptations."""
        print('Starting training...')
        start_time = time.time()

        # Create folders for saving
        os.makedirs(f'./results/models/{self.env_name}', exist_ok=True)
        if self.do_plot_eigenvectors:
            os.makedirs(f'./results/visuals/{self.env_name}', exist_ok=True)

        # Training loop
        self.permute_step = self.total_train_steps // 10
        step = 0
        epoch = 0

        # Get all available states for global evaluations
        all_states = None
        if self.is_tabular:
            all_states = torch.tensor(self.get_states(), device=self.device).float()
            all_edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)

        while step < self.total_train_steps:
            epoch += 1
            print(f'Starting epoch {epoch}')

            # Get a direct batch from replay buffer instead of using the graph sampler
            # This more closely matches how the JAX implementation works
            batch_data = self._get_train_batch()

            # Process states with GNN to get node embeddings
            self.gnn.train()

            # Process start and future states through GNN
            start_states = batch_data['state']
            future_states = batch_data['future_state']

            # For orthogonality constraints, use uncorrelated states
            uncorrelated_1 = batch_data['uncorrelated_state_1']
            uncorrelated_2 = batch_data['uncorrelated_state_2']

            # Check if this is a permutation step
            # In the train method, replace permutation logic:
            is_permutation_step = False
            if step > 10000:  # Don't permute early in training
                # Exponential decay schedule for permutation frequency
                permute_prob = 0.2 * math.exp(-0.5 * step / self.total_train_steps)
                is_permutation_step = random.random() < permute_prob

            if is_permutation_step:
                # Keep some continuity with previous permutation
                self.past_permutation_array = self.permutation_array.clone()
                partial_perm = torch.randperm(self.d // 2, device=self.device)
                self.permutation_array[: self.d // 2] = partial_perm
                print(f'Partial permutation updated at step {step + 1}')

            # Process through GNN - use full graph for start/future and dummy edges for constraints
            # This better reflects actual dynamics while maintaining independence in the constraints
            full_edge_index = self.full_graph['edge_index']
            dummy_edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)

            # Get embeddings
            start_representation = self.gnn(start_states, full_edge_index)
            end_representation = self.gnn(future_states, full_edge_index)

            # Use uncorrelated states with dummy edges for orthogonality constraints
            constraint_representation_1 = self.gnn(uncorrelated_1, dummy_edge_index)
            constraint_representation_2 = self.gnn(uncorrelated_2, dummy_edge_index)

            # Apply permutation
            start_representation = self.permute_representations(start_representation)
            end_representation = self.permute_representations(end_representation)
            constraint_representation_1 = self.permute_representations(
                constraint_representation_1
            )
            constraint_representation_2 = self.permute_representations(
                constraint_representation_2
            )

            # Check for NaNs in representations and skip if found
            if (
                torch.isnan(start_representation).any()
                or torch.isnan(end_representation).any()
                or torch.isnan(constraint_representation_1).any()
                or torch.isnan(constraint_representation_2).any()
            ):
                print('Warning: NaN detected in representations, skipping batch')
                continue

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute ALLO loss
            try:
                loss = self.allo_loss(
                    start_representation=start_representation,
                    end_representation=end_representation,
                    constraint_representation_1=constraint_representation_1,
                    constraint_representation_2=constraint_representation_2,
                    duals=self.dual_params,
                    barrier_coef=self.barrier_coef,
                )

                # Check for NaN loss
                if torch.isnan(loss).any():
                    print('Warning: NaN loss detected, skipping batch')
                    continue

                # Backward pass and update
                loss.backward()

                # Add gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), max_norm=0.5)

                self.optimizer.step()
                self.lr_scheduler.step()

                # Update dual parameters and barrier coefficient
                with torch.no_grad():
                    self.barrier_coef = self.allo_loss.update_barrier_coefficient(
                        self.barrier_coef
                    )
                    self.dual_params, self.dual_velocities = (
                        self.allo_loss.update_duals(
                            self.dual_params, self.dual_velocities, self.barrier_coef
                        )
                    )

                # Extract metrics
                metrics = {}
                for key, value in self.allo_loss.metrics.items():
                    metrics[key] = value

                # Update counter
                self._global_step += 1
                step += 1

            except Exception as e:
                print(f'Error in loss computation: {e}')
                import traceback

                traceback.print_exc()
                continue

            # Log and print info
            is_log_step = ((step + 1) % self.print_freq) == 0
            if is_log_step:
                # Calculate steps per second
                elapsed = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed

                # Store metrics
                self.train_info['loss_total'] = loss.item()
                self.train_info['graph_loss'] = metrics.get('graph_loss', 0.0)
                self.train_info['dual_loss'] = metrics.get('dual_loss', 0.0)
                self.train_info['barrier_loss'] = metrics.get('barrier_loss', 0.0)
                self.train_info['steps_per_second'] = steps_per_sec

                # Compute cosine similarity with all states if available
                if self.is_tabular and all_states is not None:
                    # Evaluate on all states for more reliable metrics
                    self.gnn.eval()
                    with torch.no_grad():
                        cosine_similarity, similarities = (
                            self.compute_cosine_similarity()
                        )
                        self.train_info['cosine_similarity'] = cosine_similarity
                        metrics['cosine_similarity'] = cosine_similarity

                        # Add individual cosine similarities
                        for i, sim in enumerate(similarities):
                            metrics[f'cosine_similarity_{i}'] = sim

                    # Also compute maximal and simple cosine similarity if methods available
                    if hasattr(self, 'compute_cosine_similarity_simple'):
                        if hasattr(self, 'compute_maximal_cosine_similarity'):
                            maximal_cs, maximal_sim = (
                                self.compute_maximal_cosine_similarity()
                            )
                            self.train_info['max_cos_sim'] = maximal_cs
                            metrics['maximal_cosine_similarity'] = maximal_cs

                            for i, sim in enumerate(maximal_sim):
                                metrics[f'maximal_cosine_similarity_{i}'] = sim

                        (
                            cs_simple,
                            sim_simple,
                            permuted_cs_simple,
                            permuted_sim_simple,
                        ) = self.compute_cosine_similarity_simple()
                        self.train_info['cos_sim_s'] = cs_simple
                        self.train_info['cos_sim_s_permuted'] = permuted_cs_simple
                        metrics['cosine_similarity_simple'] = cs_simple
                        metrics['cosine_similarity_simple_permuted'] = (
                            permuted_cs_simple
                        )

                        for i, sim in enumerate(sim_simple):
                            metrics[f'cosine_similarity_simple_{i}'] = sim
                        for i, sim in enumerate(permuted_sim_simple):
                            metrics[f'cosine_similarity_simple_permuted_{i}'] = sim

                # Print info
                self._print_train_info()

                # Log to wandb
                if self.use_wandb:
                    metrics['step'] = step
                    metrics['global_step'] = self._global_step
                    metrics['examples'] = self._global_step * self.batch_size
                    metrics['wall_clock_time'] = elapsed
                    metrics['steps_per_second'] = steps_per_sec
                    metrics['barrier_coefficient'] = self.barrier_coef.item()
                    self.logger.log(metrics)

            # Save model checkpoint
            is_last_step = step >= self.total_train_steps
            is_save_step = self.save_model and (
                ((step % self.save_model_every) == 0) or is_last_step
            )
            if is_save_step:
                self.save_checkpoint(self.train_info.get('cosine_similarity'))

            # Plot eigenvectors if requested and at the end
            if self.do_plot_eigenvectors and is_last_step:
                self.plot_eigenvectors()

        print(f'Training finished in {time.time() - start_time:.2f} seconds.')
        return self.gnn

    @property
    def is_tabular(self):
        """Whether this is a tabular environment."""
        return self.env_family in ['Grid-v0']

    def _print_train_info(self):
        """Print training info."""
        header_str = f'=== Step {self._global_step} ==='
        print(header_str)
        for key, value in self.train_info.items():
            print(f'{key}: {value:.6f}')
        print('=' * len(header_str))

    def build_environment(self):
        """Build the appropriate environment based on env_family."""
        if self.is_tabular:
            self.build_tabular_environment()
        elif self.env_family == 'Atari-v5':
            self.build_atari_environment()
        else:
            raise ValueError(f'Invalid environment family: {self.env_family}')

    def build_tabular_environment(self):
        """Build tabular environment and load eigenvectors."""
        if self.env is not None:
            print('Environment already initialized, skipping build_environment')
            return

        # Properly set save_eig attribute to True by default
        if not hasattr(self, 'save_eig'):
            self.save_eig = True

        # Load eigenvectors and eigenvalues if they exist
        path_eig = f'./src/env/grid/eigval/{self.env_name}.npz'

        # First check if the file exists directly
        if os.path.exists(path_eig):
            print(f'Found cached eigenvectors at {path_eig}')
            try:
                from src.env.grid.utils import load_eig

                eig, eig_not_found = load_eig(path_eig)
                print('Successfully loaded cached eigenvectors')
            except Exception as e:
                print(f'Error loading cached eigenvectors: {e}')
                eig = None
                eig_not_found = True
        else:
            print(
                f'No cached eigenvectors found at {path_eig}, will compute and save them'
            )
            eig = None
            eig_not_found = True

        # Create environment
        try:
            import gymnasium as gym
            from gymnasium.wrappers import TimeLimit
            from src.env.wrapper.norm_obs import NormObs

            path_txt_grid = f'./src/env/grid/txts/{self.env_name}.txt'
            env = gym.make(
                self.env_family,
                path=path_txt_grid,
                render_mode='rgb_array',
                use_target=False,
                eig=eig,
                obs_mode=self.obs_mode if hasattr(self, 'obs_mode') else 'xy',
                window_size=self.window_size if hasattr(self, 'window_size') else None,
            )

            # Wrap environment with time limit
            max_steps = getattr(self, 'max_episode_steps', 1000)
            env = TimeLimit(env, max_episode_steps=max_steps)

            # Wrap environment with observation normalization
            reduction_factor = getattr(self, 'reduction_factor', 1.0)
            env = NormObs(env, reduction_factor=reduction_factor)

            # Set seed
            env.reset(seed=self.seed)

            # Set environment as attribute
            self.env = env

            # Save eigenvectors if needed - force saving if not found
            if eig_not_found and self.save_eig:
                print(f'Computing and saving eigenvectors to {path_eig}...')
                try:
                    os.makedirs(os.path.dirname(path_eig), exist_ok=True)
                    self.env.unwrapped.save_eigenpairs(path_eig)
                    print(f'Successfully saved eigenvectors to {path_eig}')
                except Exception as e:
                    print(f'Error saving eigenvectors: {e}')
                    import traceback

                    traceback.print_exc()

            # Process eigenvalues and eigenvectors
            if hasattr(self.env.unwrapped, 'round_eigenvalues') and hasattr(
                self, 'eigval_precision_order'
            ):
                self.env.unwrapped.round_eigenvalues(self.eigval_precision_order)

            if hasattr(self.env.unwrapped, 'get_eigenvalues'):
                eigenvalues = self.env.unwrapped.get_eigenvalues()
                print(f'Environment: {self.env_name}')
                print(f'Environment eigenvalues: {eigenvalues}')

                # Create eigenvector dictionary if not provided
                if not self.eigvec_dict and hasattr(
                    self.env.unwrapped, 'get_eigenvectors'
                ):
                    real_eigval = eigenvalues[: self.d]
                    real_eigvec = self.env.unwrapped.get_eigenvectors()[:, : self.d]

                    eigvec_dict = {}
                    for i, eigval in enumerate(real_eigval):
                        if eigval not in eigvec_dict:
                            eigvec_dict[eigval] = []
                        eigvec_dict[eigval].append(
                            torch.tensor(real_eigvec[:, i], device=self.device).float()
                        )
                    self.eigvec_dict = eigvec_dict
                    print(
                        f'Created eigenvector dictionary with {len(eigvec_dict)} unique eigenvalues'
                    )

        except Exception as e:
            print(f'Failed to build tabular environment: {e}')
            import traceback

            traceback.print_exc()

    def build_transition_graph(self):
        """Build graph from actual replay buffer transitions."""
        print('Building transition-based graph from replay buffer...')
        start_time = time.time()

        # Sample a substantial number of transitions from replay buffer
        n_samples = min(50000, self.replay_buffer.current_size)
        states, future_states = self.replay_buffer.sample_pairs(
            batch_size=n_samples,
            discount=1.0,  # Use direct transitions only
        )

        # Initialize graph builder and add transitions
        print(f'Processing {n_samples} transitions...')

        # Convert transitions to tensors
        state_tensors = self._convert_steps_to_tensor(states)
        future_state_tensors = self._convert_steps_to_tensor(future_states)

        # Add transitions to graph builder
        for state, future_state in zip(state_tensors, future_state_tensors):
            self.graph_builder.add_transition(state, future_state)

        # Get graph data
        self.full_graph = self.graph_builder.get_full_graph()
        self.graph_builder.visualize_graph()

        # Create PyG Data object
        x = self.full_graph['x']
        edge_index = self.full_graph['edge_index']
        self.graph_data = Data(x=x, edge_index=edge_index)

        # Report statistics
        print(f'Graph built with {x.shape[0]} nodes and {edge_index.shape[1]} edges')
        print(f'Graph building time: {time.time() - start_time:.2f} seconds')

    def build_atari_environment(self):
        """Build Atari environment."""
        if self.env is not None:
            return

        try:
            import gymnasium as gym
            from gymnasium.wrappers import TimeLimit
            from src.env.wrapper.norm_obs import NormObsAtari

            env_name = f'ALE/{self.env_name}-v5'
            env = gym.make(env_name)

            # Wrap environment with observation normalization
            env = NormObsAtari(env)

            # Wrap environment with time limit
            max_steps = getattr(self, 'max_episode_steps', 1000)
            env = TimeLimit(env, max_episode_steps=max_steps)

            # Set seed
            env.reset(seed=self.seed)

            # Set environment as attribute
            self.env = env
        except Exception as e:
            print(f'Failed to build Atari environment: {e}')
            import traceback

            traceback.print_exc()

    def collect_experience(self) -> None:
        """Collect experience for the replay buffer if needed."""
        if self.replay_buffer is None or self.replay_buffer.current_size > 0:
            print('Replay buffer already contains data, skipping collection')
            return

        if not hasattr(self, 'n_samples'):
            print('No n_samples attribute, setting to default 100000')
            self.n_samples = 100000

        try:
            # Create agent for collection
            from src.policy import DiscreteUniformRandomPolicy as Policy
            from src.agent.agent import BehaviorAgent as Agent
            from src.tools import timer_tools

            policy = Policy(num_actions=self.env.action_space.n, seed=self.seed)
            agent = Agent(policy)

            # Collect trajectories from random actions
            print('Start collecting samples.')
            timer = timer_tools.Timer()
            total_n_steps = 0
            collect_batch = 10_000

            while total_n_steps < self.n_samples:
                n_steps = min(collect_batch, self.n_samples - total_n_steps)
                steps = agent.collect_experience(self.env, n_steps)
                self.replay_buffer.add_steps(steps)
                total_n_steps += n_steps
                print(f'({total_n_steps}/{self.n_samples}) steps collected.')

            time_cost = timer.time_cost()
            print(f'Data collection finished, time cost: {time_cost}s')

            # Plot visitation counts for xy environments
            if hasattr(self, 'obs_mode') and self.obs_mode in ['xy']:
                self.plot_visitation_counts(timer)

        except Exception as e:
            print(f'Failed to collect experience: {e}')
            import traceback

            traceback.print_exc()

    def plot_visitation_counts(self, timer):
        """Plot visitation counts for tabular environments."""
        try:
            (
                min_visitation,
                max_visitation,
                visitation_entropy,
                max_entropy,
                visitation_freq,
            ) = self.replay_buffer.plot_visitation_counts(
                self.env.get_states()['xy_agent'],
                self.env_name,
                self.env.unwrapped.get_grid().astype(bool),
            )
            time_cost = timer.time_cost()
            print(f'Visitation evaluated, time cost: {time_cost}s')
            print(f'Min visitation: {min_visitation}')
            print(f'Max visitation: {max_visitation}')
            print(f'Visitation entropy: {visitation_entropy}/{max_entropy}')
        except Exception as e:
            print(f'Failed to plot visitation counts: {e}')

    def save_checkpoint(self, cosine_similarity=None):
        """Save model checkpoint."""
        # Create directory if it doesn't exist
        os.makedirs(f'./results/models/{self.env_name}', exist_ok=True)

        # Save latest model
        checkpoint_path = f'./results/models/{self.env_name}/last_{self._date_time}.pt'
        checkpoint = {
            'model_state_dict': self.gnn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dual_params': self.dual_params,
            'barrier_coef': self.barrier_coef,
            'dual_velocities': self.dual_velocities,
            'global_step': self._global_step,
            'best_cosine_similarity': self._best_cosine_similarity,
        }
        torch.save(checkpoint, checkpoint_path)

        # Save best model if cosine similarity improved
        should_save_best = (
            cosine_similarity is not None
            and cosine_similarity > self._best_cosine_similarity
        )
        if should_save_best:
            self._best_cosine_similarity = cosine_similarity
            best_path = f'./results/models/{self.env_name}/best_{self._date_time}.pt'
            torch.save(checkpoint, best_path)

            # Log best model to wandb
            if self.use_wandb:
                best_model = wandb.Artifact(
                    name='best_model',
                    type='model',
                    description=f'Best model found during training (CS: {cosine_similarity:.4f}).',
                )
                best_model.add_file(best_path)
                self.logger.log_artifact(best_model)

        # Log latest model to wandb
        if self.use_wandb:
            last_model = wandb.Artifact(
                name='last_model',
                type='model',
                description='Most recent model.',
            )
            last_model.add_file(checkpoint_path)
            self.logger.log_artifact(last_model)

    def get_states(self):
        """Get all states from the environment."""
        if not hasattr(self.env, 'get_states'):
            raise NotImplementedError("Environment doesn't support get_states method")

        state_dict = self.env.get_states()

        # Determine which key to use based on environment
        if 'xy_agent' in state_dict:
            return state_dict['xy_agent']
        elif 'pixels' in state_dict:
            return state_dict['pixels']
        elif 'grid' in state_dict:
            return state_dict['grid']
        else:
            raise ValueError('Unknown state format')

    def compute_cosine_similarity(self):
        """Compute cosine similarity with ground truth eigenvectors."""
        if not self.is_tabular or self.eigvec_dict is None:
            return 0.0, torch.zeros(self.d, device=self.device)

        # Get states
        states = self.get_states()
        states_tensor = torch.tensor(states, device=self.device).float()

        # Create a batch
        x = states_tensor

        # Construct dummy edges for the graph (fully connected)
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)

        # Get embeddings for all nodes
        self.gnn.eval()
        with torch.no_grad():
            approx_eigvec = self.gnn(x, edge_index)

        # Normalize approximated eigenvectors
        norms = torch.linalg.norm(approx_eigvec, dim=0, keepdim=True)
        approx_eigvec = approx_eigvec / norms.clamp(min=1e-10)

        # Permute the approximated eigenvectors
        approx_eigvec = approx_eigvec[:, self.past_permutation_array]

        # Convert eigvec_dict to tensor format
        unique_real_eigval = sorted(self.eigvec_dict.keys(), reverse=True)

        id_ = 0
        similarities = []
        for i, eigval in enumerate(unique_real_eigval):
            multiplicity = len(self.eigvec_dict[eigval])

            # Compute cosine similarity
            if multiplicity == 1:
                # Get eigenvectors associated with the current eigenvalue
                current_real_eigvec = (
                    self.eigvec_dict[eigval][0].to(self.device).float()
                )
                current_approx_eigvec = approx_eigvec[:, id_]

                # Compute cosine similarity
                pos_sim = torch.dot(current_real_eigvec, current_approx_eigvec)
                similarities.append(torch.maximum(pos_sim, -pos_sim).to('cpu').item())
            else:
                # Get eigenvectors associated with the current eigenvalue
                current_real_eigvec = torch.stack(
                    [
                        torch.tensor(np.array(v.cpu()), device=self.device).float()
                        for v in self.eigvec_dict[eigval]
                    ],
                    dim=1,
                )
                current_approx_eigvec = approx_eigvec[:, id_ : id_ + multiplicity]

                # Rotate approximated eigenvectors to match the space spanned by the real eigenvectors
                optimal_approx_eigvec = self.rotate_eigenvectors(
                    current_real_eigvec, current_approx_eigvec
                )

                norms = torch.linalg.norm(optimal_approx_eigvec, dim=0, keepdim=True)
                optimal_approx_eigvec = optimal_approx_eigvec / norms.clamp(min=1e-10)

                # Compute cosine similarity
                for j in range(multiplicity):
                    real = current_real_eigvec[:, j]
                    approx = optimal_approx_eigvec[:, j]
                    pos_sim = torch.dot(real, approx)
                    similarities.append(torch.maximum(pos_sim, -pos_sim).item())

            id_ += multiplicity

        # Convert to tensor
        similarities = torch.tensor(similarities, device=self.device)

        # Compute average cosine similarity
        cosine_similarity = similarities.mean().item()

        return cosine_similarity, similarities

    def compute_maximal_cosine_similarity(self, params_encoder=None):
        """Compute maximal cosine similarity with ground truth eigenvectors."""
        if not self.is_tabular or self.eigvec_dict is None:
            return 0.0, torch.zeros(self.d, device=self.device)

        # Get states
        states = self.get_states()
        states_tensor = torch.tensor(states, device=self.device).float()

        # Create a batch
        x = states_tensor.to(self.device)

        # Construct dummy edges for the graph
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)

        # Get embeddings for all nodes
        self.gnn.eval()
        with torch.no_grad():
            approx_eigvec = self.gnn(x, edge_index)

        norms = torch.linalg.norm(approx_eigvec, dim=0, keepdim=True)
        approx_eigvec = approx_eigvec / norms.clamp(min=1e-10)

        # Collect all real eigenvectors
        real_eigvec = []
        for eigval in sorted(self.eigvec_dict.keys(), reverse=True):
            for vec in self.eigvec_dict[eigval]:
                real_eigvec.append(
                    torch.tensor(np.array(vec.cpu()), device=self.device).float()
                )

        # Stack into a tensor
        real_eigvec = torch.stack(real_eigvec, dim=1)

        # Rotate approximated eigenvectors
        optimal_approx_eigvec = self.rotate_eigenvectors(real_eigvec, approx_eigvec)
        norms = torch.linalg.norm(optimal_approx_eigvec, dim=0, keepdim=True)
        optimal_approx_eigvec = optimal_approx_eigvec / norms.clamp(min=1e-10)

        # Compute cosine similarity
        similarities = []
        for j in range(self.d):
            real = real_eigvec[:, j]
            approx = optimal_approx_eigvec[:, j]
            pos_sim = torch.dot(real, approx)
            similarities.append(torch.maximum(pos_sim, -pos_sim).item())

        # Convert to tensor
        similarities = torch.tensor(similarities, device=self.device)

        # Compute average cosine similarity
        cosine_similarity = similarities.mean().item()

        return cosine_similarity, similarities

    def compute_cosine_similarity_simple(self):
        """Compute simple cosine similarity with ground truth eigenvectors."""
        if not self.is_tabular or self.eigvec_dict is None:
            return (
                0.0,
                torch.zeros(self.d, device=self.device),
                0.0,
                torch.zeros(self.d, device=self.device),
            )

        # Get states
        states = self.get_states()
        states_tensor = torch.tensor(states, device=self.device).float()

        # Create a batch
        x = states_tensor

        # Construct dummy edges for the graph (fully connected)
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)

        # Get embeddings for all nodes
        self.gnn.eval()
        with torch.no_grad():
            approx_eigvec = self.gnn(x, edge_index)

        # Normalize approximated eigenvectors
        norms = torch.linalg.norm(approx_eigvec, dim=0, keepdim=True)
        approx_eigvec = approx_eigvec / norms.clamp(min=1e-10)

        # Compute cosine similarities for both non-permuted and permuted versions

        # 1. Non-permuted version
        permuted_approx_eigvec = approx_eigvec[:, self.past_permutation_array]
        unique_real_eigval = sorted(self.eigvec_dict.keys(), reverse=True)

        id_ = 0
        similarities = []
        for i, eigval in enumerate(unique_real_eigval):
            multiplicity = len(self.eigvec_dict[eigval])

            if multiplicity == 1:
                current_real_eigvec = torch.tensor(
                    np.array(self.eigvec_dict[eigval][0].cpu()), device=self.device
                ).float()
                current_approx_eigvec = permuted_approx_eigvec[:, id_]

                pos_sim = torch.dot(current_real_eigvec, current_approx_eigvec)
                similarities.append(torch.maximum(pos_sim, -pos_sim).item())
            else:
                # Stack real eigenvectors
                current_real_eigvec = torch.stack(
                    [
                        torch.tensor(np.array(v.cpu()), device=self.device).float()
                        for v in self.eigvec_dict[eigval]
                    ],
                    dim=1,
                )
                current_approx_eigvec = permuted_approx_eigvec[
                    :, id_ : id_ + multiplicity
                ]

                # Compute projections
                projection_matrix = torch.einsum(
                    'ij,ik->jk', current_approx_eigvec, current_real_eigvec
                )

                # Compute generalized cosine similarity
                cos_sim = torch.sqrt(torch.sum(projection_matrix**2, dim=1))
                for similarity in cos_sim:
                    similarities.append(similarity.item())

            id_ += multiplicity

        # 2. Permuted version
        permuted_approx_eigvec = approx_eigvec[:, self.permutation_array]

        id_ = 0
        permuted_similarities = []
        for i, eigval in enumerate(unique_real_eigval):
            multiplicity = len(self.eigvec_dict[eigval])

            if multiplicity == 1:
                current_real_eigvec = torch.tensor(
                    np.array(self.eigvec_dict[eigval][0].cpu()), device=self.device
                ).float()
                current_approx_eigvec = permuted_approx_eigvec[:, id_]

                pos_sim = torch.dot(current_real_eigvec, current_approx_eigvec)
                permuted_similarities.append(torch.maximum(pos_sim, -pos_sim).item())
            else:
                # Stack real eigenvectors
                current_real_eigvec = torch.stack(
                    [
                        torch.tensor(np.array(v.cpu()), device=self.device).float()
                        for v in self.eigvec_dict[eigval]
                    ],
                    dim=1,
                )
                current_approx_eigvec = permuted_approx_eigvec[
                    :, id_ : id_ + multiplicity
                ]

                # Compute projections
                projection_matrix = torch.einsum(
                    'ij,ik->jk', current_approx_eigvec, current_real_eigvec
                )

                # Compute generalized cosine similarity
                cos_sim = torch.sqrt(torch.sum(projection_matrix**2, dim=1))
                for similarity in cos_sim:
                    permuted_similarities.append(similarity.item())

            id_ += multiplicity

        # Convert to tensors
        similarities = torch.tensor(similarities, device=self.device)
        permuted_similarities = torch.tensor(permuted_similarities, device=self.device)

        # Compute average cosine similarities
        cosine_similarity = similarities.mean().item()
        permuted_cosine_similarity = permuted_similarities.mean().item()

        return (
            cosine_similarity,
            similarities,
            permuted_cosine_similarity,
            permuted_similarities,
        )

    def reset_counters(self) -> None:
        """Reset step and log counters."""
        self.step_counter = 0
        self.log_counter = 0

    def update_counters(self) -> None:
        """Update step and log counters."""
        self.step_counter += 1
        self.log_counter = (self.log_counter + 1) % self.print_freq

    def rotate_eigenvectors(self, u_list, E):
        """Rotate eigenvectors to better match the target."""
        # Convert to torch tensors if they aren't already
        if not isinstance(u_list, torch.Tensor):
            u_list = torch.tensor(u_list, device=self.device).float()
        if not isinstance(E, torch.Tensor):
            E = torch.tensor(E, device=self.device).float()

        rotation_vectors = []

        # Compute first eigenvector
        u1 = u_list[:, 0].view(-1, 1)
        w1_times_lambda_1 = 0.5 * E.T @ u1
        w1 = w1_times_lambda_1 / torch.linalg.norm(w1_times_lambda_1).clamp(min=1e-10)
        rotation_vectors.append(w1)

        # Compute remaining eigenvectors
        for k in range(1, u_list.shape[1]):
            uk = u_list[:, k].view(-1, 1)
            Wk = torch.cat(rotation_vectors, dim=1)
            improper_wk = E.T @ uk
            bk = Wk.T @ improper_wk
            Ak = Wk.T @ Wk
            mu_k = torch.linalg.solve(Ak, bk)
            wk_times_lambda_k = 0.5 * (improper_wk - Wk @ mu_k)
            wk = wk_times_lambda_k / torch.linalg.norm(wk_times_lambda_k).clamp(
                min=1e-10
            )
            rotation_vectors.append(wk)

        # Use rotation vectors as columns of the optimal rotation matrix
        R = torch.cat(rotation_vectors, dim=1)

        # Obtain list of rotated eigenvectors
        rotated_eigvec = E @ R
        return rotated_eigvec

    def plot_eigenvectors(self):
        """Plot learned eigenvectors."""
        if not self.is_tabular:
            print("Environment doesn't support eigenvector visualization")
            return

        if not hasattr(self.env, 'get_states') or not hasattr(
            self.env.unwrapped, 'get_grid'
        ):
            print("Environment doesn't support eigenvector visualization")
            return

        # Get states
        states = self.get_states()
        states_tensor = torch.tensor(states, device=self.device).float()

        # Create a batch
        x = states_tensor

        # Construct dummy edges for the graph
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)

        # Get embeddings for all nodes
        self.gnn.eval()
        with torch.no_grad():
            approx_eigvec = self.gnn(x, edge_index)

        # Normalize eigenvectors
        norms = torch.linalg.norm(approx_eigvec, dim=0, keepdim=True)
        approx_eigvec = approx_eigvec / norms.clamp(min=1e-10)

        # Obtain sign of first non-zero element of eigenvectors
        first_non_zero_id = torch.argmax((approx_eigvec != 0).float(), dim=0)

        # Choose directions of eigenvectors
        signs = torch.sign(
            approx_eigvec[first_non_zero_id, torch.arange(approx_eigvec.shape[1])]
        )
        approx_eigvec = approx_eigvec * signs.reshape(1, -1)

        # Convert to numpy for plotting
        grid = self.env.unwrapped.get_grid().astype(bool)
        vmin = approx_eigvec.min().item()
        vmax = approx_eigvec.max().item()

        # Create directory for saving plots
        os.makedirs(f'./results/visuals/{self.env_name}', exist_ok=True)

        # Plot approximated eigenvectors
        for i in range(self.d):
            eigenvector = approx_eigvec[:, i].cpu().numpy()
            fig_path = self.plot_single_eigenvector(
                states, i, eigenvector, grid, vmin, vmax
            )

            # Log to wandb
            if self.use_wandb and self.log_eigenvectors:
                self.logger.log(
                    {
                        f'eigenvector_{i}': wandb.Image(
                            fig_path, caption=f'Eigenvector {i}'
                        )
                    }
                )

        print('Eigenvectors plotted.')

    def permute_representations(self, representations):
        """Permute entries in the second dimension of the representations."""
        permuted_representations = representations.clone()[:, self.permutation_array]
        return permuted_representations

    def plot_single_eigenvector(
        self, states, eigenvector_id, eigenvector, grid, vmin, vmax
    ):
        """Plot a single eigenvector."""
        # Obtain x, y, z coordinates, where z is the eigenvector value
        y = states[:, 0]
        x = states[:, 1]
        z = eigenvector

        # Calculate tile size
        x_num_tiles = np.unique(x).shape[0]
        x_tile_size = (np.max(x) - np.min(x)) / max(1, x_num_tiles - 1)
        y_num_tiles = np.unique(y).shape[0]
        y_tile_size = (np.max(y) - np.min(y)) / max(1, y_num_tiles - 1)

        # Create grid for interpolation
        ti_x = np.linspace(
            x.min() - x_tile_size, x.max() + x_tile_size, x_num_tiles + 2
        )
        ti_y = np.linspace(
            y.min() - y_tile_size, y.max() + y_tile_size, y_num_tiles + 2
        )
        XI, YI = np.meshgrid(ti_x, ti_y)

        # Interpolate
        rbf = Rbf(x, y, z, function='cubic')
        ZI = rbf(XI, YI)
        ZI_bounds = 85 * np.ma.masked_where(grid, np.ones_like(ZI))
        ZI_free = np.ma.masked_where(~grid, ZI)

        # Generate color mesh
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        mesh = ax.pcolormesh(
            XI, YI, ZI_free, shading='auto', cmap='coolwarm', vmin=vmin, vmax=vmax
        )
        ax.pcolormesh(XI, YI, ZI_bounds, shading='auto', cmap='Greys', vmin=0, vmax=255)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(mesh, ax=ax, shrink=0.5, pad=0.05)

        # Set title
        ax.set_title(f'Eigenvector {eigenvector_id}')

        # Save figure
        fig_path = f'./results/visuals/{self.env_name}/learned_eigenvector_{eigenvector_id}_{self._date_time}.pdf'

        if not os.path.exists(os.path.dirname(fig_path)):
            os.makedirs(os.path.dirname(fig_path))

        plt.savefig(
            fig_path,
            bbox_inches='tight',
            dpi=300,
            transparent=True,
        )

        # Create a PNG version for wandb
        png_path = f'./results/visuals/{self.env_name}/learned_eigenvector_{eigenvector_id}_{self._date_time}.png'
        plt.savefig(
            png_path,
            bbox_inches='tight',
            dpi=100,
        )

        plt.close(fig)

        return png_path
