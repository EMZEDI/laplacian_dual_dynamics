import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product


class ALLOLoss(nn.Module):
    def __init__(
        self,
        d: int,
        use_barrier_normalization: bool = True,
        lr_barrier_coefs: float = 1e-3,
        min_barrier_coefs: float = 1e-5,
        max_barrier_coefs: float = 1e5,
        lr_duals: float = 1e-3,
        lr_dual_velocities: float = 1e-3,
        min_duals: float = -1e5,
        max_duals: float = 1e5,
        use_barrier_for_duals: float = 1.0,
        device=None,
    ):
        """A PyTorch-style loss function for augmented Lagrangian optimization of graph embeddings.

        Args:
            d (int): The number of representation dimensions (e.g., eigenfunctions).
            use_barrier_normalization (bool): Whether to normalize the total loss by the barrier coefficient.
            lr_barrier_coefs (float): Learning rate for updating barrier coefficients.
            min_barrier_coefs (float): Lower bound for barrier coefficients.
            max_barrier_coefs (float): Upper bound for barrier coefficients.
            lr_duals (float): Learning rate for updating dual variables.
            lr_dual_velocities (float): Learning rate for smoothing dual updates.
            min_duals (float): Lower bound for dual variables.
            max_duals (float): Upper bound for dual variables.
            use_barrier_for_duals (float): Scalar factor to modulate the barrier coefficient's influence in dual updates.
            device: Device to create tensors on (defaults to None, which uses current device)
        """
        super().__init__()
        self.d = d
        self.coefficient_vector = torch.ones(d, device=device)
        self.use_barrier_normalization = use_barrier_normalization
        self.lr_barrier_coefs = lr_barrier_coefs
        self.min_barrier_coefs = min_barrier_coefs
        self.max_barrier_coefs = max_barrier_coefs
        self.lr_duals = lr_duals
        self.lr_dual_velocities = lr_dual_velocities
        self.min_duals = min_duals
        self.max_duals = max_duals
        self.use_barrier_for_duals = use_barrier_for_duals

        # For metrics tracking
        self.metrics = {}
        # For storing error matrices for updates
        self.error_matrix = None
        self.quadratic_error_matrix = None

    def compute_graph_drawing_loss(
        self, start_representation: torch.Tensor, end_representation: torch.Tensor
    ):
        """Compute the graph drawing loss as the weighted distance between start and end representations."""
        # Compute mean squared differences per dimension
        graph_induced_norms = ((start_representation - end_representation) ** 2).mean(0)
        # Dot with the coefficient vector to weight the contributions
        loss = torch.dot(graph_induced_norms, self.coefficient_vector)

        # Store metrics for later retrieval
        for i in range(self.d):
            self.metrics[f'graph_norm({i})'] = graph_induced_norms[i].item()

        return loss

    def compute_orthogonality_error_matrix(
        self, representation_batch_1: torch.Tensor, representation_batch_2: torch.Tensor
    ):
        """Compute the error matrix enforcing orthonormality constraints."""
        n = representation_batch_1.shape[0]
        # Compute inner products with one side detached
        inner_product_matrix_1 = (
            torch.einsum(
                'ij,ik->jk', representation_batch_1, representation_batch_1.detach()
            )
            / n
        )
        inner_product_matrix_2 = (
            torch.einsum(
                'ij,ik->jk', representation_batch_2, representation_batch_2.detach()
            )
            / n
        )

        # Enforce constraints on the lower triangular part (including diagonal)
        error_matrix_1 = torch.tril(
            inner_product_matrix_1
            - torch.eye(self.d, device=representation_batch_1.device)
        )
        error_matrix_2 = torch.tril(
            inner_product_matrix_2
            - torch.eye(self.d, device=representation_batch_1.device)
        )
        error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)
        quadratic_error_matrix = error_matrix_1 * error_matrix_2

        # Store inner products for metrics
        for i, j in product(range(self.d), range(self.d)):
            if i >= j:
                self.metrics[f'inner({i},{j})'] = inner_product_matrix_1[i, j].item()

        # Save error matrices for later use in update methods
        self.error_matrix = error_matrix
        self.quadratic_error_matrix = quadratic_error_matrix

        return error_matrix, quadratic_error_matrix

    def compute_orthogonality_loss(
        self, error_matrix, quadratic_error_matrix, duals, barrier_coef
    ):
        """Compute dual and barrier losses for orthogonality constraints."""
        # Use .detach() to stop gradients flowing back to dual variables and barrier coefficient
        dual_loss = (duals.detach() * error_matrix).sum()
        quadratic_error = quadratic_error_matrix.sum()
        barrier_loss = barrier_coef.detach() * quadratic_error

        # Store for metrics
        for i, j in product(range(self.d), range(self.d)):
            if i >= j:
                self.metrics[f'beta({i},{j})'] = duals[i, j].item()
        self.metrics['barrier_coeff'] = barrier_coef.item()

        return dual_loss, barrier_loss

    def update_barrier_coefficient(self, barrier_coef):
        """Update barrier coefficient using the quadratic error matrix."""
        if self.quadratic_error_matrix is None:
            raise ValueError('Must call forward before updating barrier coefficient')

        # Clip negative values to 0 and take the mean
        updates = torch.clamp(self.quadratic_error_matrix, min=0).mean()
        updated_barrier_coef = barrier_coef + self.lr_barrier_coefs * updates
        updated_barrier_coef = torch.clamp(
            updated_barrier_coef, min=self.min_barrier_coefs, max=self.max_barrier_coefs
        )

        return updated_barrier_coef

    def update_duals(self, duals, dual_velocities, barrier_coef):
        """Update dual variables using the error matrix."""
        if self.error_matrix is None:
            raise ValueError('Must call forward before updating duals')

        # Only update the lower triangular part
        updates = torch.tril(self.error_matrix)

        # Adjust learning rate using barrier coefficient
        adjusted_barrier = 1 + self.use_barrier_for_duals * (barrier_coef - 1)
        lr = self.lr_duals * adjusted_barrier

        # Update duals
        updated_duals = duals + lr * updates
        updated_duals = torch.clamp(
            updated_duals, min=self.min_duals, max=self.max_duals
        )
        updated_duals = torch.tril(updated_duals)

        # Update dual velocities
        dual_update = updated_duals - duals
        norm_dual_velocities = dual_velocities.norm()
        init_coeff = torch.isclose(
            norm_dual_velocities,
            torch.tensor(0.0, device=dual_velocities.device),
            rtol=1e-10,
            atol=1e-13,
        ).float()
        update_rate = init_coeff + (1 - init_coeff) * self.lr_dual_velocities
        updated_dual_velocities = dual_velocities + update_rate * (
            dual_update - dual_velocities
        )

        return updated_duals, updated_dual_velocities

    def forward(
        self,
        start_representation: torch.Tensor,
        end_representation: torch.Tensor,
        constraint_representation_1: torch.Tensor,
        constraint_representation_2: torch.Tensor,
        duals: torch.Tensor = None,
        barrier_coef: torch.Tensor = None,
    ):
        """Standard PyTorch-style forward pass for computing the ALLO loss.

        Args:
            start_representation: Tensor of shape [batch_size, d]
            end_representation: Tensor of shape [batch_size, d]
            constraint_representation_1: Tensor for orthogonality constraints [batch_size, d]
            constraint_representation_2: Tensor for orthogonality constraints [batch_size, d]
            duals: Dual variables tensor of shape [d, d] (lower triangular)
            barrier_coef: Scalar barrier coefficient

        Returns:
            loss: Scalar loss value
        """
        # Reset metrics dictionary
        self.metrics = {}
        device = start_representation.device

        # Initialize parameters if not provided
        if duals is None:
            duals = torch.zeros((self.d, self.d), device=device)

        if barrier_coef is None:
            barrier_coef = torch.tensor(1.0, device=device)

        # Compute graph drawing loss
        graph_loss = self.compute_graph_drawing_loss(
            start_representation, end_representation
        )

        # Compute orthogonality constraints
        error_matrix, quadratic_error_matrix = self.compute_orthogonality_error_matrix(
            constraint_representation_1, constraint_representation_2
        )

        # Compute dual and barrier losses
        dual_loss, barrier_loss = self.compute_orthogonality_loss(
            error_matrix, quadratic_error_matrix, duals, barrier_coef
        )

        # Total loss
        lagrangian = graph_loss + dual_loss + barrier_loss
        loss = lagrangian

        if self.use_barrier_normalization:
            loss = loss / barrier_coef.detach()

        # Store high-level metrics
        self.metrics['train_loss'] = loss.item()
        self.metrics['graph_loss'] = graph_loss.item()
        self.metrics['dual_loss'] = dual_loss.item()
        self.metrics['barrier_loss'] = barrier_loss.item()

        return loss

    def get_metrics(self):
        """Return the computed metrics from the last forward pass."""
        return self.metrics


# Example usage:
if __name__ == '__main__':
    # Define a model that produces the required representations
    class DummyEncoder(nn.Module):
        def __init__(self, input_dim, d):
            super().__init__()
            self.fc = nn.Linear(input_dim, d)

        def forward(self, x):
            rep = self.fc(x)
            # For this example, we simply duplicate the representation
            return rep, rep * 0.5, rep, rep * 0.5

    # Setup
    d = 5
    input_dim = 10
    batch_size = 32

    # Instantiate our loss module and encoder
    allo_loss = ALLOLoss(d=d)
    encoder = DummyEncoder(input_dim=input_dim, d=d)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

    # Create parameters for augmented Lagrangian optimization
    duals = torch.zeros((d, d))
    barrier_coef = torch.tensor(1.0)
    dual_velocities = torch.zeros((d, d))

    # Create a dummy training batch
    train_batch = torch.randn(batch_size, input_dim)

    # Training loop example
    optimizer.zero_grad()

    # Get representations from encoder
    start_rep, end_rep, constraint_rep1, constraint_rep2 = encoder(train_batch)

    # Compute loss
    loss = allo_loss(
        start_rep,
        end_rep,
        constraint_rep1,
        constraint_rep2,
        duals=duals,
        barrier_coef=barrier_coef,
    )

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update augmented Lagrangian parameters
    with torch.no_grad():
        barrier_coef = allo_loss.update_barrier_coefficient(barrier_coef)
        duals, dual_velocities = allo_loss.update_duals(
            duals, dual_velocities, barrier_coef
        )

    # Get metrics
    metrics = allo_loss.get_metrics()
    print('Loss:', loss.item())
    print('Metrics:', metrics)
